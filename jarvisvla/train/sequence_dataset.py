"""
Sequence Dataset for Stateful VLA Training on OpenVPT Data

This module provides:
1. VPTSequenceDataset: Returns long sequences of consecutive frames from /data/vvm33/vpt_contractor
2. InventoryTextEncoder: Frozen BERT encoder for inventory descriptions

Data Format (OpenVPT /data/vvm33/vpt_contractor):
    - MP4 files: Gameplay footage
    - JSONL files: Per-tick data with inventory, actions, etc.
    
    JSONL format per tick:
    {
        "tick": int,
        "inventory": [{"type": "oak_log", "quantity": 1}, ...],
        "mouse": {...},
        "keyboard": {...},
        "isGuiOpen": bool,
        "isGuiInventory": bool,
        ...
    }

BERT Encoding:
    - Converts inventory to canonical text string:
      "oak_log count:1 | cobblestone count:64"
    - Only lists non-empty slots
    - Encoded with frozen BERT-base-uncased (768-dim)
    - BERT can embed any text, so new items or NBT data can be described.
      The frozen encoder ensures a stable target space; later we may replace
      the MLP head with structured prediction to better handle novel items.
      The JarvisVLA paper uses a similar frozen encoder approach for some tasks.
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np
from PIL import Image
import cv2
from transformers import AutoTokenizer, AutoModel


class InventoryTextEncoder:
    """
    Frozen BERT encoder for inventory descriptions.
    
    Encodes canonical inventory strings like:
        "oak_log count:1 | cobblestone count:64 | diamond_pickaxe count:1"
        "empty inventory" (when no items)
    
    into 768-dimensional normalized embeddings.
    
    BERT can embed any text, so new items or NBT data can be described in the
    inventory string. The frozen encoder ensures a stable target space; later we
    may replace the MLP head with structured prediction to better handle novel
    items. The JarvisVLA paper uses a similar frozen encoder approach for some tasks.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Load frozen encoder
        print(f"Loading BERT encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"  BERT loaded: {embedding_dim}d embeddings on {device}")
    
    def _inventory_to_canonical_string(self, inventory_list: List[Dict]) -> str:
        """
        Convert inventory list to canonical text string.
        
        Format: "item_name count:N | item_name count:N | ..."
        Only non-empty slots are listed.
        
        Args:
            inventory_list: List of {type, quantity} dicts
        
        Returns:
            Canonical string representation
        """
        items_desc = []
        
        for item in inventory_list:
            item_type = item.get('type', '')
            quantity = item.get('quantity', 0)
            
            # Skip empty/air items
            if item_type and item_type != 'air' and quantity > 0:
                # Clean up item name (remove namespace if present)
                if ':' in item_type:
                    item_type = item_type.split(':')[-1]
                
                items_desc.append(f"{item_type} count:{quantity}")
        
        if items_desc:
            return " | ".join(items_desc)
        else:
            return "empty inventory"
    
    @torch.no_grad()
    def encode_inventory(
        self,
        inventory_list: List[Dict],
    ) -> torch.Tensor:
        """
        Encode inventory list to 768-dim embedding.
        
        Args:
            inventory_list: List of {type, quantity} dicts
        
        Returns:
            embedding: [768] normalized embedding
        """
        # Convert to canonical string
        text = self._inventory_to_canonical_string(inventory_list)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,  # Should be enough for inventory descriptions
        ).to(self.device)
        
        # Encode
        outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[0, 0, :]  # [768]
        
        # Normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    @torch.no_grad()
    def encode_batch(
        self,
        inventory_lists: List[List[Dict]],
    ) -> torch.Tensor:
        """
        Encode a batch of inventories efficiently.
        
        Args:
            inventory_lists: List of inventory lists
        
        Returns:
            embeddings: [batch, 768]
        """
        # Convert all to strings
        texts = [self._inventory_to_canonical_string(inv) for inv in inventory_lists]
        
        # Batch tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        
        # Encode
        outputs = self.model(**inputs)
        
        # Extract [CLS] embeddings
        embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class VPTSequenceDataset(IterableDataset):
    """
    Iterable dataset for OpenVPT sequences from /data/vvm33/vpt_contractor.
    
    Returns sequences of consecutive frames with BERT-encoded inventory targets.
    
    Each sample contains:
    - frames: List of PIL Images [sequence_length]
    - texts: List of text prompts [sequence_length]
    - inventory_embeddings: [sequence_length, 768] - BERT-encoded per-tick inventory
    
    Args:
        data_dir: Directory containing VPT data (default: /data/vvm33/vpt_contractor)
        inventory_encoder: InventoryTextEncoder instance
        sequence_length: Number of consecutive frames per sequence
        stride: Stride between sequences (for overlapping)
        frame_size: Size to resize frames to
        max_sequences: Max sequences to load (for testing)
    """
    
    DEFAULT_DATA_DIR = "/data/vvm33/vpt_contractor"
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        inventory_encoder: Optional[InventoryTextEncoder] = None,
        sequence_length: int = 200,
        stride: int = 100,
        frame_size: Tuple[int, int] = (224, 224),
        max_sequences: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir or self.DEFAULT_DATA_DIR)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        self.inventory_encoder = inventory_encoder
        self.sequence_length = sequence_length
        self.stride = stride
        self.frame_size = frame_size
        self.max_sequences = max_sequences
        
        # Build index of all valid sequences
        self.sequences = self._build_sequence_index()
        
        print(f"VPTSequenceDataset loaded:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Stride: {stride}")
    
    def _build_sequence_index(self) -> List[Dict]:
        """Build index of all valid sequences from JSONL files."""
        sequences = []
        
        # Find all JSONL files
        jsonl_files = sorted(self.data_dir.glob("*.jsonl"))
        
        print(f"Found {len(jsonl_files)} JSONL files")
        
        for jsonl_file in jsonl_files:
            # Find corresponding MP4 file
            mp4_file = self.data_dir / f"{jsonl_file.stem}.mp4"
            
            if not mp4_file.exists():
                print(f"Warning: No MP4 found for {jsonl_file.name}, skipping")
                continue
            
            # Read all ticks from JSONL
            ticks = []
            try:
                with open(jsonl_file, 'r') as f:
                    for line_num, line in enumerate(f):
                        try:
                            data = json.loads(line)
                            tick_data = {
                                'tick': data.get('tick', line_num),
                                'inventory': data.get('inventory', []),
                                'mouse': data.get('mouse', {}),
                                'keyboard': data.get('keyboard', {}),
                                'isGuiOpen': data.get('isGuiOpen', False),
                                'isGuiInventory': data.get('isGuiInventory', False),
                                'hotbar': data.get('hotbar', 0),
                                'yaw': data.get('yaw', 0.0),
                                'pitch': data.get('pitch', 0.0),
                            }
                            ticks.append(tick_data)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Warning: Error reading {jsonl_file}: {e}")
                continue
            
            # Create sequences from ticks
            if len(ticks) >= self.sequence_length:
                for start_idx in range(0, len(ticks) - self.sequence_length + 1, self.stride):
                    sequences.append({
                        'video_path': str(mp4_file),
                        'jsonl_path': str(jsonl_file),
                        'start_idx': start_idx,
                        'length': self.sequence_length,
                        'ticks': ticks[start_idx:start_idx + self.sequence_length],
                    })
                    
                    if self.max_sequences and len(sequences) >= self.max_sequences:
                        break
            
            if self.max_sequences and len(sequences) >= self.max_sequences:
                break
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __iter__(self) -> Iterator[Dict]:
        """
        Iterate over sequences.
        
        Yields:
            Dictionary with:
                - frames: List of PIL Images [sequence_length]
                - texts: List of strings [sequence_length]
                - inventory_embeddings: [sequence_length, 768] BERT-encoded
                - actions: List of action dicts [sequence_length]
        """
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process loading
            sequences = self.sequences
        else:
            # Multi-worker: split sequences among workers
            per_worker = len(self.sequences) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.sequences)
            sequences = self.sequences[start:end]
        
        for seq_info in sequences:
            yield self._load_sequence(seq_info)
    
    def _load_sequence(self, seq_info: Dict) -> Dict:
        """Load a single sequence."""
        video_path = seq_info['video_path']
        ticks = seq_info['ticks']
        
        # Load frames from video
        frames = self._load_video_frames(video_path, ticks)
        
        # Generate text prompts
        texts = self._generate_texts(ticks)
        
        # Encode inventory with BERT
        if self.inventory_encoder is not None:
            inventory_embeddings = []
            for tick_data in ticks:
                inv_emb = self.inventory_encoder.encode_inventory(tick_data['inventory'])
                inventory_embeddings.append(inv_emb)
            inventory_embeddings = torch.stack(inventory_embeddings)  # [seq_len, 768]
        else:
            # Fallback: dummy embeddings
            inventory_embeddings = torch.zeros(len(ticks), 768)
        
        # Extract actions
        actions = [tick_data['mouse'] for tick_data in ticks]
        
        return {
            'frames': frames,
            'texts': texts,
            'inventory_embeddings': inventory_embeddings,
            'actions': actions,
            'video_path': video_path,
            'start_tick': ticks[0]['tick'],
        }
    
    def _load_video_frames(
        self,
        video_path: str,
        ticks: List[Dict],
    ) -> List[Image.Image]:
        """Load frames for a sequence of ticks."""
        frames = []
        
        if not Path(video_path).exists():
            # Return dummy frames
            return [Image.new('RGB', self.frame_size, color='black')] * len(ticks)
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return [Image.new('RGB', self.frame_size, color='black')] * len(ticks)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for tick_data in ticks:
                frame_idx = tick_data['tick']
                
                # Clamp to valid range
                if frame_idx >= total_frames:
                    frame_idx = total_frames - 1
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize
                    frame = cv2.resize(frame, self.frame_size)
                    # Convert to PIL
                    pil_image = Image.fromarray(frame)
                    frames.append(pil_image)
                else:
                    frames.append(Image.new('RGB', self.frame_size, color='black'))
            
            cap.release()
            
        except Exception as e:
            print(f"Warning: Error loading video {video_path}: {e}")
            frames = [Image.new('RGB', self.frame_size, color='black')] * len(ticks)
        
        return frames
    
    def _generate_texts(self, ticks: List[Dict]) -> List[str]:
        """Generate text prompts for each tick."""
        texts = []
        
        for tick_data in ticks:
            gui_open = tick_data.get('isGuiOpen', False)
            gui_inventory = tick_data.get('isGuiInventory', False)
            inventory = tick_data.get('inventory', [])
            
            if gui_inventory:
                text = f"Inventory screen open with {len(inventory)} items"
            elif gui_open:
                text = "GUI screen is open"
            else:
                text = "Continue playing Minecraft based on current view"
            
            texts.append(text)
        
        return texts


class VPTSequenceDataCollator:
    """
    Data collator for sequence batches.
    
    Converts sequences into tensors suitable for the StatefulVLATrainer.
    """
    
    def __init__(
        self,
        processor,
        max_seq_length: int = 512,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of sequences.
        
        Args:
            examples: List of sequence dicts from VPTSequenceDataset
        
        Returns:
            Batched tensors:
            - pixel_values: [batch, seq_len, ...]
            - input_ids: [batch, seq_len, max_seq_length]
            - labels: [batch, seq_len, max_seq_length]
            - inventory_embeddings: [batch, seq_len, 768]
            - attention_mask: [batch, seq_len, max_seq_length]
        """
        batch_size = len(examples)
        seq_len = len(examples[0]['frames'])
        
        all_pixel_values = []
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_inventory_embeddings = []
        
        for example in examples:
            frames = example['frames']
            texts = example['texts']
            inventory_embs = example['inventory_embeddings']  # [seq_len, 768]
            
            seq_pixel_values = []
            seq_input_ids = []
            seq_attention_masks = []
            seq_labels = []
            
            for frame, text in zip(frames, texts):
                # Process with processor
                processed = self.processor(
                    text=text,
                    images=frame,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=self.max_seq_length,
                    truncation=True,
                )
                
                if 'pixel_values' in processed:
                    seq_pixel_values.append(processed['pixel_values'])
                seq_input_ids.append(processed['input_ids'][0])
                seq_attention_masks.append(processed['attention_mask'][0])
                
                # Labels for language modeling
                labels = processed['input_ids'][0].clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                seq_labels.append(labels)
            
            # Stack sequence
            if seq_pixel_values:
                # Stack pixel values: [seq_len, num_images, C, H, W]
                all_pixel_values.append(torch.cat(seq_pixel_values, dim=0))
            all_input_ids.append(torch.stack(seq_input_ids))
            all_attention_masks.append(torch.stack(seq_attention_masks))
            all_labels.append(torch.stack(seq_labels))
            all_inventory_embeddings.append(inventory_embs)
        
        # Batch
        batch = {
            'input_ids': torch.stack(all_input_ids),  # [batch, seq_len, text_len]
            'attention_mask': torch.stack(all_attention_masks),  # [batch, seq_len, text_len]
            'labels': torch.stack(all_labels),  # [batch, seq_len, text_len]
            'inventory_embeddings': torch.stack(all_inventory_embeddings),  # [batch, seq_len, 768]
        }
        
        if all_pixel_values:
            batch['pixel_values'] = torch.stack(all_pixel_values)
        
        return batch


def create_sequence_dataloaders(
    data_dir: str = "/data/vvm33/vpt_contractor",
    processor=None,
    inventory_encoder: Optional[InventoryTextEncoder] = None,
    sequence_length: int = 200,
    batch_size: int = 2,
    num_workers: int = 4,
    max_sequences: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create train and validation dataloaders for sequence training.
    
    Args:
        data_dir: Directory containing VPT data
        processor: VLM processor
        inventory_encoder: Text encoder for inventory
        sequence_length: Frames per sequence
        batch_size: Batch size (number of sequences)
        num_workers: Number of data loading workers
        max_sequences: Max sequences to load (for testing)
    
    Returns:
        train_loader, val_loader (val_loader is None for now)
    """
    # Create dataset
    dataset = VPTSequenceDataset(
        data_dir=data_dir,
        inventory_encoder=inventory_encoder,
        sequence_length=sequence_length,
        max_sequences=max_sequences,
    )
    
    # Create collator
    collator = VPTSequenceDataCollator(processor=processor)
    
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
    )
    
    return train_loader, None
