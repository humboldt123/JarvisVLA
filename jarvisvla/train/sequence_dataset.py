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

Item Encoding:
    - Each inventory slot is encoded independently via character trigrams (768-dim).
    - Trigram embeddings are far more discriminative than BERT for short Minecraft
      item names, and are open-vocabulary (modded items work automatically).
    - The embedding matrix is fixed (seeded random, never trained).
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
try:
    cv2.setLogLevel(0)  # suppress ffmpeg "moov atom not found" spam from truncated VPT files
except AttributeError:
    pass  # some cv2 builds don't expose setLogLevel
import hashlib


class InventoryTextEncoder:
    """
    Character-trigram encoder for Minecraft item type names.

    Replaces the previous BERT encoder. BERT collapsed short domain-specific
    strings ('oak_log', 'stone_pickaxe') into nearly identical CLS embeddings.
    Character trigrams are far more discriminative for item identifiers:
        'oak_log' and 'birch_log' share '_lo', 'log' → similar vectors (correct)
        'oak_log' and 'diamond_pickaxe' share almost no trigrams → distant vectors

    Open-vocabulary: any new item name (modded items) decomposes into trigrams
    automatically — no retraining or fixed vocabulary needed.

    The embedding matrix is fixed (seeded random, never trained), giving a
    stable target space for the InfoNCE loss.
    """

    EMBED_DIM   = 768
    NUM_BUCKETS = 8192  # trigram hash buckets
    SEED        = 42

    def __init__(
        self,
        embedding_dim: int = 768,
        device: str = "cpu",   # kept for API compatibility, unused
    ):
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng(self.SEED)
        mat = rng.standard_normal((self.NUM_BUCKETS, embedding_dim)).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / (norms + 1e-8)   # [NUM_BUCKETS, EMBED_DIM], row-normalised

    # ------------------------------------------------------------------
    # Core encoder
    # ------------------------------------------------------------------

    def _trigram_hash(self, s: str) -> int:
        """Deterministic bucket index (not affected by PYTHONHASHSEED)."""
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % self.NUM_BUCKETS

    def _name_to_embedding(self, name: str) -> np.ndarray:
        """Map an item type name to a 768-dim unit vector via character trigrams."""
        name = name.lower()
        trigrams = [name[i:i+3] for i in range(len(name) - 2)]
        if not trigrams:
            trigrams = [name]   # very short names (1-2 chars) use the whole string
        indices = [self._trigram_hash(t) for t in trigrams]
        vecs = self._matrix[indices]            # [n_trigrams, EMBED_DIM]
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        return (mean_vec / (norm + 1e-8)).astype(np.float32)

    # ------------------------------------------------------------------
    # Slot helpers
    # ------------------------------------------------------------------

    def _slot_to_type_text(self, slot_data: Dict) -> str:
        """Return just the item type name for BERT encoding, e.g. 'oak_log' or 'empty slot'.

        Separating type from count lets BERT distinguish item names cleanly:
        'granite count:11' vs 'stone_pickaxe count:1' collapse to nearly identical
        BERT CLS embeddings; 'granite' vs 'stone_pickaxe' do not.
        """
        item_type = slot_data.get('type', '')
        quantity = slot_data.get('quantity', 0)
        if item_type and item_type != 'air' and quantity > 0:
            if ':' in item_type:
                item_type = item_type.split(':')[-1]
            return item_type
        return "empty slot"

    def _slot_to_count(self, slot_data: Dict) -> int:
        """Return the item stack count (0 for empty/air slots)."""
        item_type = slot_data.get('type', '')
        quantity = slot_data.get('quantity', 0)
        if item_type and item_type != 'air' and quantity > 0:
            return int(quantity)
        return 0

    def _slot_to_nbt_text(self, slot_data: Dict) -> str:
        """Return a canonical text string encoding any NBT metadata.

        Currently returns 'no_nbt' — placeholder until the richer data-collection
        mod provides NBT.

        # --- Future NBT implementation ---
        # The mod will supply an 'nbt' dict in the JSONL tick, e.g.:
        #   {"type": "diamond_pickaxe", "quantity": 1,
        #    "nbt": {"enchantments": {"sharpness": 2, "unbreaking": 3},
        #            "damage": 150}}
        #
        # Strategy: recursively flatten the nbt dict into sorted key=value pairs
        # separated by spaces, e.g. "enchantments.sharpness=2
        # enchantments.unbreaking=3 damage=150".  Sorting by key makes identical
        # NBT always produce identical strings regardless of insertion order.
        # BERT-encode this string → 768-dim → InfoNCE loss against other NBT
        # strings in the batch, identical to the type head.
        #
        # Items with no NBT all map to "no_nbt" (same BERT embedding) so the
        # nbt_head learns to distinguish vanilla vs enchanted items without
        # needing to discriminate between two plain items.  New enchant types or
        # arbitrary mod NBT fields appear as new BERT tokens automatically.
        #
        # def _flatten_nbt(nbt: dict, prefix: str = "") -> List[str]:
        #     pairs = []
        #     for k, v in sorted(nbt.items()):
        #         key = f"{prefix}.{k}" if prefix else k
        #         if isinstance(v, dict):
        #             pairs.extend(_flatten_nbt(v, key))
        #         else:
        #             pairs.append(f"{key}={v}")
        #     return pairs
        #
        # nbt = slot_data.get('nbt', {})
        # if nbt:
        #     return ' '.join(_flatten_nbt(nbt))
        """
        return "no_nbt"

    @staticmethod
    def inventory_has_items(inventory_list: List[Dict]) -> bool:
        """Return True if any slot contains a non-air item."""
        return any(
            item.get('type', '') not in ('', 'air') and item.get('quantity', 0) > 0
            for item in inventory_list
        )

    def encode_inventory_per_slot(
        self,
        inventory_list: List[Dict],
        num_slots: int = 36,
    ) -> torch.Tensor:
        """
        Encode each inventory slot to a 768-dim unit vector via character trigrams.

        Returns [num_slots, 768]. Slot assignment uses the 'slot' field if
        present, else list order.

        Args:
            inventory_list: List of {type, quantity, slot?, ...} dicts
            num_slots: Number of inventory slots (default: 36 for Minecraft)

        Returns:
            embeddings: [num_slots, 768] normalised per-slot embeddings
        """
        slot_names = ["empty slot"] * num_slots
        for i, item in enumerate(inventory_list):
            slot_idx = item.get('slot', i)
            if isinstance(slot_idx, int) and 0 <= slot_idx < num_slots:
                slot_names[slot_idx] = self._slot_to_type_text(item)
            elif i < num_slots:
                slot_names[i] = self._slot_to_type_text(item)
        vecs = np.stack([self._name_to_embedding(n) for n in slot_names])  # [num_slots, 768]
        return torch.from_numpy(vecs)

    def encode_inventory_counts(
        self,
        inventory_list: List[Dict],
        num_slots: int = 36,
    ) -> torch.Tensor:
        """Return a LongTensor [num_slots] of item counts (0 for empty slots)."""
        slot_counts = torch.zeros(num_slots, dtype=torch.long)
        for i, item in enumerate(inventory_list):
            slot_idx = item.get('slot', i)
            count = self._slot_to_count(item)
            if count > 0:
                if isinstance(slot_idx, int) and 0 <= slot_idx < num_slots:
                    slot_counts[slot_idx] = count
                elif i < num_slots:
                    slot_counts[i] = count
        return slot_counts


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
        
        # Encode inventory with BERT — per-slot: [seq_len, 36, 768]
        if self.inventory_encoder is not None:
            inventory_embeddings = torch.stack([
                self.inventory_encoder.encode_inventory_per_slot(t['inventory'])
                for t in ticks
            ])
            inventory_has_items = [
                InventoryTextEncoder.inventory_has_items(t['inventory']) for t in ticks
            ]
        else:
            inventory_embeddings = torch.zeros(len(ticks), 36, 768)
            inventory_has_items = [False] * len(ticks)

        # Extract actions
        actions = [tick_data['mouse'] for tick_data in ticks]

        return {
            'frames': frames,
            'texts': texts,
            'inventory_embeddings': inventory_embeddings,
            'inventory_has_items': inventory_has_items,
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
            - inventory_embeddings: [batch, seq_len, 36, 768]
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


class OnDemandSequenceLoader:
    """
    Loads VPT sequences on-demand to avoid OOM during training.

    Builds a lightweight index over JSONL/MP4 pairs — only line counts,
    no frames or embeddings in memory.  Sequences are loaded from disk on
    __getitem__ and kept in a small LRU cache.

    Each returned sequence dict contains:
        frames: List[PIL.Image]  (length = sequence_length)
        inventory_embeddings: [seq_len, 36, 768]  per-slot BERT targets
        inventory_has_items: List[bool]  cheap non-empty flag (no BERT needed at train time)
        ticks: List[dict]  raw JSONL tick data
        source_file: str
    """

    def __init__(
        self,
        jsonl_files: List[Path],
        sequence_length: int,
        encoder: InventoryTextEncoder,
        cache_size: int = 10,
    ):
        self.jsonl_files = jsonl_files
        self.sequence_length = sequence_length
        self.encoder = encoder
        self.cache: Dict = {}
        self.cache_size = cache_size
        # index: list of (jsonl_idx, num_sequences_in_file)
        self.index: List[Tuple[int, int]] = []

        for jsonl_idx, jsonl_file in enumerate(jsonl_files):
            mp4_file = jsonl_file.with_suffix('.mp4')
            if not mp4_file.exists():
                continue
            tick_count = 0
            with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in f:
                    tick_count += 1
            num_seqs = max(0, (tick_count - sequence_length) // sequence_length + 1)
            if num_seqs > 0:
                self.index.append((jsonl_idx, num_seqs))

    def __len__(self) -> int:
        return sum(count for _, count in self.index)

    def __getitem__(self, idx: int) -> Dict:
        if idx in self.cache:
            return self.cache[idx]
        current = 0
        for jsonl_idx, count in self.index:
            if idx < current + count:
                seq = self._load_sequence(jsonl_idx, idx - current)
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[idx] = seq
                return seq
            current += count
        raise IndexError(f"Index {idx} out of range (loader has {len(self)} sequences)")

    def _load_sequence(self, jsonl_idx: int, seq_in_file: int) -> Dict:
        jsonl_file = self.jsonl_files[jsonl_idx]
        mp4_file = jsonl_file.with_suffix('.mp4')
        start_tick = seq_in_file * self.sequence_length

        ticks = []
        with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i < start_tick:
                    continue
                if i >= start_tick + self.sequence_length:
                    break
                try:
                    data = json.loads(line)
                    ticks.append({
                        'tick': data.get('tick', i),
                        'inventory': data.get('inventory', []),
                        'mouse': data.get('mouse', {}),
                        'keyboard': data.get('keyboard', {}),
                        'isGuiOpen': data.get('isGuiOpen', False),
                        'isGuiInventory': data.get('isGuiInventory', False),
                        'hotbar': data.get('hotbar', 0),
                        'yaw': data.get('yaw', 0.0),
                        'pitch': data.get('pitch', 0.0),
                    })
                except json.JSONDecodeError:
                    continue

        cap = cv2.VideoCapture(str(mp4_file))
        frames = []
        if cap.isOpened():
            for i in range(start_tick, start_tick + self.sequence_length):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(Image.fromarray(frame))
        cap.release()
        # Pad with black frames if video was truncated or had a bad moov atom
        while len(frames) < self.sequence_length:
            frames.append(Image.new('RGB', (224, 224), color='black'))

        # Per-slot BERT type targets: [seq_len, 36, 768]  (type name only, no count)
        inv_embs = torch.stack([
            self.encoder.encode_inventory_per_slot(t['inventory']) for t in ticks
        ])
        # Per-slot integer counts: [seq_len, 36]
        inv_counts = torch.stack([
            self.encoder.encode_inventory_counts(t['inventory']) for t in ticks
        ])
        # Cheap non-empty flag
        has_items = [InventoryTextEncoder.inventory_has_items(t['inventory']) for t in ticks]

        # Per-slot type names for NN decoding in eval: List[List[str]] (seq_len, 36)
        # "oak_log" or "empty slot" — counts are in inv_counts, not here
        slot_texts_per_tick = []
        for t in ticks:
            slot_texts = ["empty slot"] * 36
            for i, item in enumerate(t['inventory']):
                slot_idx = item.get('slot', i)
                if isinstance(slot_idx, int) and 0 <= slot_idx < 36:
                    slot_texts[slot_idx] = self.encoder._slot_to_type_text(item)
                elif i < 36:
                    slot_texts[i] = self.encoder._slot_to_type_text(item)
            slot_texts_per_tick.append(slot_texts)

        return {
            'frames': frames,
            'inventory_embeddings': inv_embs,     # [seq_len, 36, 768] type BERT embs
            'inventory_counts': inv_counts,        # [seq_len, 36] int counts
            'inventory_has_items': has_items,
            'inventory_slot_texts': slot_texts_per_tick,  # type names for NN vocab
            'ticks': ticks,
            'source_file': jsonl_file.name,
        }
