"""
VPT (Video Pre-Training) Dataset Loader for JARVIS-VLA.

This module loads the OpenAI VPT dataset format:
- Videos: MP4 files with gameplay footage
- Annotations: JSONL files with per-tick data including inventory

The dataset returns:
- Video frames (sampled at appropriate intervals)
- Text prompts (if available)
- Inventory targets (for auxiliary training)
- Action targets (for main training)
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

from jarvisvla.train.item_vocabulary import ItemVocabulary


class VPTDataset(Dataset):
    """
    Dataset for OpenAI VPT (Video Pre-Training) data.
    
    Loads video frames and inventory data from VPT format:
    - videos/: MP4 files
    - annotations/: JSONL files with tick-by-tick data
    
    Each sample contains:
    - frames: Video frames (sampled)
    - text: Instruction/prompt text
    - inventory: List of {type, quantity} dicts
    - actions: Keyboard/mouse actions
    
    Args:
        data_dir: Directory containing 'videos' and 'annotations' subdirs
        item_vocabulary: ItemVocabulary for mapping item names to indices
        num_frames: Number of frames to sample per video (default: 4)
        frame_size: Size to resize frames to (default: (224, 224))
        max_samples: Maximum number of samples to load (None = all)
    """
    
    def __init__(
        self,
        data_dir: str,
        item_vocabulary: Optional[ItemVocabulary] = None,
        num_frames: int = 4,
        frame_size: Tuple[int, int] = (224, 224),
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.annotations_dir = self.data_dir / "annotations"
        self.item_vocab = item_vocabulary
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Build index of all samples
        self.samples = self._build_index(max_samples)
        
        print(f"VPTDataset loaded:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Frames per sample: {num_frames}")
    
    def _build_index(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Build index of all available samples."""
        samples = []
        
        if not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {self.annotations_dir}")
        
        # Find all annotation files
        jsonl_files = sorted(self.annotations_dir.glob("*.jsonl"))
        
        for jsonl_file in jsonl_files:
            video_file = self.videos_dir / f"{jsonl_file.stem}.mp4"
            
            # Read annotation file
            with open(jsonl_file, 'r') as f:
                for tick_num, line in enumerate(f):
                    if max_samples and len(samples) >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line)
                        
                        # Only include samples with inventory data
                        if data.get('inventory'):
                            samples.append({
                                'video_path': str(video_file),
                                'annotation_path': str(jsonl_file),
                                'tick': data.get('tick', tick_num),
                                'inventory': data.get('inventory', []),
                                'actions': {
                                    'keyboard': data.get('keyboard', {}),
                                    'mouse': data.get('mouse', {}),
                                    'hotbar': data.get('hotbar', 0),
                                },
                                'gui_open': data.get('isGuiOpen', False),
                                'yaw': data.get('yaw', 0),
                                'pitch': data.get('pitch', 0),
                            })
                    except json.JSONDecodeError:
                        continue
            
            if max_samples and len(samples) >= max_samples:
                break
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_video_frames(self, video_path: str, tick: int) -> List[Image.Image]:
        """
        Load frames from video around a specific tick.
        
        VPT videos are recorded at 20 FPS (1 tick = 1 frame).
        We sample num_frames frames around the target tick.
        
        MP4 ↔ JSONL pairing: Same basename, different extension
        Example: video.mp4 ↔ video.jsonl
        
        Args:
            video_path: Path to MP4 file
            tick: Target tick number
        
        Returns:
            List of PIL Images
        """
        frames = []
        
        # Check if video exists
        if not Path(video_path).exists():
            # Video not available - return dummy frames
            return [Image.new('RGB', self.frame_size, color='black') for _ in range(self.num_frames)]
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return [Image.new('RGB', self.frame_size, color='black') for _ in range(self.num_frames)]
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # VPT is 20 FPS, so tick ≈ frame number
            # Sample frames around target tick
            start_tick = max(0, tick - self.num_frames // 2)
            frame_indices = [start_tick + i for i in range(self.num_frames)]
            
            for frame_idx in frame_indices:
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
            frames = [Image.new('RGB', self.frame_size, color='black') for _ in range(self.num_frames)]
        
        return frames
    
    def _process_inventory(self, inventory_list: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process inventory list to tensors.
        
        Args:
            inventory_list: List of {type, quantity} dicts
        
        Returns:
            item_indices: (36,) tensor of item class indices
            item_counts: (36,) tensor of item counts
        """
        items = torch.zeros(36, dtype=torch.long)
        counts = torch.zeros(36, dtype=torch.long)
        
        for i, item in enumerate(inventory_list[:36]):
            item_name = item.get('type', '')
            quantity = item.get('quantity', 0)
            
            if self.item_vocab:
                item_idx = self.item_vocab.get_item_index(item_name)
            else:
                # Without vocabulary, can't map names
                item_idx = 0
            
            items[i] = item_idx
            counts[i] = min(quantity, 64)
        
        return items, counts
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - images: List of PIL Images
                - text: Text prompt
                - inventory_items: (36,) tensor
                - inventory_counts: (36,) tensor
                - actions: Dict of action data
        """
        sample = self.samples[idx]
        
        # Load video frames
        images = self._load_video_frames(sample['video_path'], sample['tick'])
        
        # Process inventory
        inventory_items, inventory_counts = self._process_inventory(sample['inventory'])
        
        # Create text prompt (simple version)
        text = f"Current inventory: {len(sample['inventory'])} items"
        
        return {
            'images': images,
            'text': text,
            'inventory_items': inventory_items,
            'inventory_counts': inventory_counts,
            'actions': sample['actions'],
            'video_path': sample['video_path'],
            'tick': sample['tick'],
        }


def build_vocabulary_from_vpt(data_dir: str, min_frequency: int = 1) -> ItemVocabulary:
    """
    Build item vocabulary from VPT dataset.
    
    Args:
        data_dir: Directory containing VPT annotations
        min_frequency: Minimum frequency for item to be included
    
    Returns:
        ItemVocabulary with all items from dataset
    """
    vocab = ItemVocabulary()
    annotations_dir = Path(data_dir) / "annotations"
    
    item_counts = {}
    
    for jsonl_file in annotations_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    for item in data.get('inventory', []):
                        item_name = item.get('type', '')
                        if item_name:
                            item_counts[item_name] = item_counts.get(item_name, 0) + 1
                except json.JSONDecodeError:
                    continue
    
    # Add items that meet frequency threshold
    for item_name, count in item_counts.items():
        if count >= min_frequency:
            vocab.add_item(item_name)
    
    print(f"Built vocabulary from VPT data:")
    print(f"  Total unique items: {len(item_counts)}")
    print(f"  Items after filtering: {vocab.num_items}")
    
    return vocab
