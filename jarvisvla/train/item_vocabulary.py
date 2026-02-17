"""
Author: AI Assistant
Date: 2026-02-17
Description: Item vocabulary with string similarity for partial credit.

This module handles:
1. Building item vocabulary from item names
2. Computing item similarities based on string overlap (not hardcoded categories)
3. Mapping between item names and indices
4. Designed for extensibility (future: metadata like enchants, custom names)

Future extension for metadata:
- Item prediction: base item type (e.g., "diamond_sword")
- Metadata prediction: enchants, custom names, durability, etc. (separate heads)
- Scoring: +1.0 for correct item, +0.5 for correct metadata, -0.5 for wrong metadata
"""

import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def compute_string_similarity(name1: str, name2: str) -> float:
    """
    Compute similarity between two item names based on shared substrings.
    
    Simple approach: count shared character n-grams (substrings).
    
    Examples:
        stone_pickaxe vs iron_pickaxe -> high similarity (shares "pickaxe")
        oak_planks vs oak_log -> medium similarity (shares "oak")
        stone_pickaxe vs oak_planks -> low similarity (shares no meaningful substring)
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if name1 == name2:
        return 1.0
    
    # Remove namespace prefixes
    name1 = name1.split(':')[-1]
    name2 = name2.split(':')[-1]
    
    if name1 == name2:
        return 1.0
    
    # Split into parts (e.g., "stone_pickaxe" -> ["stone", "pickaxe"])
    parts1 = set(name1.split('_'))
    parts2 = set(name2.split('_'))
    
    # Count shared parts
    shared = parts1 & parts2
    all_parts = parts1 | parts2
    
    if not all_parts:
        return 0.0
    
    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    return len(shared) / len(all_parts)


class ItemVocabulary:
    """
    Vocabulary for Minecraft items with string-based partial credit.
    
    NO HARDCODED CATEGORIES - similarity is computed from item names at runtime.
    
    This class:
    1. Maps item names (e.g., "stone_pickaxe") to indices
    2. Computes similarity between items using string overlap
    3. Builds similarity matrix for partial credit loss
    
    Future extensibility:
    - Can be extended to handle metadata (enchants, custom names)
    - Metadata will be separate prediction heads
    - Scoring: +1.0 for item, +0.5 for metadata, -0.5 for wrong metadata
    """
    
    def __init__(self, item_to_idx: Optional[Dict[str, int]] = None):
        """
        Initialize vocabulary.
        
        Args:
            item_to_idx: Optional pre-built mapping from item names to indices.
                        If None, starts with just 'empty' at index 0.
        """
        if item_to_idx is None:
            item_to_idx = {'empty': 0}  # 0 is reserved for empty slot
        
        self.item_to_idx = item_to_idx
        self.idx_to_item = {idx: name for name, idx in item_to_idx.items()}
        self.num_items = len(item_to_idx)
        
        # Pre-compute similarity matrix
        self._similarity_matrix = None
    
    def get_item_index(self, item_name: str) -> int:
        """
        Get the index for an item name.
        
        Strips namespace prefixes (e.g., "minecraft:stone_pickaxe" -> "stone_pickaxe")
        Returns 0 (empty) for unknown items.
        """
        # Strip namespace prefix
        clean_name = item_name.split(':')[-1]
        return self.item_to_idx.get(clean_name, 0)  # Unknown -> empty
    
    def get_item_name(self, idx: int) -> str:
        """Get item name from index."""
        return self.idx_to_item.get(idx, 'empty')
    
    def add_item(self, item_name: str) -> int:
        """Add a new item to the vocabulary. Returns its index."""
        clean_name = item_name.split(':')[-1]
        if clean_name not in self.item_to_idx:
            idx = len(self.item_to_idx)
            self.item_to_idx[clean_name] = idx
            self.idx_to_item[idx] = clean_name
            self.num_items = len(self.item_to_idx)
            # Invalidate cached similarity matrix
            self._similarity_matrix = None
        return self.item_to_idx[clean_name]
    
    def build_similarity_matrix(self) -> torch.Tensor:
        """
        Build the similarity matrix for all items in vocabulary.
        
        Matrix[i, j] = similarity between item i and item j
        Computed using string similarity (shared substrings), NOT hardcoded categories.
        
        Returns:
            Similarity matrix of shape (num_items, num_items)
        """
        n = self.num_items
        matrix = torch.eye(n)  # Diagonal = 1.0 (exact match)
        
        for i in range(n):
            name_i = self.get_item_name(i)
            for j in range(i + 1, n):
                name_j = self.get_item_name(j)
                sim = compute_string_similarity(name_i, name_j)
                matrix[i, j] = sim
                matrix[j, i] = sim  # Symmetric
        
        self._similarity_matrix = matrix
        return matrix
    
    def get_similarity_matrix(self) -> torch.Tensor:
        """Get the similarity matrix, building it if necessary."""
        if self._similarity_matrix is None:
            return self.build_similarity_matrix()
        return self._similarity_matrix
    
    def save(self, path: Path):
        """Save vocabulary to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'item_to_idx': self.item_to_idx,
                'num_items': self.num_items,
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ItemVocabulary':
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(item_to_idx=data['item_to_idx'])


def create_inventory_tensor(
    inventory_list: List[Dict],
    vocab: ItemVocabulary,
    num_slots: int = 36,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert VPT inventory format to tensor.
    
    VPT format: [{"type": "stone_pickaxe", "quantity": 1}, ...]
    
    Args:
        inventory_list: List of item dicts from VPT
        vocab: ItemVocabulary for mapping names to indices
        num_slots: Number of inventory slots (default 36)
    
    Returns:
        item_tensor: (num_slots,) tensor of item indices
        count_tensor: (num_slots,) tensor of item counts
    """
    item_tensor = torch.zeros(num_slots, dtype=torch.long)
    count_tensor = torch.zeros(num_slots, dtype=torch.long)
    
    for i, item in enumerate(inventory_list[:num_slots]):
        item_name = item.get('type', 'empty')
        quantity = item.get('quantity', 0)
        
        item_idx = vocab.get_item_index(item_name)
        item_tensor[i] = item_idx
        count_tensor[i] = min(quantity, 64)  # Cap at 64 (Minecraft max stack)
    
    return item_tensor, count_tensor
