"""
Auxiliary prediction heads and memory projection layers for Stateful JARVIS-VLA.

This module provides:
1. InventoryClassificationHead: Predicts discrete item IDs and count classes (cross-entropy)
2. MemoryProjections: W_in (memory→hidden) and W_out (hidden→memory) for recurrent memory

Key Design:
- Memory dimension (512) is separate from model hidden dimension (3584 for Qwen2VL-7B)
- W_in projects memory to model space for input as a token
- W_out compresses updated memory back to memory_dim
- Auxiliary head uses raw hidden state (before W_out projection)
- Item type and count are both discrete classification problems (cross-entropy, not regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class InventoryClassificationHead(nn.Module):
    """
    Auxiliary head that predicts per-slot inventory state as discrete classes.

    Outputs two tensors per forward pass (both are raw logits for cross-entropy):
        item_logits   [batch, num_slots, item_vocab_size]  — item type classification
        count_logits  [batch, num_slots, count_classes]    — count bucket classification

    Cross-entropy losses:
        item_loss  = CE(item_logits.view(-1, item_vocab_size), item_ids.view(-1))
        count_loss = CE(count_logits[non_empty], count_targets[non_empty])

    Why cross-entropy instead of cosine similarity:
        - Gradients are strong until the model is confident on the correct class.
        - No embedding space to exploit by collapsing to a centroid.
        - Directly interpretable: output is a probability over item types.

    Phase 2 additions (not yet implemented):
        - durability_pct: float regression 0.0-1.0
        - enchant_id / enchant_lvl: classification per enchantment slot (up to 7)

    Args:
        hidden_dim:      Dimension of the model's hidden state (3584 for Qwen2VL-7B)
        item_vocab_size: Number of item classes including empty (0). Default: 976.
        count_classes:   Number of count bucket classes. Default: 128 (covers 0-127).
        num_slots:       Number of inventory slots. Default: 36.
        dropout:         Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        hidden_dim: int,
        item_vocab_size: int = 976,
        count_classes: int = 128,
        num_slots: int = 36,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.item_vocab_size = item_vocab_size
        self.count_classes = count_classes
        self.num_slots = num_slots

        intermediate_dim = hidden_dim // 4

        # Shared trunk: hidden → slot features
        # Each slot gets an independent projection direction so the head can
        # learn different readout directions for different slots.
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Item type head: [batch, intermediate_dim] → [batch, num_slots * item_vocab_size]
        self.item_head = nn.Linear(intermediate_dim, num_slots * item_vocab_size)

        # Count head: [batch, intermediate_dim] → [batch, num_slots * count_classes]
        self.count_head = nn.Linear(intermediate_dim, num_slots * count_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Warm-start item head: bias class 0 (empty) slightly lower so the model
        # doesn't default to predicting everything as empty from step 1.
        # Bias ~-0.5 on class 0 vs 0.0 on other classes gives modest prior
        # toward non-empty predictions.
        with torch.no_grad():
            if self.item_head.bias is not None:
                self.item_head.bias.view(self.num_slots, self.item_vocab_size)[:, 0] -= 0.5

        # Warm-start count head: bias class 1 (count=1) slightly higher — most
        # non-empty slots have at least 1 item.
        with torch.no_grad():
            if self.count_head.bias is not None:
                self.count_head.bias.view(self.num_slots, self.count_classes)[:, 1] += 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, hidden_dim]

        Returns:
            item_logits:  [batch, num_slots, item_vocab_size]  — raw logits for CE
            count_logits: [batch, num_slots, count_classes]    — raw logits for CE
        """
        batch_size = hidden_states.shape[0]

        trunk_out = self.trunk(hidden_states)   # [batch, intermediate_dim]

        item_logits = self.item_head(trunk_out).view(
            batch_size, self.num_slots, self.item_vocab_size
        )   # [batch, S, item_vocab_size]

        count_logits = self.count_head(trunk_out).view(
            batch_size, self.num_slots, self.count_classes
        )   # [batch, S, count_classes]

        return item_logits, count_logits


# Backward-compat alias — code that instantiated InventoryEmbeddingHead can
# be migrated gradually.  The constructor signature differs (see above).
InventoryEmbeddingHead = InventoryClassificationHead


class MemoryProjections(nn.Module):
    """
    Input and output projections for the recurrent memory token.
    
    The memory has a separate, smaller dimension (e.g., 512) that gets
    projected to/from the model's hidden dimension (e.g., 3584).
    
    Architecture:
        Input:  W_in:  Linear(memory_dim, hidden_dim)  - projects prev_memory to input token
        Output: W_out: Linear(hidden_dim, memory_dim)  - compresses updated memory for next step
    
    This mirrors the paper's use of a two-layer MLP for image projection,
    but we use linear layers for simplicity (can be upgraded to MLP).
    
    Args:
        memory_dim: Dimension of recurrent memory (e.g., 512)
        hidden_dim: Dimension of model's hidden states (e.g., 3584)
        use_mlp: If True, use 2-layer MLP for W_in instead of linear
    """
    
    def __init__(
        self,
        memory_dim: int = 1024,
        hidden_dim: int = 3584,
        use_mlp: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        
        # W_in: memory_dim -> hidden_dim (projects memory to model input space)
        if use_mlp:
            # Two-layer MLP similar to the paper's image projector
            intermediate_dim = (memory_dim + hidden_dim) // 2
            self.W_in = nn.Sequential(
                nn.Linear(memory_dim, intermediate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, hidden_dim),
            )
        else:
            # Simple linear projection (sufficient for now)
            self.W_in = nn.Linear(memory_dim, hidden_dim)
        
        # W_out: hidden_dim -> memory_dim (candidate new memory content)
        self.W_out = nn.Linear(hidden_dim, memory_dim)

        # GRU-style update gate.
        #
        # Without gating, every step does:
        #   new_memory = W_out(hidden)
        # which fully overwrites the memory — inventory seen at frame t is gone by t+1.
        #
        # With a GRU gate:
        #   z          = sigmoid(W_gate([hidden ; prev_memory]))   ∈ (0,1)^memory_dim
        #   candidate  = W_out(hidden)
        #   new_memory = z * prev_memory  +  (1 - z) * candidate
        #
        # When z ≈ 1 → mostly keep prev_memory  (nothing changed, preserve state)
        # When z ≈ 0 → mostly use candidate     (inventory updated, write new content)
        #
        # The model learns z from both the current hidden state AND the previous
        # memory, so it can detect "I just saw the inventory screen open" and
        # choose to write, vs "I'm just walking around" and choose to preserve.
        self.W_gate = nn.Linear(hidden_dim + memory_dim, memory_dim)

        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stability."""
        # Initialize W_in
        if isinstance(self.W_in, nn.Sequential):
            for module in self.W_in:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        else:
            nn.init.xavier_uniform_(self.W_in.weight, gain=0.1)
            if self.W_in.bias is not None:
                nn.init.constant_(self.W_in.bias, 0)
        
        # Initialize W_out
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.1)
        if self.W_out.bias is not None:
            nn.init.constant_(self.W_out.bias, 0)
    
    def project_in(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Project memory to hidden dimension for input as token.
        
        Args:
            memory: [batch, memory_dim]
        
        Returns:
            projected: [batch, hidden_dim] - ready to be appended as token embedding
        """
        return self.W_in(memory)
    
    def project_out(self, hidden: torch.Tensor, prev_memory: torch.Tensor) -> torch.Tensor:
        """
        GRU-style gated memory update.

        Args:
            hidden:      [batch, hidden_dim]  — hidden state at memory token position
            prev_memory: [batch, memory_dim]  — memory carried in from the previous step

        Returns:
            new_memory: [batch, memory_dim]
        """
        candidate = self.W_out(hidden)
        gate = torch.sigmoid(self.W_gate(torch.cat([hidden, prev_memory], dim=-1)))
        return gate * prev_memory + (1 - gate) * candidate


class HiddenStateExtractor(nn.Module):
    """
    Extracts the hidden state at a specific position (e.g., memory token).
    
    For the memory token, we want the hidden state at the first position
    (since memory is prepended before visual/text tokens).
    """
    
    def __init__(self, extract_position: str = "last"):
        super().__init__()
        self.extract_position = extract_position
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract hidden state from the specified position.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] - for finding actual last token
        
        Returns:
            hidden: [batch, hidden_dim]
        """
        if self.extract_position == "last":
            if attention_mask is not None:
                # Find actual last non-padded token
                seq_lengths = attention_mask.sum(dim=1).long() - 1
                batch_size = hidden_states.shape[0]
                return hidden_states[torch.arange(batch_size), seq_lengths]
            else:
                # Just take last position
                return hidden_states[:, -1, :]
        
        elif self.extract_position == "first":
            return hidden_states[:, 0, :]
        
        else:
            raise ValueError(f"Unknown extract_position: {self.extract_position}")


def create_inventory_text_embeddings(
    inventory_list: List[Dict],
    text_encoder,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Create target embeddings from inventory data using a frozen text encoder.
    
    Creates canonical text strings like:
        "diamond_pickaxe count:1 | cobblestone count:64"
    
    Args:
        inventory_list: List of {type, quantity} dicts
        text_encoder: Frozen text encoder (e.g., BERT)
        tokenizer: Tokenizer for text_encoder
        device: Device to place tensors on
    
    Returns:
        embeddings: [num_slots, 768] normalized embeddings
    """
    # Build canonical string (only non-empty slots)
    items_desc = []
    for item in inventory_list:
        item_type = item.get('type', '')
        quantity = item.get('quantity', 0)
        
        if item_type and item_type != 'air':
            # Clean up item name
            if ':' in item_type:
                item_type = item_type.split(':')[-1]
            items_desc.append(f"{item_type} count:{quantity}")
    
    if items_desc:
        text = " | ".join(items_desc)
    else:
        text = "empty inventory"
    
    # Tokenize and encode
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)
    
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        # Use [CLS] token embedding
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output[0]
        else:
            embedding = outputs.last_hidden_state[0, 0, :]  # First token (CLS)
    
    # Normalize
    embedding = F.normalize(embedding, p=2, dim=-1)
    
    return embedding
