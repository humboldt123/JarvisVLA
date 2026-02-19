"""
Auxiliary prediction heads and memory projection layers for Stateful JARVIS-VLA.

This module provides:
1. InventoryEmbeddingHead: Predicts 768-dim inventory embeddings from hidden state
2. MemoryProjections: W_in (memory→hidden) and W_out (hidden→memory) for recurrent memory

Key Design:
- Memory dimension (512) is separate from model hidden dimension (3584 for Qwen2VL-7B)
- W_in projects memory to model space for input as a token
- W_out compresses updated memory back to memory_dim
- Auxiliary head uses raw hidden state (before W_out projection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class InventoryEmbeddingHead(nn.Module):
    """
    Auxiliary head that predicts per-slot inventory state from the memory hidden state.

    Outputs two tensors per forward pass:
        type_embeddings  [batch, num_slots, output_dim]  — L2-normalised BERT-space
                         embedding for each slot's item type.  Trained with InfoNCE
                         (contrastive) loss so the model must distinguish items, not
                         just land in the general "item description" cluster.
        count_preds      [batch, num_slots]              — predicted log(count+1) for
                         each slot.  Trained with MSE against log(true_count+1).

    # --- Future NBT head (not yet implemented) ---
    # A third output nbt_embeddings [batch, num_slots, output_dim] would encode
    # arbitrary NBT metadata (enchantments, durability, custom names, etc.) as a
    # BERT embedding of the flattened NBT string.  Same InfoNCE loss as the type
    # head.  Slots with no NBT all map to BERT("no_nbt").
    # Add: self.nbt_head = copy of fusion + Linear(output_dim, output_dim) + LayerNorm

    Loss computation is done externally in the training loop so that InfoNCE can
    aggregate non-empty slots across the full BPTT chunk for more negatives.

    Args:
        hidden_dim:  Dimension of the model's hidden state (3584 for Qwen2VL-7B)
        output_dim:  Dimension of BERT embeddings (768)
        num_slots:   Number of inventory slots (36 for Minecraft)
        dropout:     Dropout rate
        temperature: InfoNCE temperature (default 0.07, same as CLIP)
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 768,
        num_slots: int = 36,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_slots = num_slots
        self.temperature = temperature

        intermediate_dim = hidden_dim // 4

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_dim),
        )

        # Learnable slot-position embeddings (hotbar slot 0, inventory row 1, …)
        self.slot_embedding = nn.Embedding(num_slots, output_dim // 8)

        # Fuse base projection with slot embedding
        self.fusion = nn.Sequential(
            nn.Linear(output_dim + output_dim // 8, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Count head: predicts log(count + 1) per slot from the fused representation
        # Softplus ensures output is always positive (target log(count+1) >= 0)
        self.count_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.GELU(),
            nn.Linear(output_dim // 4, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, hidden_dim]

        Returns:
            type_embeddings: [batch, num_slots, output_dim]  L2-normalised
            count_preds:     [batch, num_slots]               predicted log(count+1)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        base = self.projection(hidden_states)                          # [batch, D]
        base = base.unsqueeze(1).expand(-1, self.num_slots, -1)        # [batch, S, D]

        slot_idx = torch.arange(self.num_slots, device=device)
        slot_emb = self.slot_embedding(slot_idx)                       # [S, D//8]
        slot_emb = slot_emb.unsqueeze(0).expand(batch_size, -1, -1)   # [batch, S, D//8]

        fused = self.fusion(torch.cat([base, slot_emb], dim=-1))       # [batch, S, D]

        type_embeddings = F.normalize(fused, p=2, dim=-1)             # [batch, S, D]
        count_preds = self.count_head(fused).squeeze(-1)               # [batch, S]

        return type_embeddings, count_preds


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
        memory_dim: int = 512,
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
        
        # W_out: hidden_dim -> memory_dim (compresses updated memory)
        self.W_out = nn.Linear(hidden_dim, memory_dim)
        
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
    
    def project_out(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state back to memory dimension.
        
        Args:
            hidden: [batch, hidden_dim] - hidden state at memory token position
        
        Returns:
            memory: [batch, memory_dim] - compressed for next timestep
        """
        return self.W_out(hidden)


class HiddenStateExtractor(nn.Module):
    """
    Extracts the hidden state at a specific position (e.g., memory token).
    
    For the memory token, we want the hidden state at the last position
    (since memory is appended after visual/text tokens).
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
