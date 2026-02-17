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
    Auxiliary head that predicts inventory embeddings from hidden state.
    
    This head forces the model to maintain inventory information in its
    hidden state across arbitrary time spans. The hidden state becomes
    a memory that persists inventory contents even when the inventory
    screen is not visible.
    
    Architecture:
        [Hidden State] → [MLP] → [768-dim Embedding per slot]
        
    Loss: Cosine similarity between predicted embedding and target
          (target is encoded by frozen text encoder like BERT)
    
    At inference: This head is DROPPED. The model uses its internal
    memory (hidden state) to answer inventory queries.
    
    Args:
        hidden_dim: Dimension of model's hidden state (e.g., 3584 for Qwen2VL-7B)
        output_dim: Dimension of inventory embedding (default: 768 for BERT)
        num_slots: Number of inventory slots (default: 36)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 768,
        num_slots: int = 36,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_slots = num_slots
        
        # MLP to project hidden state to inventory embedding
        # Using a bottleneck architecture for efficiency
        intermediate_dim = hidden_dim // 4
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_dim),
        )
        
        # Per-slot embeddings (slot 0 = hotbar slot 0, etc.)
        # This helps distinguish which slot we're predicting
        self.slot_embedding = nn.Embedding(num_slots, output_dim // 8)
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim + output_dim // 8, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stability."""
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
    ) -> torch.Tensor:
        """
        Forward pass to predict inventory embeddings.
        
        Args:
            hidden_states: [batch, hidden_dim] - the model's hidden state
                          (raw_updated_memory from the memory token position)
        
        Returns:
            embeddings: [batch, num_slots, output_dim]
                       Predicted embeddings for each inventory slot
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # Project base hidden state
        base_embedding = self.projection(hidden_states)  # [batch, output_dim]
        
        # Expand for all slots
        base_embedding = base_embedding.unsqueeze(1).expand(-1, self.num_slots, -1)
        # [batch, num_slots, output_dim]
        
        # Add slot embeddings
        slot_indices = torch.arange(self.num_slots, device=device)
        slot_emb = self.slot_embedding(slot_indices)  # [num_slots, output_dim//8]
        slot_emb = slot_emb.unsqueeze(0).expand(batch_size, -1, -1)
        # [batch, num_slots, output_dim//8]
        
        # Concatenate and fuse
        combined = torch.cat([base_embedding, slot_emb], dim=-1)
        embeddings = self.fusion(combined)  # [batch, num_slots, output_dim]
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def compute_loss(
        self,
        predicted_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        is_non_empty: Optional[torch.Tensor] = None,
        non_empty_weight: float = 5.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cosine similarity loss with weighting for non-empty frames.
        
        The OpenVPT dataset is heavily imbalanced (most frames have empty inventory).
        Without weighting, the model could ignore rare non-empty events and simply
        learn to always predict "empty". We weight non-empty frames more heavily
        to ensure the model learns to predict inventory contents.
        
        Args:
            predicted_embeddings: [batch, num_slots, output_dim]
            target_embeddings: [batch, num_slots, output_dim] 
                              (from frozen text encoder)
            valid_mask: [batch, num_slots] - which slots to compute loss on
            is_non_empty: [batch] - boolean tensor indicating non-empty frames
            non_empty_weight: Weight multiplier for non-empty frames (default: 5.0)
        
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics for logging
        """
        # Normalize targets (in case they aren't already)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)
        
        # Cosine similarity for each slot
        similarity = (predicted_embeddings * target_embeddings).sum(dim=-1)
        # [batch, num_slots], range [-1, 1]
        
        # Loss: 1 - cosine_similarity (so lower is better)
        per_slot_loss = 1.0 - similarity
        
        # Apply mask if provided
        if valid_mask is not None:
            per_slot_loss = per_slot_loss * valid_mask
            frame_loss = per_slot_loss.sum(dim=-1) / (valid_mask.sum(dim=-1) + 1e-8)
        else:
            frame_loss = per_slot_loss.mean(dim=-1)
        # frame_loss: [batch]
        
        # Weight non-empty frames more heavily because the dataset is heavily imbalanced
        # (most frames have empty inventory). Without weighting, the model could ignore
        # rare non-empty events and simply learn to always predict "empty".
        if is_non_empty is not None:
            # is_non_empty: [batch] boolean -> float weights
            frame_weights = torch.where(
                is_non_empty,
                torch.tensor(non_empty_weight, dtype=frame_loss.dtype, device=frame_loss.device),
                torch.tensor(1.0, dtype=frame_loss.dtype, device=frame_loss.device)
            )
            weighted_loss = (frame_loss * frame_weights).sum() / (frame_weights.sum() + 1e-8)
            
            # Track metrics for empty vs non-empty separately
            with torch.no_grad():
                empty_mask = ~is_non_empty
                empty_loss = frame_loss[empty_mask].mean() if empty_mask.any() else torch.tensor(0.0)
                non_empty_loss = frame_loss[is_non_empty].mean() if is_non_empty.any() else torch.tensor(0.0)
        else:
            weighted_loss = frame_loss.mean()
            empty_loss = frame_loss.mean()
            non_empty_loss = torch.tensor(0.0)
        
        loss = weighted_loss
        
        # Compute metrics
        with torch.no_grad():
            avg_similarity = similarity.mean().item()
            accurate_slots = (similarity > 0.9).float()
            if valid_mask is not None:
                accuracy = (accurate_slots * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            else:
                accuracy = accurate_slots.mean()
            
            # Separate accuracies for empty vs non-empty
            if is_non_empty is not None:
                empty_sim = similarity[~is_non_empty].mean().item() if (~is_non_empty).any() else 0.0
                non_empty_sim = similarity[is_non_empty].mean().item() if is_non_empty.any() else 0.0
            else:
                empty_sim = avg_similarity
                non_empty_sim = 0.0
        
        metrics = {
            'inventory_embedding_loss': loss.item(),
            'inventory_cosine_similarity': avg_similarity,
            'inventory_embedding_accuracy': accuracy.item(),
            'inventory_empty_loss': empty_loss.item() if isinstance(empty_loss, torch.Tensor) else empty_loss,
            'inventory_nonempty_loss': non_empty_loss.item() if isinstance(non_empty_loss, torch.Tensor) else non_empty_loss,
            'inventory_empty_cosine_sim': empty_sim,
            'inventory_nonempty_cosine_sim': non_empty_sim,
        }
        
        return loss, metrics


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
