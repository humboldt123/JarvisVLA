"""
Recurrent Inventory Head with Temporal Memory

Feeds previous inventory prediction as input to stabilize predictions across frames.
Model learns to:
- Trust visual input when inventory is open/visible
- Propagate memory when inventory is closed
- Update when visual evidence contradicts memory

NOTE: If adding another head (e.g., action prediction), we should feed its 
previous output as well using the same pattern: concatenate with current inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class RecurrentInventoryHead(nn.Module):
    """
    Inventory head that maintains temporal consistency by feeding
    previous predictions back as input.
    
    Architecture:
        [Visual Features] + [Previous Inventory Embedding] → [Fusion] → [Prediction]
    """
    
    def __init__(
        self,
        input_dim: int,
        num_slots: int = 36,
        num_item_classes: int = 256,
        hidden_dim: int = 512,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        predict_count: bool = True,
        use_partial_credit: bool = True,
        similarity_matrix: Optional[torch.Tensor] = None,
        empty_threshold: float = 0.5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.num_item_classes = num_item_classes
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.predict_count = predict_count
        self.use_partial_credit = use_partial_credit
        self.empty_threshold = empty_threshold
        
        # Embeddings for previous inventory state
        # Each slot: item_id + count (discrete bins)
        self.item_embedding = nn.Embedding(num_item_classes, embedding_dim)
        self.count_bins = [0, 1, 2, 4, 8, 16, 32, 48, 64]  # Discrete count bins
        self.count_embedding = nn.Embedding(len(self.count_bins), embedding_dim // 2)
        
        # Position embedding for slot index
        self.slot_embedding = nn.Embedding(num_slots, embedding_dim // 2)
        
        # Total previous state embedding per slot
        prev_state_dim = num_slots * (embedding_dim + embedding_dim // 2 + embedding_dim // 2)
        
        # Fusion: visual features + previous inventory state
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + prev_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Item classifier (predicts for all slots)
        self.item_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_slots * num_item_classes),
        )
        
        # Count regressor (per slot)
        if predict_count:
            self.count_regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_slots),
            )
        
        # Store similarity matrix for partial credit
        if similarity_matrix is not None:
            self.register_buffer('similarity_matrix', similarity_matrix)
        else:
            self.register_buffer('similarity_matrix', None)
    
    def embed_previous_inventory(
        self, 
        prev_items: torch.Tensor,
        prev_counts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed previous inventory state into a vector.
        
        Args:
            prev_items: [batch, num_slots] item indices
            prev_counts: [batch, num_slots] count values (optional)
            
        Returns:
            [batch, num_slots * embedding_dim] embedded state
        """
        batch_size = prev_items.shape[0]
        
        # Item embeddings
        item_emb = self.item_embedding(prev_items)  # [batch, slots, emb_dim]
        
        # Count embeddings (bin counts into discrete categories)
        if prev_counts is not None:
            # Bin the counts
            count_bins = torch.zeros_like(prev_counts)
            for i, threshold in enumerate(self.count_bins[1:], 1):
                count_bins = torch.where(prev_counts >= threshold, 
                                         torch.full_like(prev_counts, i), 
                                         count_bins)
            count_emb = self.count_embedding(count_bins)  # [batch, slots, emb//2]
        else:
            count_emb = torch.zeros(batch_size, self.num_slots, self.embedding_dim // 2,
                                   device=prev_items.device)
        
        # Slot position embeddings
        slot_indices = torch.arange(self.num_slots, device=prev_items.device)
        slot_emb = self.slot_embedding(slot_indices)  # [slots, emb//2]
        slot_emb = slot_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, slots, emb//2]
        
        # Concatenate: item + count + position
        combined = torch.cat([item_emb, count_emb, slot_emb], dim=-1)  # [batch, slots, total_emb]
        
        # Flatten to vector
        return combined.view(batch_size, -1)  # [batch, slots * total_emb]
    
    def forward(
        self, 
        visual_features: torch.Tensor,
        prev_items: Optional[torch.Tensor] = None,
        prev_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with temporal memory.
        
        Args:
            visual_features: [batch, input_dim] from vision backbone
            prev_items: [batch, num_slots] previous item predictions (None = empty)
            prev_counts: [batch, num_slots] previous count predictions (optional)
            
        Returns:
            item_logits: [batch, num_slots, num_item_classes]
            count_logits: [batch, num_slots] or None
        """
        batch_size = visual_features.shape[0]
        device = visual_features.device
        
        # Default to empty inventory if no previous state
        if prev_items is None:
            prev_items = torch.zeros(batch_size, self.num_slots, 
                                    dtype=torch.long, device=device)
        if prev_counts is None and self.predict_count:
            prev_counts = torch.zeros(batch_size, self.num_slots,
                                     dtype=torch.long, device=device)
        
        # Embed previous inventory
        prev_state_emb = self.embed_previous_inventory(prev_items, prev_counts)
        
        # Fuse visual + memory
        combined = torch.cat([visual_features, prev_state_emb], dim=-1)
        fused = self.fusion(combined)
        
        # Predict
        item_logits = self.item_classifier(fused)
        item_logits = item_logits.view(batch_size, self.num_slots, self.num_item_classes)
        
        count_logits = None
        if self.predict_count:
            count_logits = self.count_regressor(fused)
        
        return item_logits, count_logits
    
    def predict(
        self, 
        item_logits: torch.Tensor,
        count_logits: Optional[torch.Tensor] = None,
        return_confidence: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Get discrete predictions from logits."""
        probs = F.softmax(item_logits, dim=-1)
        max_probs, predicted_items = torch.max(probs, dim=-1)
        
        # Threshold-based empty detection
        empty_mask = max_probs < self.empty_threshold
        predicted_items = torch.where(empty_mask, 
                                      torch.zeros_like(predicted_items),
                                      predicted_items)
        
        result = {
            'items': predicted_items,
            'item_probs': max_probs,
            'empty_mask': empty_mask,
        }
        
        if count_logits is not None:
            predicted_counts = torch.clamp(count_logits.round(), 0, 64).long()
            predicted_counts = torch.where(empty_mask, 
                                          torch.zeros_like(predicted_counts),
                                          predicted_counts)
            result['counts'] = predicted_counts
        
        return result
    
    def compute_loss(
        self,
        item_logits: torch.Tensor,
        item_targets: torch.Tensor,
        count_logits: Optional[torch.Tensor] = None,
        count_targets: Optional[torch.Tensor] = None,
        empty_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute loss with partial credit support."""
        batch_size = item_logits.shape[0]
        
        # Valid mask (not padding)
        valid_mask = (item_targets != -100).float()
        
        # Item prediction loss
        item_logits_flat = item_logits.view(-1, self.num_item_classes)
        item_targets_flat = item_targets.view(-1)
        
        # Compute on all valid slots (including empty)
        if self.use_partial_credit and self.similarity_matrix is not None:
            # Partial credit loss using similarity matrix
            item_probs = F.softmax(item_logits_flat, dim=-1)
            
            # Get target similarities for each prediction
            valid_targets = item_targets_flat.clamp(0, self.num_item_classes - 1)
            target_sims = self.similarity_matrix[valid_targets]  # [batch*slots, num_classes]
            
            # Weighted cross-entropy: -sum(target_sim * log(pred_prob))
            log_probs = F.log_softmax(item_logits_flat, dim=-1)
            per_slot_loss = -(target_sims * log_probs).sum(dim=-1)
            
            # Apply valid mask
            per_slot_loss = per_slot_loss * valid_mask.view(-1)
            item_loss = per_slot_loss.sum() / valid_mask.sum().clamp(min=1)
        else:
            # Standard cross-entropy
            item_loss = F.cross_entropy(
                item_logits_flat, 
                item_targets_flat.clamp(0),
                reduction='none'
            )
            item_loss = (item_loss * valid_mask.view(-1)).sum() / valid_mask.sum().clamp(min=1)
        
        total_loss = item_loss
        loss_dict = {'item_loss': item_loss.item()}
        
        # Count loss (MSE on non-empty slots)
        if self.predict_count and count_logits is not None and count_targets is not None:
            non_empty_mask = (item_targets != 0).float() * valid_mask
            count_loss = F.mse_loss(count_logits.float(), count_targets.float(), reduction='none')
            count_loss = (count_loss * non_empty_mask).sum() / non_empty_mask.sum().clamp(min=1)
            
            total_loss = total_loss + 0.1 * count_loss
            loss_dict['count_loss'] = count_loss.item()
        
        # Compute accuracy metrics
        with torch.no_grad():
            pred_result = self.predict(item_logits, count_logits)
            pred_items = pred_result['items']
            
            # Overall accuracy
            correct = ((pred_items == item_targets) * valid_mask).sum()
            total = valid_mask.sum()
            loss_dict['accuracy'] = (correct / total).item() if total > 0 else 0
            
            # Empty vs non-empty accuracy
            empty_mask_target = (item_targets == 0)
            empty_correct = ((pred_items == item_targets) * empty_mask_target * valid_mask).sum()
            empty_total = (empty_mask_target * valid_mask).sum()
            loss_dict['empty_accuracy'] = (empty_correct / empty_total).item() if empty_total > 0 else 0
            
            non_empty_mask_target = (item_targets != 0)
            non_empty_correct = ((pred_items == item_targets) * non_empty_mask_target * valid_mask).sum()
            non_empty_total = (non_empty_mask_target * valid_mask).sum()
            loss_dict['non_empty_accuracy'] = (non_empty_correct / non_empty_total).item() if non_empty_total > 0 else 0
            loss_dict['non_empty_count'] = non_empty_total.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
