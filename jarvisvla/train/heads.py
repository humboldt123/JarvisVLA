"""
Auxiliary prediction heads for JARVIS-VLA training.

The inventory head outputs RAW LOGITS (not JSON/strings).

Output format:
- item_logits: (batch, 36, num_classes) - Raw scores per slot per item
- count_logits: (batch, 36, 64) - Raw scores per slot per count (optional)

For prediction:
- Apply threshold: if max(logit) < threshold → empty
- Otherwise: argmax(logit) = predicted item

This avoids forcing the model to "waste" predictions on empty slots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class InventoryHead(nn.Module):
    """
    Auxiliary head for predicting player inventory from visual features.
    
    OUTPUT FORMAT:
    ============
    During forward():
        - item_logits: (batch, 36, num_item_classes) - RAW LOGITS
        - count_logits: (batch, 36, 64) - RAW LOGITS (optional)
    
    During predict():
        - item_predictions: (batch, 36) - Class indices
        - item_confidences: (batch, 36) - Confidence scores (0-1)
        - item_counts: (batch, 36) - Predicted counts
        
        Empty slots are detected via threshold on confidence:
        - If confidence < threshold → slot is empty (class 0)
        - Else → slot contains predicted item
    
    THRESHOLD-BASED EMPTY DETECTION:
    ================================
    Instead of forcing class 0 prediction for empty slots, we:
    1. Get softmax probabilities over all classes
    2. If max_prob < empty_threshold → assume slot is empty
    3. Else → predict argmax class
    
    This handles the 61.6% empty slot rate naturally without wasting model capacity.
    
    LOSS COMPUTATION:
    ================
    We compute loss on ALL slots (including empty), but:
    - For empty slots: reward high probability on class 0
    - For non-empty: reward correct item prediction
    
    EXTENSIBILITY:
    =============
    Future metadata (enchants, custom names) can be added as separate heads:
    - Base item: current item_classifier
    - Enchants: separate head (optional)
    - Custom names: separate head (optional)
    - Scoring: +1.0 base, +0.5 metadata bonus, -0.5 metadata penalty
    """
    
    def __init__(
        self,
        input_dim: int,
        num_slots: int = 36,
        num_item_classes: int = 256,
        hidden_dim: Optional[int] = None,
        predict_count: bool = False,
        max_count: int = 64,
        dropout: float = 0.1,
        use_partial_credit: bool = True,
        similarity_matrix: Optional[torch.Tensor] = None,
        empty_threshold: float = 0.5,  # Confidence threshold for empty detection
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.num_item_classes = num_item_classes
        self.predict_count = predict_count
        self.max_count = max_count
        self.use_partial_credit = use_partial_credit
        self.empty_threshold = empty_threshold
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 256)
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Item classifier
        self.item_classifier = nn.Linear(hidden_dim, num_slots * num_item_classes)
        
        # Count predictor (optional)
        if predict_count:
            self.count_predictor = nn.Linear(hidden_dim, num_slots * max_count)
        
        # Similarity matrix for partial credit
        self.register_buffer('similarity_matrix', similarity_matrix)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def set_similarity_matrix(self, similarity_matrix: torch.Tensor):
        """Set similarity matrix for partial credit."""
        self.similarity_matrix = similarity_matrix
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass - outputs RAW LOGITS.
        
        Args:
            hidden_states: (batch, input_dim) or (batch, seq_len, input_dim)
            attention_mask: Optional mask for pooling
        
        Returns:
            item_logits: (batch, num_slots, num_item_classes) - RAW LOGITS
            count_logits: (batch, num_slots, max_count) - RAW LOGITS (optional)
        """
        # Handle different input shapes
        if hidden_states.dim() == 3:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                hidden_states = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                hidden_states = hidden_states.mean(1)
        
        # Extract features
        features = self.feature_extractor(hidden_states)
        
        # Predict items
        item_logits = self.item_classifier(features)
        item_logits = item_logits.view(-1, self.num_slots, self.num_item_classes)
        
        # Predict counts
        count_logits = None
        if self.predict_count:
            count_logits = self.count_predictor(features)
            count_logits = count_logits.view(-1, self.num_slots, self.max_count)
        
        return item_logits, count_logits
    
    def predict(
        self,
        item_logits: torch.Tensor,
        count_logits: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with threshold-based empty detection.
        
        Args:
            item_logits: (batch, num_slots, num_item_classes)
            count_logits: (batch, num_slots, max_count) - optional
            return_confidence: Whether to return confidence scores
        
        Returns:
            Dictionary with:
                - items: (batch, num_slots) - Predicted item indices (0 = empty)
                - counts: (batch, num_slots) - Predicted counts
                - is_empty: (batch, num_slots) - Boolean mask for empty slots
                - confidence: (batch, num_slots) - Confidence scores (optional)
        """
        # Get probabilities
        probs = F.softmax(item_logits, dim=-1)  # (batch, num_slots, num_classes)
        
        # Get max probability and corresponding class
        max_probs, predicted_classes = probs.max(dim=-1)  # (batch, num_slots)
        
        # Apply threshold: if confidence < threshold, slot is empty
        is_empty = max_probs < self.empty_threshold  # (batch, num_slots)
        
        # Set empty slots to class 0
        predicted_items = predicted_classes.clone()
        predicted_items[is_empty] = 0  # 0 = empty
        
        result = {
            'items': predicted_items,
            'is_empty': is_empty,
            'max_probabilities': max_probs,
        }
        
        # Predict counts
        if count_logits is not None:
            count_probs = F.softmax(count_logits, dim=-1)
            predicted_counts = count_probs.argmax(dim=-1)
            result['counts'] = predicted_counts
        
        return result
    
    def compute_loss(
        self,
        item_logits: torch.Tensor,
        item_targets: torch.Tensor,
        count_logits: Optional[torch.Tensor] = None,
        count_targets: Optional[torch.Tensor] = None,
        empty_slot_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss on ALL slots including empty ones.
        
        This is important because:
        - 61.6% of slots are typically empty
        - We want the model to learn to predict empty slots correctly
        - Not just ignore them
        
        Args:
            item_logits: (batch, num_slots, num_classes)
            item_targets: (batch, num_slots) - Target class indices (0 = empty)
            count_logits: Optional count logits
            count_targets: Optional target counts
            empty_slot_idx: Index for empty class (typically 0)
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary with loss components
        """
        batch_size, num_slots, num_classes = item_logits.shape
        device = item_logits.device
        
        # Compute loss on ALL valid slots (including empty!)
        # valid_mask only ignores padding (-100), not empty slots
        valid_mask = (item_targets != -100).float()
        
        loss_dict = {}
        
        # Item loss with partial credit
        if self.use_partial_credit and self.similarity_matrix is not None:
            item_loss = self._compute_partial_credit_loss(
                item_logits, item_targets, valid_mask
            )
        else:
            item_loss = self._compute_standard_loss(
                item_logits, item_targets, valid_mask, num_classes
            )
        
        loss_dict['inventory_item_loss'] = item_loss.item()
        total_loss = item_loss
        
        # Count loss (only for non-empty slots - counts don't matter for empty)
        if self.predict_count and count_logits is not None and count_targets is not None:
            # Only compute count loss for non-empty slots
            non_empty_mask = (item_targets != empty_slot_idx).float() * valid_mask
            
            count_loss = F.cross_entropy(
                count_logits.view(-1, self.max_count),
                count_targets.view(-1).clamp(0, self.max_count - 1),
                reduction='none',
            )
            count_loss = count_loss.view(batch_size, num_slots)
            masked_count_loss = (count_loss * non_empty_mask).sum() / (non_empty_mask.sum() + 1e-8)
            
            loss_dict['inventory_count_loss'] = masked_count_loss.item()
            total_loss = total_loss + 0.1 * masked_count_loss
        
        loss_dict['inventory_total_loss'] = total_loss.item()
        
        # Log some statistics
        with torch.no_grad():
            pred_items = item_logits.argmax(dim=-1)
            correct = (pred_items == item_targets) * valid_mask.bool()
            accuracy = correct.float().sum() / valid_mask.sum().clamp(min=1)
            loss_dict['inventory_accuracy'] = accuracy.item()
            
            # Separate accuracies for empty vs non-empty
            empty_mask = (item_targets == empty_slot_idx) * valid_mask.bool()
            non_empty_mask = (item_targets != empty_slot_idx) * valid_mask.bool()
            
            if empty_mask.sum() > 0:
                empty_acc = (correct & empty_mask).float().sum() / empty_mask.sum()
                loss_dict['inventory_empty_accuracy'] = empty_acc.item()
            
            if non_empty_mask.sum() > 0:
                non_empty_acc = (correct & non_empty_mask).float().sum() / non_empty_mask.sum()
                loss_dict['inventory_non_empty_accuracy'] = non_empty_acc.item()
        
        return total_loss, loss_dict
    
    def _compute_standard_loss(
        self,
        item_logits: torch.Tensor,
        item_targets: torch.Tensor,
        compute_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Standard cross-entropy loss on all valid slots."""
        batch_size, num_slots, _ = item_logits.shape
        
        loss = F.cross_entropy(
            item_logits.view(-1, num_classes),
            item_targets.view(-1),
            reduction='none',
            ignore_index=-100,
        )
        
        loss = loss.view(batch_size, num_slots)
        masked_loss = (loss * compute_mask).sum() / (compute_mask.sum() + 1e-8)
        
        return masked_loss
    
    def _compute_partial_credit_loss(
        self,
        item_logits: torch.Tensor,
        item_targets: torch.Tensor,
        compute_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Partial credit loss using string similarity."""
        batch_size, num_slots, num_classes = item_logits.shape
        device = item_logits.device
        
        sim_matrix = self.similarity_matrix.to(device)
        
        # Create soft targets
        soft_targets = sim_matrix[item_targets]  # (batch, num_slots, num_classes)
        soft_targets = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Cross-entropy with soft targets
        log_probs = F.log_softmax(item_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1)  # (batch, num_slots)
        
        masked_loss = (loss * compute_mask).sum() / (compute_mask.sum() + 1e-8)
        
        return masked_loss
    
    def compute_accuracy(
        self,
        item_logits: torch.Tensor,
        item_targets: torch.Tensor,
        empty_slot_idx: int = 0,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute accuracy with threshold-based empty detection.
        
        Args:
            item_logits: Predicted logits
            item_targets: Target indices
            empty_slot_idx: Index for empty class
            threshold: Override threshold for empty detection
        
        Returns:
            Dictionary with accuracy metrics
        """
        if threshold is None:
            threshold = self.empty_threshold
        
        batch_size, num_slots, _ = item_logits.shape
        device = item_logits.device
        
        valid_mask = (item_targets != -100)
        
        # Get predictions with threshold
        predictions = self.predict(item_logits, empty_threshold=threshold)
        predicted_items = predictions['items']
        
        # Overall accuracy
        correct = (predicted_items == item_targets) & valid_mask
        overall_acc = correct.float().sum() / valid_mask.sum().clamp(min=1)
        
        # Non-empty accuracy (only count non-empty target slots)
        non_empty_mask = (item_targets != empty_slot_idx) & valid_mask
        non_empty_correct = (correct & non_empty_mask).float().sum()
        non_empty_acc = non_empty_correct / non_empty_mask.sum().clamp(min=1)
        
        # Empty slot accuracy
        empty_mask = (item_targets == empty_slot_idx) & valid_mask
        empty_correct = (correct & empty_mask).float().sum()
        empty_acc = empty_correct / empty_mask.sum().clamp(min=1)
        
        # Partial credit accuracy
        partial_acc = 0.0
        if self.similarity_matrix is not None:
            sim_matrix = self.similarity_matrix.to(device)
            pred_sim = sim_matrix[predicted_items, item_targets]
            partial_acc = (pred_sim * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)
        
        return {
            'overall_accuracy': overall_acc.item(),
            'non_empty_accuracy': non_empty_acc.item(),
            'empty_accuracy': empty_acc.item(),
            'partial_accuracy': partial_acc.item(),
        }
