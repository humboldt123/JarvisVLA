"""
Custom trainer for JARVIS-VLA with auxiliary inventory prediction.

ARCHITECTURE:
============
Input: [Video Frames] + [Text Prompt]
           ↓
    Qwen2VL Backbone (frozen or fine-tuned)
           ↓
    Hidden States (pooled)
           ↓
    ├─→ [Inventory Head] ──→ Inventory predictions (auxiliary loss)
    └─→ [LM Head] ──→ Text + Actions (main loss)

The inventory head is only active during training and is ignored during inference.
Loss = Main_Loss + inventory_loss_weight * Inventory_Loss
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from transformers import Trainer


class VLAInventoryTrainer(Trainer):
    """
    Custom trainer for JARVIS-VLA with auxiliary inventory prediction.
    
    The trainer:
    1. Runs normal forward pass through Qwen2VL (text/action outputs)
    2. Extracts hidden states from the model
    3. Passes them through inventory head  
    4. Computes inventory loss
    5. Adds inventory loss to main loss with configurable weight
    
    Args:
        inventory_head: Optional inventory head to train
        inventory_loss_weight: Weight for inventory auxiliary loss (default: 0.1)
        **kwargs: Additional args passed to parent Trainer
    """
    
    def __init__(
        self,
        inventory_head: Optional[nn.Module] = None,
        inventory_loss_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.inventory_head = inventory_head
        self.inventory_loss_weight = inventory_loss_weight
        
        if inventory_head is not None:
            self.inventory_head.to(self.args.device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: main loss + inventory auxiliary loss.
        
        Architecture:
        [Frames + Text] → [Backbone] → [Hidden States]
                                           ↓
                              ┌─→ [Inventory Head] → Loss_inv
                              └─→ [LM Head] → Loss_main
        
        Total Loss = Loss_main + weight * Loss_inv
        """
        # Check if we have inventory targets
        has_inventory = 'inventory_items' in inputs
        
        # Extract inventory targets if present
        if has_inventory:
            inventory_items = inputs.pop('inventory_items', None)
            inventory_counts = inputs.pop('inventory_counts', None)
        
        # Main model forward pass
        # This goes: [Frames + Text] → [Backbone] → [LM Head] → outputs
        outputs = model(**inputs)
        main_loss = outputs.loss
        
        # Compute inventory loss if head is available and we have targets
        inventory_loss = 0.0
        inventory_metrics = {}
        
        if self.inventory_head is not None and has_inventory:
            # Extract hidden states from model output
            # outputs.hidden_states is tuple of (batch, seq_len, hidden_dim) for each layer
            hidden_states = outputs.hidden_states
            
            if hidden_states is not None:
                # Get last layer hidden states
                last_hidden = hidden_states[-1]  # (batch, seq_len, hidden_dim)
                
                # Pool hidden states - use mean pooling over sequence
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    # Masked mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    pooled_features = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
                else:
                    # Simple mean pooling
                    pooled_features = last_hidden.mean(dim=1)
                
                # Forward through inventory head: [Hidden] → [Inventory Head] → predictions
                item_logits, count_logits = self.inventory_head(pooled_features)
                
                # Compute inventory loss
                inv_loss, inv_loss_dict = self.inventory_head.compute_loss(
                    item_logits=item_logits,
                    item_targets=inventory_items,
                    count_logits=count_logits,
                    count_targets=inventory_counts,
                )
                
                inventory_loss = inv_loss
                inventory_metrics = inv_loss_dict
                
                # Log inventory metrics
                for key, value in inv_loss_dict.items():
                    self.log({f"train/{key}": value})
        
        # Combine losses: Loss_total = Loss_main + weight * Loss_inv
        total_loss = main_loss + self.inventory_loss_weight * inventory_loss
        
        # Log combined loss
        self.log({"train/total_loss": total_loss.item()})
        self.log({"train/main_loss": main_loss.item()})
        if has_inventory:
            self.log({"train/inventory_loss": inventory_loss if isinstance(inventory_loss, float) else inventory_loss.item()})
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Prediction step - inventory head is ignored during inference.
        
        During inference, we only use: [Frames + Text] → [Backbone] → [LM Head] → Actions
        """
        # Remove inventory targets if present (not needed for inference)
        inputs.pop('inventory_items', None)
        inputs.pop('inventory_counts', None)
        
        # Run normal prediction (inventory head not used)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
