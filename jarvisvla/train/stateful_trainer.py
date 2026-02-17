"""
Stateful VLA Trainer with Truncated Backpropagation Through Time (BPTT)

This trainer implements training for the stateful JarvisVLA model with:
1. Truncated BPTT with configurable chunk size (default: 16 frames)
2. Memory state carried between chunks but gradients truncated
3. Auxiliary inventory loss computed at each step within a chunk
4. Gradient accumulation across timesteps within a chunk

Truncated BPTT with chunk size 16 allows the model to learn long-term dependencies
via the carried memory, while keeping the backward graph shallow enough for a 7B
model. Gradient checkpointing further reduces memory footprint. This follows common
practice for training recurrent transformers (e.g., Recurrent Memory Transformer).

Training Loop (with chunk_size=16):
    memory = init_memory()
    
    for chunk_start in range(0, sequence_length, chunk_size):
        chunk_loss = 0
        
        # Forward through chunk (keeping computation graph)
        for t in range(chunk_start, min(chunk_start + chunk_size, seq_len)):
            output = model(frame[t], text[t], memory)
            loss_t = compute_loss(output)
            chunk_loss += loss_t
            memory = output.new_memory  # Carry forward, but will detach after chunk
        
        # Backward through chunk (gradients flow through 16 steps only)
        chunk_loss.backward()
        
        # Detach memory for next chunk (no gradient flow between chunks)
        memory = memory.detach()
        
        optimizer.step()
        optimizer.zero_grad()
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any, Iterator
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction
import numpy as np
from torch.utils.data import DataLoader, IterableDataset


class StatefulVLATrainer(Trainer):
    """
    Custom trainer for Stateful JarvisVLA with truncated BPTT.
    
    Key Features:
    - Processes sequences in chunks of consecutive frames
    - Within each chunk, gradients flow through all timesteps
    - Between chunks, memory is detached (truncated BPTT)
    - Auxiliary inventory loss computed at each step
    - Compatible with gradient checkpointing for memory efficiency
    
    Args:
        inventory_loss_weight: Weight for inventory auxiliary loss (default: 0.1)
        bptt_chunk_size: Chunk size for truncated BPTT (default: 16)
        max_grad_norm: Max gradient norm for clipping
        **kwargs: Additional args passed to parent Trainer
    """
    
    def __init__(
        self,
        inventory_loss_weight: float = 0.1,
        bptt_chunk_size: int = 16,
        max_grad_norm: float = 1.0,
        non_empty_loss_weight: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.inventory_loss_weight = inventory_loss_weight
        self.bptt_chunk_size = bptt_chunk_size
        self.max_grad_norm = max_grad_norm
        self.non_empty_loss_weight = non_empty_loss_weight
        
        # Enable gradient checkpointing on base model if available
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
            self.model.base_model.gradient_checkpointing_enable()
            print("[StatefulVLATrainer] Gradient checkpointing enabled on base model")
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute loss for a batch using truncated BPTT.
        
        Expected input format (from SequenceDataCollator):
        - pixel_values: [batch, seq_len, num_images, C, H, W] or processed format
        - input_ids: [batch, seq_len, text_seq_len]
        - labels: [batch, seq_len, text_seq_len]
        - inventory_embeddings: [batch, seq_len, 768] - BERT-encoded inventory targets
        - attention_mask: [batch, seq_len, text_seq_len]
        
        Args:
            model: StatefulJarvisVLA model
            inputs: Dictionary of batched sequence data
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for loss scaling)
        
        Returns:
            loss: Total loss averaged over the sequence
            outputs: Optional dict with metrics
        """
        # Check if we have sequence data
        has_sequence = 'sequence_length' in inputs or len(inputs.get('input_ids', []).shape) >= 3
        
        if not has_sequence:
            # Single-step training (fallback)
            return self._compute_single_step_loss(model, inputs, return_outputs)
        
        # Sequence training with truncated BPTT
        return self._compute_sequence_loss_truncated_bptt(model, inputs, return_outputs)
    
    def _compute_single_step_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Compute loss for single-step (non-sequence) training."""
        # Extract inventory targets if present
        inventory_embeddings = inputs.pop('inventory_embeddings', None)
        
        # Forward pass
        outputs = model(**inputs, inventory_embeddings=inventory_embeddings)
        
        loss = outputs.loss
        
        if return_outputs:
            return loss, {'memory': outputs.new_memory}
        return loss, None
    
    def _compute_sequence_loss_truncated_bptt(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute loss for sequence training with truncated BPTT.
        
        Implementation:
        1. Divide sequence into chunks of size bptt_chunk_size (default 16)
        2. For each chunk:
           - Forward through all timesteps in chunk (keep computation graph)
           - Accumulate losses
           - Backward through chunk
           - Detach memory for next chunk
        3. Memory carries forward between chunks but gradients don't flow
        """
        # Extract sequence data
        pixel_values = inputs.get('pixel_values')  # [batch, seq_len, ...]
        input_ids = inputs.get('input_ids')  # [batch, seq_len, text_len]
        labels = inputs.get('labels')  # [batch, seq_len, text_len]
        inventory_embeddings = inputs.get('inventory_embeddings')  # [batch, seq_len, 768]
        attention_mask = inputs.get('attention_mask')  # [batch, seq_len, text_len]
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = self.args.device
        
        # Initialize memory state
        memory = model.init_memory(batch_size, device)
        
        # Track metrics across sequence
        total_main_loss = 0.0
        total_inventory_loss = 0.0
        total_cosine_sim = 0.0
        num_steps = 0
        
        # Process sequence in chunks
        chunk_losses = []
        
        for chunk_start in range(0, seq_len, self.bptt_chunk_size):
            chunk_end = min(chunk_start + self.bptt_chunk_size, seq_len)
            
            # Accumulate loss for this chunk
            chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)
            chunk_main_loss = 0.0
            chunk_inv_loss = 0.0
            chunk_cosine_sim = 0.0
            chunk_steps = 0
            
            # Forward through chunk
            for t in range(chunk_start, chunk_end):
                # Extract data for this timestep
                # Note: pixel_values handling depends on Qwen2VL's expected format
                frame_t = self._extract_timestep_data(pixel_values, t)
                input_ids_t = input_ids[:, t, :]
                labels_t = labels[:, t, :]
                inv_emb_t = inventory_embeddings[:, t, :] if inventory_embeddings is not None else None
                attn_mask_t = attention_mask[:, t, :] if attention_mask is not None else None
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids_t,
                    pixel_values=frame_t,
                    prev_memory=memory,
                    attention_mask=attn_mask_t,
                    labels=labels_t,
                    inventory_embeddings=inv_emb_t,
                )
                
                # Accumulate loss
                if outputs.loss is not None:
                    # Weight by 1/seq_len to normalize
                    step_loss = outputs.loss / seq_len
                    chunk_loss = chunk_loss + step_loss
                    chunk_main_loss += outputs.loss.item() / seq_len
                
                # Track inventory metrics
                if outputs.inventory_embedding is not None and inv_emb_t is not None:
                    # Compute auxiliary loss for metrics
                    inv_loss, inv_metrics = model.inventory_embedding_head.compute_loss(
                        predicted_embeddings=outputs.inventory_embedding,
                        target_embeddings=inv_emb_t,
                    )
                    chunk_inv_loss += inv_loss.item() / seq_len
                    chunk_cosine_sim += inv_metrics.get('inventory_cosine_similarity', 0.0)
                
                # Update memory for next timestep WITHIN chunk (gradients flow)
                memory = outputs.new_memory
                chunk_steps += 1
            
            # Store chunk loss for backward
            chunk_losses.append(chunk_loss)
            
            # Accumulate metrics
            total_main_loss += chunk_main_loss
            total_inventory_loss += chunk_inv_loss
            total_cosine_sim += chunk_cosine_sim / max(chunk_steps, 1)
            num_steps += chunk_steps
            
            # DETACH memory between chunks (truncated BPTT)
            # This prevents gradients from flowing back to previous chunks
            memory = memory.detach()
        
        # Total loss is sum of all chunk losses
        # Note: Each chunk loss already has backward() called on it during training loop
        # We return the last chunk's loss for the Trainer's backward() call
        # Actually, we need to handle this differently - see training_step
        
        total_loss = sum(chunk_losses)
        
        # Compute average metrics
        avg_metrics = {
            'train/main_loss': total_main_loss,
            'train/inventory_loss': total_inventory_loss,
            'train/cosine_similarity': total_cosine_sim / max(seq_len // self.bptt_chunk_size, 1),
            'train/bptt_chunks': len(chunk_losses),
        }
        
        # Log metrics
        for key, value in avg_metrics.items():
            self.log({key: value})
        
        if return_outputs:
            return total_loss, {
                'final_memory': memory,
                'metrics': avg_metrics,
                'chunk_losses': chunk_losses,
            }
        
        return total_loss, None
    
    def _extract_timestep_data(self, data: Optional[torch.Tensor], t: int):
        """Extract data for timestep t from batched sequence data."""
        if data is None:
            return None
        
        # Handle different tensor shapes
        if data.dim() >= 3:
            # [batch, seq_len, ...] -> [batch, ...]
            return data[:, t]
        return data
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to handle truncated BPTT properly.
        
        The default Trainer calls backward() once per batch.
        For truncated BPTT, we need to:
        1. Process sequence in chunks
        2. Call backward() after each chunk
        3. Accumulate gradients across chunks
        """
        model.train()
        
        # Check if using sequence training
        has_sequence = len(inputs.get('input_ids', []).shape) >= 3
        
        if not has_sequence:
            # Use default behavior for non-sequence data
            return super().training_step(model, inputs, num_items_in_batch)
        
        # Sequence training with truncated BPTT
        return self._training_step_truncated_bptt(model, inputs, num_items_in_batch)
    
    def _detect_non_empty_inventory(
        self, 
        inventory_embeddings: torch.Tensor,
        empty_threshold: float = 0.95,
    ) -> torch.Tensor:
        """
        Detect which frames have non-empty inventory.
        
        We detect empty inventory by checking if the embedding is close to
        the "empty inventory" template embedding. If similarity is high,
        the inventory is empty; otherwise, it's non-empty.
        
        Args:
            inventory_embeddings: [batch, seq_len, 768] BERT embeddings
            empty_threshold: Cosine similarity threshold for empty detection
        
        Returns:
            is_non_empty: [batch, seq_len] boolean tensor
        """
        batch_size, seq_len, emb_dim = inventory_embeddings.shape
        device = inventory_embeddings.device
        
        # Create "empty inventory" embedding on the fly
        # We could cache this, but computing it once per batch is fine
        from transformers import AutoTokenizer, AutoModel
        
        # Load BERT on the same device
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()
        
        with torch.no_grad():
            inputs = tokenizer("empty inventory", return_tensors="pt", padding=True).to(device)
            empty_emb = model(**inputs).last_hidden_state[0, 0, :]  # [768]
            empty_emb = torch.nn.functional.normalize(empty_emb, dim=-1)
        
        # Compute similarity to empty template for all frames
        # inventory_embeddings is already normalized
        similarity_to_empty = torch.nn.functional.cosine_similarity(
            inventory_embeddings.view(-1, emb_dim),
            empty_emb.unsqueeze(0).expand(batch_size * seq_len, -1),
            dim=-1
        ).view(batch_size, seq_len)
        
        # Non-empty if similarity to empty is low
        is_non_empty = similarity_to_empty < empty_threshold
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return is_non_empty
    
    def _training_step_truncated_bptt(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Training step with truncated BPTT.
        
        Process:
        1. Divide sequence into chunks
        2. For each chunk: forward, loss.backward(), optimizer step
        3. Memory carries forward but is detached between chunks
        
        Non-empty inventory frames are weighted more heavily to handle
        class imbalance (most frames have empty inventory).
        """
        # Extract sequence data
        pixel_values = inputs.get('pixel_values')
        input_ids = inputs.get('input_ids')
        labels = inputs.get('labels')
        inventory_embeddings = inputs.get('inventory_embeddings')
        attention_mask = inputs.get('attention_mask')
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = self.args.device
        
        # Detect non-empty frames for loss weighting
        # This helps with class imbalance (most frames have empty inventory)
        is_non_empty = None
        if inventory_embeddings is not None:
            with torch.no_grad():
                is_non_empty = self._detect_non_empty_inventory(inventory_embeddings)
        
        # Initialize memory
        memory = model.init_memory(batch_size, device)
        
        # Track total loss for return
        total_loss = torch.tensor(0.0, device=device)
        
        # Track metrics
        empty_frame_count = 0
        nonempty_frame_count = 0
        
        # Enable gradient computation
        with self.compute_loss_context_manager():
            for chunk_start in range(0, seq_len, self.bptt_chunk_size):
                chunk_end = min(chunk_start + self.bptt_chunk_size, seq_len)
                
                # Forward through chunk
                chunk_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                for t in range(chunk_start, chunk_end):
                    frame_t = self._extract_timestep_data(pixel_values, t)
                    input_ids_t = input_ids[:, t, :]
                    labels_t = labels[:, t, :]
                    inv_emb_t = inventory_embeddings[:, t, :] if inventory_embeddings is not None else None
                    attn_mask_t = attention_mask[:, t, :] if attention_mask is not None else None
                    
                    # Check if this frame is non-empty
                    frame_is_non_empty = is_non_empty[:, t] if is_non_empty is not None else None
                    if frame_is_non_empty is not None:
                        if frame_is_non_empty.any():
                            nonempty_frame_count += 1
                        else:
                            empty_frame_count += 1
                    
                    # Forward
                    outputs = model(
                        input_ids=input_ids_t,
                        pixel_values=frame_t,
                        prev_memory=memory,
                        attention_mask=attn_mask_t,
                        labels=labels_t,
                        inventory_embeddings=inv_emb_t,
                    )
                    
                    # Accumulate loss
                    if outputs.loss is not None:
                        # Normalize by total sequence length for stable gradients
                        step_loss = outputs.loss / seq_len
                        
                        # Weight non-empty frames more heavily
                        # Weight non‑empty frames more heavily because the dataset is heavily imbalanced
                        # (most frames have empty inventory). Without weighting, the model could ignore
                        # rare non‑empty events and simply learn to always predict "empty".
                        if frame_is_non_empty is not None and self.non_empty_loss_weight != 1.0:
                            # Create weight tensor: non_empty_weight for non-empty, 1.0 for empty
                            frame_weight = torch.where(
                                frame_is_non_empty,
                                torch.tensor(self.non_empty_loss_weight, dtype=step_loss.dtype, device=step_loss.device),
                                torch.tensor(1.0, dtype=step_loss.dtype, device=step_loss.device)
                            )
                            step_loss = step_loss * frame_weight.mean()  # Average across batch
                        
                        chunk_loss = chunk_loss + step_loss
                    
                    # Update memory within chunk (gradients flow)
                    memory = outputs.new_memory
                
                # Backward for this chunk
                if self.args.gradient_accumulation_steps > 1:
                    chunk_loss = chunk_loss / self.args.gradient_accumulation_steps
                
                chunk_loss.backward()
                
                # Accumulate total loss
                total_loss = total_loss + chunk_loss.detach()
                
                # DETACH memory for next chunk (truncated BPTT)
                # This is the key: gradients don't flow between chunks
                memory = memory.detach()
        
        # Log frame distribution
        if empty_frame_count + nonempty_frame_count > 0:
            self.log({
                'train/empty_frames': empty_frame_count,
                'train/nonempty_frames': nonempty_frame_count,
                'train/nonempty_ratio': nonempty_frame_count / (empty_frame_count + nonempty_frame_count),
            })
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        
        return total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Prediction step for evaluation.
        
        During evaluation, we process sequences without gradients
        and track inventory prediction accuracy.
        """
        # Remove auxiliary targets
        inputs.pop('inventory_embeddings', None)
        
        # Check if sequence data
        has_sequence = 'input_ids' in inputs and len(inputs['input_ids'].shape) >= 3
        
        if not has_sequence:
            # Single-step evaluation
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs.loss
            return (loss, None, None) if prediction_loss_only else (loss, outputs.logits, inputs.get('labels'))
        
        # Sequence evaluation
        with torch.no_grad():
            loss, eval_outputs = self._compute_sequence_loss_truncated_bptt(
                model, inputs, return_outputs=True
            )
        
        return (loss, None, None) if prediction_loss_only else (loss, None, None)
    
    def evaluate_inventory_retention(
        self,
        eval_dataset,
        gap_sizes: List[int] = [10, 50, 100, 200],
    ) -> Dict[str, float]:
        """
        Evaluate how well the model retains inventory information over gaps.
        
        This measures the model's ability to remember inventory contents
        across frames when the inventory is not visible.
        
        Args:
            eval_dataset: Evaluation dataset with sequences
            gap_sizes: List of gap sizes to test (in frames)
        
        Returns:
            Dictionary of metrics by gap size
        """
        self.model.eval()
        
        results = {}
        device = self.args.device
        
        for gap in gap_sizes:
            similarities = []
            
            for example in eval_dataset:
                frames = example.get('frames')
                inventory_targets = example.get('inventory_embeddings')
                
                if len(frames) < gap + 2:
                    continue
                
                with torch.no_grad():
                    # Initialize memory
                    memory = self.model.init_memory(1, device)
                    
                    # Process frames up to gap
                    for t in range(gap + 1):
                        frame_t = frames[t:t+1] if frames else None
                        inv_t = inventory_targets[t:t+1] if t == 0 else None
                        
                        outputs = self.model(
                            pixel_values=frame_t,
                            prev_memory=memory,
                            inventory_embeddings=inv_t,
                        )
                        memory = outputs.new_memory
                    
                    # Check inventory prediction at gap
                    if outputs.inventory_embedding is not None:
                        pred_emb = outputs.inventory_embedding[0]  # [36, 768]
                        target_emb = inventory_targets[gap]  # [36, 768]
                        
                        # Compute cosine similarity per slot
                        sim = torch.nn.functional.cosine_similarity(
                            pred_emb, target_emb, dim=-1
                        )
                        similarities.append(sim.mean().item())
            
            if similarities:
                results[f'inventory_retention_gap_{gap}'] = np.mean(similarities)
        
        return results


class SequenceDataCollator:
    """
    Data collator for sequence-based training.
    
    Converts individual samples into batched sequences for BPTT training.
    
    BERT can embed any text, so new items or NBT data can be described in the
    inventory string. The frozen encoder ensures a stable target space; later we
    may replace the MLP head with structured prediction to better handle novel
    items. The JarvisVLA paper uses a similar frozen encoder approach for some tasks.
    """
    
    def __init__(
        self,
        processor,
        max_seq_length: int = 512,
        sequence_length: int = 200,  # Frames per sequence
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.sequence_length = sequence_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of sequences.
        
        Args:
            examples: List of sequence dicts from VPTSequenceDataset
        
        Returns:
            Batched tensors:
            - pixel_values: [batch, seq_len, processed_image_shape]
            - input_ids: [batch, seq_len, text_seq_len]
            - labels: [batch, seq_len, text_seq_len]
            - inventory_embeddings: [batch, seq_len, 768]
            - attention_mask: [batch, seq_len, text_seq_len]
        """
        batch_size = len(examples)
        
        # Collect all sequences
        all_pixel_values = []
        all_input_ids = []
        all_labels = []
        all_inventory_embeddings = []
        all_attention_masks = []
        
        for example in examples:
            frames = example['frames']
            texts = example['texts']
            inventory_embs = example['inventory_embeddings']  # [seq_len, 768]
            
            seq_pixel_values = []
            seq_input_ids = []
            seq_labels = []
            seq_attention_masks = []
            
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
                
                seq_pixel_values.append(processed.get('pixel_values'))
                seq_input_ids.append(processed['input_ids'][0])
                seq_attention_masks.append(processed['attention_mask'][0])
                
                # Labels = input_ids shifted (for next token prediction)
                labels = processed['input_ids'][0].clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                seq_labels.append(labels)
            
            # Stack sequence
            if seq_pixel_values[0] is not None:
                all_pixel_values.append(torch.stack([pv[0] for pv in seq_pixel_values]))
            all_input_ids.append(torch.stack(seq_input_ids))
            all_attention_masks.append(torch.stack(seq_attention_masks))
            all_labels.append(torch.stack(seq_labels))
            all_inventory_embeddings.append(inventory_embs)  # Already tensor [seq_len, 768]
        
        # Batch
        batch = {
            'input_ids': torch.stack(all_input_ids),  # [batch, seq_len, text_len]
            'attention_mask': torch.stack(all_attention_masks),  # [batch, seq_len, text_len]
            'labels': torch.stack(all_labels),  # [batch, seq_len, text_len]
            'inventory_embeddings': torch.stack(all_inventory_embeddings),  # [batch, seq_len, 768]
        }
        
        if all_pixel_values:
            batch['pixel_values'] = torch.stack(all_pixel_values)  # [batch, seq_len, ...]
        
        return batch
