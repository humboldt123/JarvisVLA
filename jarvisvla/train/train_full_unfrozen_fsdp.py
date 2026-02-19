#!/usr/bin/env python3
"""
PROPER training script for FULLY unfrozen StatefulJarvisVLA (8.3B params) using PyTorch FSDP.

Features:
- SHARDED_STATE_DICT for safe checkpointing
- Comprehensive metrics logging (JSON Lines format)
- Empty vs non-empty accuracy tracking
- Cosine similarity metrics
- Gradient norm monitoring
- Automatic plotting script included
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer

sys.path.insert(0, '/home/vvm33/JarvisVLA')
from jarvisvla.models.stateful_vla import wrap_model_for_stateful_training


class MetricsLogger:
    """Logs training metrics in JSON Lines format for easy plotting."""
    
    def __init__(self, log_file: str, rank: int):
        self.log_file = log_file
        self.rank = rank
        self.metrics_history = []
        
        if rank == 0:
            # Create directory if needed
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            # Write header
            with open(log_file, 'w') as f:
                f.write(f"# Training metrics started at {datetime.now().isoformat()}\n")
    
    def log(self, metrics: Dict, step: int):
        """Log metrics for a step."""
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)
        
        if self.rank == 0:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
    
    def log_summary(self, summary: Dict):
        """Log final summary."""
        if self.rank == 0:
            with open(self.log_file, 'a') as f:
                f.write(f"# SUMMARY: {json.dumps(summary)}\n")


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_fsdp_model(model, device_id):
    """Wrap model with FSDP."""
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    def wrap_policy_fn(module):
        if isinstance(module, Qwen2VLDecoderLayer):
            return True
        return False
    
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=wrap_policy_fn)
    
    model = FSDP(
        model,
        device_id=device_id,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    
    return model


def compute_inventory_metrics(predicted_emb, target_emb, empty_emb, threshold=0.95):
    """
    Compute inventory prediction metrics.
    
    Returns:
        Dict with accuracy, cosine similarity, empty/non-empty classification metrics
    """
    metrics = {}
    
    # Cosine similarity
    cos_sim = F.cosine_similarity(predicted_emb.flatten(), target_emb.flatten(), dim=0)
    metrics['cosine_similarity'] = cos_sim.item()
    
    # Classify as empty or non-empty
    pred_sim_to_empty = F.cosine_similarity(predicted_emb.flatten(), empty_emb, dim=0)
    target_sim_to_empty = F.cosine_similarity(target_emb.flatten(), empty_emb, dim=0)
    
    pred_is_empty = pred_sim_to_empty.item() > threshold
    target_is_empty = target_sim_to_empty.item() > threshold
    
    metrics['pred_empty'] = 1.0 if pred_is_empty else 0.0
    metrics['target_empty'] = 1.0 if target_is_empty else 0.0
    metrics['empty_accuracy'] = 1.0 if pred_is_empty == target_is_empty else 0.0
    
    return metrics


def save_checkpoint(model, step, losses, metrics_logger, output_dir, rank):
    """Save FSDP checkpoint using SHARDED_STATE_DICT (safe for training)."""
    if rank != 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{step}.pt')
    
    # Use SHARDED_STATE_DICT - doesn't corrupt training
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model_state_dict = model.state_dict()
    
    torch.save({
        'step': step,
        'model_state_dict': model_state_dict,
        'losses': losses,
    }, checkpoint_path)
    
    print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/vvm33/vpt_contractor')
    parser.add_argument('--output_dir', type=str, default='/data/vvm33/checkpoints/full_unfrozen_proper')
    parser.add_argument('--train_jsonls', type=int, default=4000)
    parser.add_argument('--test_jsonls', type=int, default=400)
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--base_model_lr', type=float, default=1e-6)
    parser.add_argument('--bptt_chunk_size', type=int, default=4)
    parser.add_argument('--memory_dim', type=int, default=512)
    parser.add_argument('--inventory_weight', type=float, default=0.1)
    parser.add_argument('--non_empty_weight', type=float, default=5.0)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--log_every', type=int, default=10, help='Log metrics every N steps')
    args = parser.parse_args()
    
    # Setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Metrics logger
    metrics_file = os.path.join(args.output_dir, 'metrics.jsonl')
    logger = MetricsLogger(metrics_file, rank)
    
    if rank == 0:
        print(f"="*70)
        print(f"PROPER FSDP Training: {world_size} GPUs")
        print(f"Output: {args.output_dir}")
        print(f"Metrics: {metrics_file}")
        print(f"="*70)
    
    # Load model
    if rank == 0:
        print("\n[1/5] Loading Qwen2-VL-7B...")
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
        trust_remote_code=True,
    )
    
    model = wrap_model_for_stateful_training(
        base_model=base_model,
        memory_dim=args.memory_dim,
        add_inventory_head=True,
        inventory_head_kwargs={'output_dim': 768},
    )
    
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    
    if rank == 0:
        print(f"  Total params: {total_params:,}")
        print(f"  All parameters unfrozen (ViT + LLM + memory + inventory)")
    
    # FSDP setup
    model = model.to(device, dtype=torch.bfloat16)
    model = get_fsdp_model(model, device_id=local_rank)
    
    # Optimizer
    param_groups = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.base_model_lr},
    ]
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'memory_projections' in name or 'inventory_embedding_head' in name:
                param_groups[0]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)
    
    optimizer = torch.optim.AdamW(param_groups)
    
    if rank == 0:
        print(f"\n[2/5] Optimizer:")
        print(f"  New params (memory+inventory): LR={args.learning_rate}")
        print(f"  Base params (ViT+LLM): LR={args.base_model_lr}")
    
    # Resume if needed
    start_step = 0
    losses = []
    if args.resume_from and os.path.exists(args.resume_from):
        if rank == 0:
            print(f"\n[2.5/5] Resuming from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('step', 0)
        losses = checkpoint.get('losses', [])
    
    # Data setup
    if rank == 0:
        print(f"\n[3/5] Loading data...")
    
    from jarvisvla.train.sequence_dataset import InventoryTextEncoder
    from jarvisvla.train.run_overnight_eval import OnDemandSequenceLoader
    
    encoder = InventoryTextEncoder(device=device)
    
    with torch.no_grad():
        empty_inputs = encoder.tokenizer("empty inventory", return_tensors="pt", padding=True).to(device)
        empty_emb = encoder.model(**empty_inputs).last_hidden_state[0, 0, :]
        empty_emb = F.normalize(empty_emb, dim=-1)
    
    jsonl_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    train_files = jsonl_files[:args.train_jsonls]
    train_loader = OnDemandSequenceLoader(train_files, 50, encoder)
    
    if rank == 0:
        print(f"  Training sequences: {len(train_loader)}")
    
    # Training loop
    if rank == 0:
        print(f"\n[4/5] Training from step {start_step} to {args.train_steps}...")
        print(f"  Logging every {args.log_every} steps")
        print(f"  Checkpointing every {args.eval_every} steps")
        print(f"\n{'Step':>8} {'Loss':>10} {'InvLoss':>10} {'CosSim':>8} {'EmptyAcc':>8} {'Time':>8}")
        print("-"*70)
    
    step_times = []
    
    for step in range(start_step, args.train_steps):
        step_start = time.time()
        
        model.train()
        model.train_aux_heads(True)  # Enable inventory head
        chunk_losses = []
        chunk_inv_losses = []
        chunk_metrics = []
        last_answer_text = ""
        
        # Initialize memory
        memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)
        
        # Random phrases for inventory text generation
        INV_PHRASES = [
            "My inventory consists of",
            "The contents of my inventory are",
            "I have in my possession",
            "Currently carrying",
            "In my backpack I have",
            "My items include",
            "I am holding",
        ]
        
        # Get random sequence
        seq_idx = torch.randint(0, len(train_loader), (1,)).item()
        seq = train_loader[seq_idx]
        start_idx = torch.randint(0, max(1, len(seq['frames']) - args.bptt_chunk_size), (1,)).item()
        
        for t in range(start_idx, min(start_idx + args.bptt_chunk_size, len(seq['frames']))):
            frame = seq['frames'][t]
            inv_target = seq['inventory_embeddings'][t:t+1].to(device)
            
            # Get raw inventory data for text generation
            raw_inventory = seq['ticks'][t]['inventory'] if 'ticks' in seq else []
            
            # Check if empty
            target_emb = inv_target[0].mean(dim=0)
            sim_to_empty = F.cosine_similarity(target_emb.unsqueeze(0), empty_emb.unsqueeze(0), dim=-1).item()
            is_non_empty = sim_to_empty < 0.95
            
            # Generate text label from inventory data
            # Format: "[random phrase]: [count] [item][plural?], ..."
            if raw_inventory:
                phrase = INV_PHRASES[torch.randint(0, len(INV_PHRASES), (1,)).item()]
                items_text = []
                for item in raw_inventory:
                    name = item.get('type', 'unknown')
                    count = item.get('quantity', 1)
                    plural = 's' if count > 1 and not name.endswith('s') else ''
                    items_text.append(f"{count} {name}{plural}")
                answer_text = f"{phrase}: {', '.join(items_text)}"
            else:
                answer_text = "My inventory is empty"
            
            # Forward
            inputs = processor(
                text="What is in the inventory?",
                images=frame,
                return_tensors="pt",
                padding='max_length',
                max_length=128,
                truncation=True,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # CRITICAL DESIGN CHOICE:
            # We intentionally set labels=None to DISABLE language modeling loss.
            # The text "What is in the inventory?" is just a prompt to process the image,
            # but we DON'T have ground truth text answers ("You have: sword, dirt...").
            # Using labels=input_ids would train the model to predict the QUESTION text,
            # not the inventory contents! Instead, we ONLY train on the auxiliary
            # inventory embedding head which learns to predict BERT embeddings of
            # the actual inventory state. This is a workaround for having no text labels.
            
            # NEW APPROACH: Generate text labels from inventory data!
            # Instead of training on embeddings only, we generate synthetic text answers
            # from the structured inventory data. Format:
            # "[random phrase] [item count] [item name] [plural?], ..."
            # Examples:
            #   "My inventory consists of: 5 dirt, 1 sword, 3 planks"
            #   "The contents of my inventory are: 2 apples, 1 pickaxe"
            # Random phrases: "My inventory consists of", "The contents of my inventory are",
            #                 "I have in my possession", "Currently carrying:", etc.
            # Item format: "{count} {name}{'s' if count>1 else ''}"
            # labels=None,  # DISABLED - we don't have ground truth text answers!
            
            # Tokenize the answer text to use as labels
            answer_inputs = processor(
                text=answer_text,
                return_tensors="pt",
                padding='max_length',
                max_length=128,
                truncation=True,
            )
            labels = answer_inputs['input_ids'].to(device)
            last_answer_text = answer_text  # Save for logging
            
            outputs = model(
                input_ids=inputs.get('input_ids'),
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                prev_memory=memory,
                attention_mask=inputs.get('attention_mask'),
                labels=labels,  # Now using generated answer text!
                inventory_embeddings=inv_target,
            )
            
            # Combined loss: main LM loss + auxiliary inventory loss
            # Main loss comes from predicting the answer text ("My inventory consists of...")
            # Inventory loss comes from matching BERT embeddings
            if outputs.loss is not None:
                loss = outputs.loss / (args.bptt_chunk_size * args.grad_accum_steps)
                
                if is_non_empty:
                    loss = loss * args.non_empty_weight
                
                loss.backward()
                
                # Log losses
                raw_loss = outputs.loss.item()
                chunk_losses.append(raw_loss)
                
                if outputs.inventory_loss is not None:
                    chunk_inv_losses.append(outputs.inventory_loss.item())
                
                # Compute inventory metrics
                if hasattr(outputs, 'inventory_embedding') and outputs.inventory_embedding is not None:
                    inv_metrics = compute_inventory_metrics(
                        outputs.inventory_embedding[0].mean(dim=0),
                        inv_target[0].mean(dim=0),
                        empty_emb
                    )
                    chunk_metrics.append(inv_metrics)
            
            memory = outputs.new_memory.detach()
        
        # Optimizer step
        if chunk_losses:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(sum(chunk_losses))
            
            # Aggregate metrics
            avg_loss = sum(chunk_losses)
            avg_inv_loss = sum(chunk_inv_losses) / len(chunk_inv_losses) if chunk_inv_losses else 0.0
            avg_cos_sim = sum(m['cosine_similarity'] for m in chunk_metrics) / len(chunk_metrics) if chunk_metrics else 0.0
            avg_empty_acc = sum(m['empty_accuracy'] for m in chunk_metrics) / len(chunk_metrics) if chunk_metrics else 0.0
            pct_empty = sum(m['target_empty'] for m in chunk_metrics) / len(chunk_metrics) if chunk_metrics else 0.0
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Log metrics
            metrics = {
                'loss': avg_loss,
                'inventory_loss': avg_inv_loss,
                'cosine_similarity': avg_cos_sim,
                'empty_accuracy': avg_empty_acc,
                'pct_empty_frames': pct_empty,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'step_time': step_time,
            }
            
            logger.log(metrics, step)
            
            # Print
            if rank == 0 and step % args.log_every == 0:
                print(f"{step:>8} {avg_loss:>10.4f} {avg_inv_loss:>10.4f} {avg_cos_sim:>8.4f} "
                      f"{avg_empty_acc:>8.2%} {step_time:>8.2f}s")
                
                # Occasionally show example answer text
                if step % 50 == 0 and last_answer_text:
                    print(f"         Example: {last_answer_text[:80]}...")
        
        # Checkpoint (safe with SHARDED_STATE_DICT)
        if (step + 1) % args.eval_every == 0:
            save_checkpoint(model, step + 1, losses, logger, args.output_dir, rank)
    
    # Final save
    save_checkpoint(model, args.train_steps, losses, logger, args.output_dir, rank)
    
    # Summary
    if rank == 0 and losses:
        summary = {
            'total_steps': len(losses),
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'avg_step_time': sum(step_times) / len(step_times),
        }
        logger.log_summary(summary)
        
        print(f"\n[5/5] Training Complete!")
        print(f"  Initial loss: {summary['initial_loss']:.4f}")
        print(f"  Final loss: {summary['final_loss']:.4f}")
        print(f"  Min loss: {summary['min_loss']:.4f}")
        print(f"  Avg step time: {summary['avg_step_time']:.2f}s")
        print(f"\nMetrics saved to: {metrics_file}")
        print(f"Plot with: python jarvisvla/train/plot_metrics.py {metrics_file}")
    
    cleanup_distributed()


"""
## About master_port

The master_port (default: 29500) is the TCP port used by PyTorch distributed training 
for inter-process communication. When you run multi-GPU training with torchrun, the 
different processes need to find each other and coordinate.

--master_port=29501  # Use this if 29500 is already in use (e.g., from a crashed run)

If you get "address already in use" error, just increment the port number.
"""

if __name__ == '__main__':
    main()
