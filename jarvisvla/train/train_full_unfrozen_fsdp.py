#!/usr/bin/env python3
"""
Training script for FULLY unfrozen StatefulJarvisVLA (8.3B params) using PyTorch FSDP.
Trains ViT + LLM + memory projections + inventory head together.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json

# FSDP imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
import functools
import torch.distributed as dist

# Model imports
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer

# Project imports
sys.path.insert(0, '/home/vvm33/JarvisVLA')
from jarvisvla.models.stateful_vla import StatefulJarvisVLA, wrap_model_for_stateful_training


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
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_fsdp_model(model, device_id):
    """Wrap model with FSDP."""
    
    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Auto-wrap policy - wrap transformer layers but NOT embeddings
    def wrap_policy_fn(module):
        # Wrap transformer decoder layers
        if isinstance(module, Qwen2VLDecoderLayer):
            return True
        return False
    
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=wrap_policy_fn)
    
    # FSDP wrapper
    model = FSDP(
        model,
        device_id=device_id,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=None,  # Keep on GPU for speed
        limit_all_gathers=True,
        use_orig_params=True,  # Fix for autograd view issues
    )
    
    return model


class SimpleSequenceDataset:
    """Simple dataset that loads sequences on demand."""
    
    def __init__(self, jsonl_files, cache_size, encoder, data_dir):
        self.jsonl_files = jsonl_files
        self.cache_size = cache_size
        self.encoder = encoder
        self.data_dir = Path(data_dir)
        
    def __len__(self):
        return len(self.jsonl_files)
    
    def __getitem__(self, idx):
        """Load a single sequence."""
        from jarvisvla.train.run_overnight_eval import OnDemandSequenceLoader
        # Create temporary loader for this file
        loader = OnDemandSequenceLoader([self.jsonl_files[idx]], 1, self.encoder, str(self.data_dir))
        return loader[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/vvm33/vpt_contractor')
    parser.add_argument('--output_dir', type=str, default='/data/vvm33/checkpoints/full_unfrozen_fsdp')
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
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"FSDP Training: {world_size} GPUs")
        print(f"Output: {args.output_dir}")
    
    # Load base model
    if rank == 0:
        print("Loading Qwen2-VL-7B...")
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
        trust_remote_code=True,
    )
    
    # Create stateful model with inventory head
    model = wrap_model_for_stateful_training(
        base_model=base_model,
        memory_dim=args.memory_dim,
        add_inventory_head=True,
        inventory_head_kwargs={'output_dim': 768},
    )
    
    # UNFREEZE EVERYTHING
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        print(f"Total params: {total_params:,}")
        print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print("  ✓ Vision Encoder (ViT) - UNFROZEN")
        print("  ✓ Qwen2VL LLM (7B) - UNFROZEN")
        print("  ✓ Memory projections - UNFROZEN")
        print("  ✓ Inventory head - UNFROZEN")
    
    # Move to device and convert to bfloat16 before FSDP
    model = model.to(device, dtype=torch.bfloat16)
    
    # Wrap with FSDP
    if rank == 0:
        print("Wrapping with FSDP...")
    model = get_fsdp_model(model, device_id=local_rank)
    
    # Create optimizer with discriminative LR
    # Need to get unwrapped model for parameter groups
    param_groups = [
        {'params': [], 'lr': args.learning_rate, 'name': 'new_params'},  # memory + inventory
        {'params': [], 'lr': args.base_model_lr, 'name': 'base_model'},  # ViT + LLM
    ]
    
    # Separate parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'memory_projections' in name or 'inventory_embedding_head' in name:
                param_groups[0]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Load encoder for inventory embeddings
    from jarvisvla.train.sequence_dataset import InventoryTextEncoder
    encoder = InventoryTextEncoder(device=device)
    
    # Get empty embedding reference
    with torch.no_grad():
        empty_inputs = encoder.tokenizer("empty inventory", return_tensors="pt", padding=True).to(device)
        empty_emb = encoder.model(**empty_inputs).last_hidden_state[0, 0, :]
        empty_emb = F.normalize(empty_emb, dim=-1)
    
    # Load data
    if rank == 0:
        print("Loading data...")
    
    from jarvisvla.train.run_overnight_eval import OnDemandSequenceLoader
    jsonl_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    train_files = jsonl_files[:args.train_jsonls]
    test_files = jsonl_files[args.train_jsonls:args.train_jsonls + args.test_jsonls]
    train_loader = OnDemandSequenceLoader(train_files, 50, encoder)
    
    if rank == 0:
        print(f"\nTraining for {args.train_steps} steps...")
    
    losses = []
    
    for step in range(args.train_steps):
        model.train()
        
        chunk_losses = []
        # Initialize memory manually
        memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)
        
        # Get random sequence
        seq_idx = torch.randint(0, len(train_loader), (1,)).item()
        seq = train_loader[seq_idx]
        
        start_idx = torch.randint(0, max(1, len(seq['frames']) - args.bptt_chunk_size), (1,)).item()
        
        for t in range(start_idx, min(start_idx + args.bptt_chunk_size, len(seq['frames']))):
            frame = seq['frames'][t]
            inv_target = seq['inventory_embeddings'][t:t+1].to(device)
            
            # Check if non-empty
            target_emb = inv_target[0].mean(dim=0)
            sim_to_empty = F.cosine_similarity(target_emb.unsqueeze(0), empty_emb.unsqueeze(0), dim=-1).item()
            is_non_empty = sim_to_empty < 0.95
            
            # Forward
            inputs = processor(
                text="What is in the inventory?",
                images=frame,
                return_tensors="pt",
                padding='max_length',
                max_length=128,
                truncation=True,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            outputs = model(
                input_ids=inputs.get('input_ids'),
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                prev_memory=memory,
                attention_mask=inputs.get('attention_mask'),
                labels=inputs.get('input_ids'),
                inventory_embeddings=inv_target,
            )
            
            if outputs.loss is not None:
                loss = outputs.loss / (args.bptt_chunk_size * args.grad_accum_steps)
                
                if is_non_empty:
                    loss = loss * args.non_empty_weight
                
                # Standard backward
                loss.backward()
                chunk_losses.append(loss.item() * (args.bptt_chunk_size * args.grad_accum_steps))
            
            memory = outputs.new_memory.detach()
        
        # Optimizer step
        if chunk_losses:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(sum(chunk_losses))
        
        # Logging
        if rank == 0 and step % 10 == 0 and losses:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            print(f"Step {step}/{args.train_steps}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (step + 1) % args.eval_every == 0 and rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_step_{step+1}.pt')
            
            # Save full state dict
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                state_dict = model.state_dict()
            
            torch.save({
                'step': step + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
    
    cleanup_distributed()
    
    if rank == 0:
        print("\nTraining complete!")


if __name__ == '__main__':
    main()
