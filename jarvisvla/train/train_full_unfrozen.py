#!/usr/bin/env python3
"""
FULL UNFROZEN Stateful JarvisVLA Training with DeepSpeed ZeRO-3

This script trains EVERYTHING:
- Vision Encoder (ViT) - learns to see inventory items
- Qwen2VL LLM (7B) - learns to understand inventory state
- Memory projections (W_in, W_out) - learns memory dynamics
- Inventory embedding head - learns to predict inventory

DeepSpeed ZeRO-3 shards optimizer states across GPUs so this fits in memory.
"""

import torch
import torch.nn.functional as F
import argparse
import json
import os
import sys
import pathlib
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import deepspeed
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from jarvisvla.models.stateful_vla import wrap_model_for_stateful_training
from jarvisvla.train import InventoryTextEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training (passed by DeepSpeed)")
    parser.add_argument("--data_dir", type=str, default="/data/vvm33/vpt_contractor")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_jsonls", type=int, default=4000)
    parser.add_argument("--test_jsonls", type=int, default=400)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--bptt_chunk_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--base_model_lr", type=float, default=1e-6)
    parser.add_argument("--inventory_weight", type=float, default=0.1)
    parser.add_argument("--non_empty_weight", type=float, default=5.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="CraftJarvis/JarvisVLA-Qwen2-VL-7B")
    parser.add_argument("--memory_dim", type=int, default=512)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config_zero3.json")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    rank = deepspeed.comm.get_rank()
    world_size = deepspeed.comm.get_world_size()
    
    if rank == 0:
        print("="*70)
        print("FULL UNFROZEN STATEFUL JARVISVLA")
        print("="*70)
        print(f"GPUs: {world_size}")
        print(f"Output: {args.output_dir}")
        print(f"Training: ViT + LLM + Memory + Inventory Head (ALL UNFROZEN)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load encoder (only on rank 0, then broadcast)
    if rank == 0:
        print("\nLoading BERT encoder...")
    encoder = InventoryTextEncoder(device=f"cuda:{rank}")
    
    # Load model
    if rank == 0:
        print("Loading base model...")
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen2VLProcessor.from_pretrained(args.model_name)
    
    # Wrap with stateful
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
    
    # Create parameter groups with discriminative LR
    param_groups = [
        {'params': model.memory_projections.parameters(), 'lr': args.learning_rate},
        {'params': model.inventory_embedding_head.parameters(), 'lr': args.learning_rate},
        {'params': model.base_model.parameters(), 'lr': args.base_model_lr},
    ]
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=param_groups,
    )
    
    # Get empty embedding
    device = torch.device(f"cuda:{rank}")
    with torch.no_grad():
        empty_inputs = encoder.tokenizer("empty inventory", return_tensors="pt", padding=True).to(encoder.device)
        empty_emb = encoder.model(**empty_inputs).last_hidden_state[0, 0, :]
        empty_emb = F.normalize(empty_emb, dim=-1).to(device)
    
    # Load data
    from jarvisvla.train.run_overnight_eval import OnDemandSequenceLoader
    jsonl_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    train_files = jsonl_files[:args.train_jsonls]
    test_files = jsonl_files[args.train_jsonls:args.train_jsonls + args.test_jsonls]
    train_loader = OnDemandSequenceLoader(train_files, 50, encoder)
    
    # Training loop
    if rank == 0:
        print(f"\nTraining for {args.train_steps} steps...")
    
    losses = []
    
    for step in range(args.train_steps):
        model_engine.train()
        model_engine.module.train_aux_heads(True)
        
        # Get random sequence
        seq_idx = torch.randint(0, len(train_loader), (1,)).item()
        seq = train_loader[seq_idx]
        
        chunk_losses = []
        memory = model_engine.module.init_memory(1, device)
        
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
            
            outputs = model_engine(
                input_ids=inputs.get('input_ids'),
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                prev_memory=memory,
                attention_mask=inputs.get('attention_mask'),
                labels=inputs.get('input_ids'),
                inventory_embeddings=inv_target,
            )
            
            if outputs.loss is not None:
                loss = outputs.loss / args.bptt_chunk_size
                
                if is_non_empty:
                    loss = loss * args.non_empty_weight
                
                model_engine.backward(loss)
                chunk_losses.append(loss.item())
            
            memory = outputs.new_memory.detach()
        
        if chunk_losses:
            model_engine.step()
            losses.append(sum(chunk_losses))
        
        # Logging
        if rank == 0 and (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else 0.0
            print(f"  Step {step+1}/{args.train_steps}: loss={avg_loss:.4f}")
        
        # Checkpoint
        if (step + 1) % args.eval_every == 0 and rank == 0:
            print(f"\n  Saving checkpoint at step {step+1}...")
            model_engine.save_checkpoint(args.output_dir, tag=f"step_{step+1}")
    
    if rank == 0:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
