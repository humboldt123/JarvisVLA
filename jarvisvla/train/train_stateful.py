"""
Training script for Stateful JarvisVLA with Recurrent Memory Token.

This script trains a stateful VLA model that:
1. Maintains a recurrent memory token (512-dim) across timesteps
2. Uses W_in/W_out projections to/from model hidden dimension (3584)
3. Appends memory token after visual/text tokens
4. Uses truncated BPTT with chunk size 16
5. Trains auxiliary inventory head with BERT targets

Truncated BPTT with chunk size 16 allows the model to learn long-term dependencies
via the carried memory, while keeping the backward graph shallow enough for a 7B
model. Gradient checkpointing further reduces memory footprint. This follows common
practice for training recurrent transformers (e.g., Recurrent Memory Transformer).

Usage:
    python -m jarvisvla.train.train_stateful \
        --model_name_or_path CraftJarvis/JarvisVLA-Qwen2-VL-7B \
        --output_dir ./checkpoints/stateful_vla \
        --sequence_length 200 \
        --batch_size 2 \
        --bptt_chunk_size 16 \
        --memory_dim 512

Key Arguments:
    --memory_dim: Dimension of recurrent memory (default: 512)
    --bptt_chunk_size: Truncated BPTT chunk size (default: 16)
    --inventory_loss_weight: Weight for auxiliary loss (default: 0.1)
    --freeze_base_model: Freeze base VLM, train only memory components
"""

import os
import sys
import json
import argparse
import pathlib
from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from jarvisvla.models.stateful_vla import wrap_model_for_stateful_training
from jarvisvla.train.sequence_dataset import (
    InventoryTextEncoder,
    VPTSequenceDataset,
    VPTSequenceDataCollator,
)
from jarvisvla.train.stateful_trainer import StatefulVLATrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stateful JarvisVLA with recurrent memory token"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="CraftJarvis/JarvisVLA-Qwen2-VL-7B",
        help="Path to pretrained JarvisVLA model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints",
    )
    
    # Memory architecture arguments
    parser.add_argument(
        "--memory_dim",
        type=int,
        default=512,
        help="Dimension of recurrent memory (default: 512)",
    )
    parser.add_argument(
        "--use_mlp_for_W_in",
        action="store_true",
        help="Use 2-layer MLP for W_in instead of linear",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/vvm33/vpt_contractor",
        help="Directory containing OpenVPT dataset",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=200,
        help="Number of consecutive frames per sequence",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Stride between sequences",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to load (for testing)",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (number of sequences per batch)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for new parameters (memory projections, heads)",
    )
    parser.add_argument(
        "--base_model_learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for base model (if not frozen)",
    )
    parser.add_argument(
        "--freeze_base_model",
        action="store_true",
        help="Freeze base VLM parameters, train only memory components",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    
    # BPTT arguments
    parser.add_argument(
        "--bptt_chunk_size",
        type=int,
        default=16,
        help="Truncated BPTT chunk size (default: 16). "
             "Gradients flow through chunks but not between chunks.",
    )
    parser.add_argument(
        "--inventory_loss_weight",
        type=float,
        default=0.1,
        help="Weight for inventory auxiliary loss",
    )
    parser.add_argument(
        "--non_empty_loss_weight",
        type=float,
        default=5.0,
        help="Weight multiplier for non-empty inventory frames (default: 5.0). "
             "The OpenVPT dataset is heavily imbalanced (most frames have empty inventory). "
             "This weight ensures the model learns to predict rare non-empty events.",
    )
    
    # Optimization arguments
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on base model",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizer(model: nn.Module, args):
    """
    Create optimizer with different learning rates for different components.
    
    New components (memory projections, heads): higher LR
    Base model: lower LR (or frozen)
    """
    base_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if this is a new component
        if any(x in name for x in ['memory_projections', 'initial_memory', 'inventory_embedding_head']):
            new_params.append(param)
        else:
            base_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': new_params, 'lr': args.learning_rate, 'name': 'new'},
    ]
    
    if not args.freeze_base_model and base_params:
        param_groups.append({
            'params': base_params,
            'lr': args.base_model_learning_rate,
            'name': 'base',
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    print("Optimizer parameter groups:")
    print(f"  New components: {len(new_params)} tensors, lr={args.learning_rate}")
    if base_params:
        print(f"  Base model: {len(base_params)} tensors, lr={args.base_model_learning_rate}")
    
    return optimizer


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(pathlib.Path(args.output_dir) / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load processor
    print(f"\nLoading processor from {args.model_name_or_path}...")
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_name_or_path,
        do_rescale=False,
        patch_size=14,
        vision_feature_select_strategy="default",
    )
    
    # Add special tokens if needed
    special_tokens_path = pathlib.Path(__file__).parent.parent / "special_tokens.json"
    if special_tokens_path.exists():
        with open(special_tokens_path) as f:
            special_tokens = json.load(f)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load base model
    print(f"\nLoading base model: {args.model_name_or_path}")
    print(f"  This is JarvisVLA-Qwen2-VL-7B with visual encoder + image projector + LM")
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    if torch.cuda.device_count() == 1:
        base_model = base_model.to(device)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled on base model")
    
    # Wrap with stateful wrapper
    print(f"\nWrapping with StatefulJarvisVLA:")
    print(f"  Memory dim: {args.memory_dim}")
    print(f"  Hidden dim: 3584 (Qwen2VL-7B)")
    print(f"  W_in: {args.memory_dim} -> 3584")
    print(f"  W_out: 3584 -> {args.memory_dim}")
    
    model = wrap_model_for_stateful_training(
        base_model=base_model,
        memory_dim=args.memory_dim,
        add_inventory_head=True,
        inventory_head_kwargs={
            'output_dim': 768,
            'num_slots': 36,
            'dropout': 0.1,
        },
    )
    
    # Freeze base model if requested
    if args.freeze_base_model:
        print("\nFreezing base model parameters (training only memory components)...")
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
    
    # Print trainable parameters
    param_counts = model.get_trainable_parameters()
    print("\nTrainable parameters:")
    for key, count in param_counts.items():
        if key != 'base_model':
            print(f"  {key}: {count:,}")
    print(f"  Fraction of total: {param_counts['fraction_new']*100:.4f}%")
    
    # Create inventory encoder (BERT)
    print(f"\nLoading BERT encoder for inventory targets...")
    inventory_encoder = InventoryTextEncoder(
        model_name="bert-base-uncased",
        device=str(device),
    )
    
    # Create dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = VPTSequenceDataset(
        data_dir=args.data_dir,
        inventory_encoder=inventory_encoder,
        sequence_length=args.sequence_length,
        stride=args.stride,
        max_sequences=args.max_sequences,
    )
    
    print(f"Dataset loaded: {len(dataset)} sequences")
    print(f"  Sequence length: {args.sequence_length} frames")
    print(f"  BPTT chunk size: {args.bptt_chunk_size} (truncated BPTT)")
    
    # Create collator
    collator = VPTSequenceDataCollator(
        processor=processor,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=False,  # We handle this manually on base model
        max_grad_norm=args.max_grad_norm,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        # Note: We don't use the Trainer's built-in evaluation
        evaluation_strategy="no",
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, args)
    
    # Create trainer
    print(f"\nInitializing StatefulVLATrainer:")
    print(f"  BPTT chunk size: {args.bptt_chunk_size}")
    print(f"  Inventory loss weight: {args.inventory_loss_weight}")
    
    trainer = StatefulVLATrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        optimizers=(optimizer, None),
        inventory_loss_weight=args.inventory_loss_weight,
        bptt_chunk_size=args.bptt_chunk_size,
        max_grad_norm=args.max_grad_norm,
        non_empty_loss_weight=args.non_empty_loss_weight,
    )
    
    # Check for existing checkpoint
    last_checkpoint = None
    if args.resume_from_checkpoint:
        last_checkpoint = args.resume_from_checkpoint
    elif pathlib.Path(args.output_dir).exists():
        last_checkpoint = get_last_checkpoint(args.output_dir)
    
    # Train
    print("\n" + "="*60)
    print("Starting training with truncated BPTT")
    print("="*60)
    print(f"Memory token is appended after visual/text tokens")
    print(f"Gradients flow through chunks of {args.bptt_chunk_size} frames")
    print(f"Memory carries between chunks but gradients are truncated")
    print("="*60 + "\n")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    print(f"\nSaving final model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    
    # Save memory projections
    proj_path = pathlib.Path(args.output_dir) / "memory_projections.pt"
    torch.save(model.memory_projections.state_dict(), proj_path)
    print(f"Memory projections saved to {proj_path}")
    
    # Save initial memory
    init_mem_path = pathlib.Path(args.output_dir) / "initial_memory.pt"
    torch.save(model.initial_memory.data, init_mem_path)
    print(f"Initial memory saved to {init_mem_path}")
    
    # Save config
    config = {
        'memory_dim': args.memory_dim,
        'hidden_dim': 3584,
        'base_model_name': args.model_name_or_path,
        'bptt_chunk_size': args.bptt_chunk_size,
        'inventory_loss_weight': args.inventory_loss_weight,
    }
    with open(pathlib.Path(args.output_dir) / "stateful_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoint: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
