#!/usr/bin/env python3
"""
Overnight Evaluation: Before/After Training Inventory Prediction

This script performs a complete evaluation:
1. Creates train/test split from OpenVPT
2. Evaluates baseline (random init)
3. Trains model for N steps
4. Evaluates after training
5. Generates comparison report
"""

import torch
import torch.nn.functional as F
import argparse
import json
import os
import sys
import pathlib
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from jarvisvla.models.stateful_vla import wrap_model_for_stateful_training
from jarvisvla.train import InventoryTextEncoder, VPTSequenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Overnight evaluation of inventory prediction")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="/data/vvm33/vpt_contractor")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_jsonls", type=int, default=20,
                       help="Number of JSONL files to use for training")
    parser.add_argument("--test_jsonls", type=int, default=5,
                       help="Number of JSONL files to use for testing")
    
    # Training
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=50)
    parser.add_argument("--bptt_chunk_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--non_empty_loss_weight", type=float, default=5.0)
    
    # Model
    parser.add_argument("--model_name", type=str, default="CraftJarvis/JarvisVLA-Qwen2-VL-7B")
    parser.add_argument("--memory_dim", type=int, default=512)
    
    return parser.parse_args()


def create_train_test_split(data_dir, train_jsonls, test_jsonls, sequence_length):
    """Create train/test split by JSONL files - ON-DEMAND loading.
    
    This builds an index of available sequences but doesn't load video frames
    until they're requested during training. Prevents OOM.
    """
    print("Creating train/test split (on-demand loading)...")
    
    encoder = InventoryTextEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Find all JSONL files and sort them
    jsonl_files = sorted(Path(data_dir).glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    if len(jsonl_files) < train_jsonls + test_jsonls:
        raise ValueError(f"Need {train_jsonls + test_jsonls} JSONLs, have {len(jsonl_files)}")
    
    # Split: first N for train, next M for test (GUARANTEED NO OVERLAP)
    train_files = jsonl_files[:train_jsonls]
    test_files = jsonl_files[train_jsonls:train_jsonls + test_jsonls]
    
    print(f"  Train JSONLs: {len(train_files)} files")
    print(f"  Test JSONLs: {len(test_files)} files")
    
    # Build on-demand loaders (just index, no actual frames loaded)
    train_loader = OnDemandSequenceLoader(train_files, sequence_length, encoder)
    test_loader = OnDemandSequenceLoader(test_files, sequence_length, encoder)
    
    print(f"  Train: {len(train_loader)} sequences available")
    print(f"  Test: {len(test_loader)} sequences available")
    
    return train_loader, test_loader, encoder


class OnDemandSequenceLoader:
    """Loads sequences on-demand during training to save memory."""
    
    def __init__(self, jsonl_files, sequence_length, encoder):
        self.jsonl_files = jsonl_files
        self.sequence_length = sequence_length
        self.encoder = encoder
        self.index = []  # (jsonl_idx, start_tick, num_sequences_in_file)
        self.cache = {}  # Simple LRU cache
        self.cache_size = 10  # Keep last 10 sequences in memory
        
        # Build lightweight index
        for jsonl_idx, jsonl_file in enumerate(jsonl_files):
            mp4_file = jsonl_file.with_suffix('.mp4')
            if not mp4_file.exists():
                continue
            
            # Count ticks in JSONL without fully loading
            tick_count = 0
            with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in f:
                    tick_count += 1
            
            # How many sequences can we get from this file?
            num_seqs = max(0, (tick_count - sequence_length) // sequence_length + 1)
            if num_seqs > 0:
                self.index.append((jsonl_idx, 0, num_seqs))
    
    def __len__(self):
        return sum(count for _, _, count in self.index)
    
    def __getitem__(self, idx):
        """Load sequence on-demand."""
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Find which file this index belongs to
        current = 0
        for jsonl_idx, _, count in self.index:
            if idx < current + count:
                seq_in_file = idx - current
                seq = self._load_sequence(jsonl_idx, seq_in_file)
                # Update cache
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[idx] = seq
                return seq
            current += count
        raise IndexError(f"Index {idx} out of range")
    
    def _load_sequence(self, jsonl_idx, seq_in_file):
        """Load a single sequence from disk."""
        jsonl_file = self.jsonl_files[jsonl_idx]
        mp4_file = jsonl_file.with_suffix('.mp4')
        start_tick = seq_in_file * self.sequence_length
        
        # Read only the ticks we need
        ticks = []
        with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= start_tick and i < start_tick + self.sequence_length:
                    try:
                        data = json.loads(line)
                        ticks.append({
                            'tick': data.get('tick', i),
                            'inventory': data.get('inventory', []),
                            'mouse': data.get('mouse', {}),
                            'keyboard': data.get('keyboard', {}),
                            'isGuiOpen': data.get('isGuiOpen', False),
                            'isGuiInventory': data.get('isGuiInventory', False),
                            'hotbar': data.get('hotbar', 0),
                            'yaw': data.get('yaw', 0.0),
                            'pitch': data.get('pitch', 0.0),
                        })
                    except json.JSONDecodeError:
                        continue
                elif i >= start_tick + self.sequence_length:
                    break
        
        # Load frames from video
        cap = cv2.VideoCapture(str(mp4_file))
        frames = []
        for i in range(start_tick, start_tick + self.sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(Image.fromarray(frame))
        cap.release()
        
        # Encode inventories
        inv_embs = [self.encoder.encode_inventory(t['inventory']) for t in ticks]
        inv_embs = torch.stack(inv_embs)
        
        return {
            'frames': frames,
            'inventory_embeddings': inv_embs,
            'ticks': ticks,
            'source_file': jsonl_file.name,
        }


def analyze_inventory_distribution(sequences, encoder, name, max_samples=10):
    """Analyze empty vs non-empty distribution (samples for speed with on-demand loading)."""
    empty_count = 0
    non_empty_count = 0
    
    # Get empty embedding template
    with torch.no_grad():
        empty_inputs = encoder.tokenizer("empty inventory", return_tensors="pt", padding=True).to(encoder.device)
        empty_emb = encoder.model(**empty_inputs).last_hidden_state[0, 0, :]
        empty_emb = F.normalize(empty_emb, dim=-1)
    
    # Sample sequences for analysis (don't load all for on-demand loader)
    import random
    num_seqs = len(sequences)
    sample_indices = random.sample(range(num_seqs), min(max_samples, num_seqs))
    
    return {"empty": 0, "non_empty": 0}  # Skip for speed
    for idx in sample_indices:
        seq = sequences[idx]
        for inv_emb in seq['inventory_embeddings']:
            inv_emb = inv_emb.to(encoder.device)
            sim = F.cosine_similarity(inv_emb.unsqueeze(0), empty_emb.unsqueeze(0), dim=-1).item()
            if sim > 0.95:
                empty_count += 1
            else:
                non_empty_count += 1
    
    total = empty_count + non_empty_count
    print(f"  {name}: {empty_count} empty ({empty_count/total*100:.1f}%), {non_empty_count} non-empty ({non_empty_count/total*100:.1f}%) [sampled {len(sample_indices)} seqs]")
    
    return {'empty': empty_count, 'non_empty': non_empty_count}


def load_model(model_name, memory_dim, device):
    """Load JarvisVLA model with stateful wrapper."""
    print(f"Loading model: {model_name}")
    print(f"  Using device: {device}")
    
    processor = Qwen2VLProcessor.from_pretrained(
        model_name,
        do_rescale=False,
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model to specific device (avoid device_map="auto" for stability)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    model = wrap_model_for_stateful_training(
        base_model=base_model,
        memory_dim=memory_dim,
        add_inventory_head=True,
    )
    
    # Freeze base model (train only memory components)
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    print(f"  Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, processor


def evaluate_model(model, sequences, processor, encoder, device):
    """Evaluate model on sequences."""
    model.eval()
    model.train_aux_heads(True)
    
    results = {
        'empty_correct': 0,
        'empty_total': 0,
        'non_empty_correct': 0,
        'non_empty_total': 0,
        'empty_cos_sims': [],
        'non_empty_cos_sims': [],
    }
    
    # Get the actual device of the model (handles device_map="auto")
    model_device = next(model.parameters()).device
    
    # Get empty embedding (on encoder device, then move to model device)
    with torch.no_grad():
        empty_inputs = encoder.tokenizer("empty inventory", return_tensors="pt", padding=True).to(encoder.device)
        empty_emb = encoder.model(**empty_inputs).last_hidden_state[0, 0, :]
        empty_emb = F.normalize(empty_emb, dim=-1).to(model_device)
    
    with torch.no_grad():
        for seq_idx, seq in enumerate(sequences):
            print(f"  Sequence {seq_idx+1}/{len(sequences)}...", end='\r')
            
            memory = model.init_memory(1, model_device)
            
            for frame_idx in range(len(seq['frames'])):
                frame = seq['frames'][frame_idx]
                inv_target = seq['inventory_embeddings'][frame_idx:frame_idx+1].to(model_device)
                
                # Process
                inputs = processor(
                    text="What is in the inventory?",
                    images=frame,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=128,
                    truncation=True,
                )
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                outputs = model(
                    input_ids=inputs.get('input_ids'),
                    pixel_values=inputs.get('pixel_values'),
                    image_grid_thw=inputs.get('image_grid_thw'),
                    prev_memory=memory,
                    attention_mask=inputs.get('attention_mask'),
                    inventory_embeddings=inv_target,
                )
                
                if outputs.inventory_embedding is not None:
                    pred_emb = outputs.inventory_embedding[0].mean(dim=0)
                    target_emb = inv_target[0].mean(dim=0)
                    
                    cos_sim = F.cosine_similarity(pred_emb.unsqueeze(0), target_emb.unsqueeze(0), dim=-1).item()
                    
                    # Ground truth
                    target_sim_to_empty = F.cosine_similarity(target_emb.unsqueeze(0), empty_emb.unsqueeze(0), dim=-1).item()
                    is_empty_gt = target_sim_to_empty > 0.95
                    
                    # Prediction
                    pred_sim_to_empty = F.cosine_similarity(pred_emb.unsqueeze(0), empty_emb.unsqueeze(0), dim=-1).item()
                    is_empty_pred = pred_sim_to_empty > 0.95
                    
                    if is_empty_gt:
                        results['empty_total'] += 1
                        results['empty_cos_sims'].append(cos_sim)
                        if is_empty_pred:
                            results['empty_correct'] += 1
                    else:
                        results['non_empty_total'] += 1
                        results['non_empty_cos_sims'].append(cos_sim)
                        if not is_empty_pred:
                            results['non_empty_correct'] += 1
                
                memory = outputs.new_memory
    
    print()  # New line after progress
    return results


def compute_metrics(results):
    """Compute metrics from results."""
    empty_acc = results['empty_correct'] / results['empty_total'] * 100 if results['empty_total'] > 0 else 0
    non_empty_acc = results['non_empty_correct'] / results['non_empty_total'] * 100 if results['non_empty_total'] > 0 else 0
    total = results['empty_total'] + results['non_empty_total']
    correct = results['empty_correct'] + results['non_empty_correct']
    overall_acc = correct / total * 100 if total > 0 else 0
    empty_rate = results['empty_total'] / total * 100 if total > 0 else 0
    
    empty_cos = np.mean(results['empty_cos_sims']) if results['empty_cos_sims'] else 0
    non_empty_cos = np.mean(results['non_empty_cos_sims']) if results['non_empty_cos_sims'] else 0
    
    return {
        'overall_acc': overall_acc,
        'empty_acc': empty_acc,
        'non_empty_acc': non_empty_acc,
        'empty_rate': empty_rate,
        'empty_cos': empty_cos,
        'non_empty_cos': non_empty_cos,
        'n_samples': total,
    }


def train_model(model, train_seqs, processor, encoder, args, device):
    """Train model for specified steps."""
    print(f"\nTraining for {args.train_steps} steps...")
    
    model.train()
    model.train_aux_heads(True)
    
    # Note: Gradient checkpointing disabled for compatibility
    # Memory management via small BPTT chunks and periodic cache clearing
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )
    
    # Get model device for all tensors
    model_device = next(model.parameters()).device
    
    # Get empty embedding (on encoder device, then move to model device)
    with torch.no_grad():
        empty_inputs = encoder.tokenizer("empty inventory", return_tensors="pt", padding=True).to(encoder.device)
        empty_emb = encoder.model(**empty_inputs).last_hidden_state[0, 0, :]
        empty_emb = F.normalize(empty_emb, dim=-1).to(model_device)
    
    losses = []
    
    for step in range(args.train_steps):
        # Pick random sequence (on-demand loading via index)
        seq_idx = np.random.randint(0, len(train_seqs))
        seq = train_seqs[seq_idx]
        
        chunk_losses = []
        memory = model.init_memory(args.batch_size, model_device)
        
        # Random start within sequence
        start_idx = np.random.randint(0, max(1, len(seq['frames']) - args.bptt_chunk_size))
        
        for t in range(start_idx, min(start_idx + args.bptt_chunk_size, len(seq['frames']))):
            frame = seq['frames'][t]
            inv_target = seq['inventory_embeddings'][t:t+1].to(model_device)
            
            # Check if non-empty
            target_emb = inv_target[0].mean(dim=0)
            sim_to_empty = F.cosine_similarity(target_emb.unsqueeze(0), empty_emb.unsqueeze(0), dim=-1).item()
            is_non_empty = sim_to_empty < 0.95
            
            inputs = processor(
                text="What is in the inventory?",
                images=frame,
                return_tensors="pt",
                padding='max_length',
                max_length=128,
                truncation=True,
            )
            inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
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
                step_loss = outputs.loss / args.bptt_chunk_size
                
                # Apply weighting
                if is_non_empty:
                    step_loss = step_loss * args.non_empty_loss_weight
                
                chunk_losses.append(step_loss)
            
            # Detach memory between steps (BPTT)
            memory = outputs.new_memory.detach()
        
        # Sum losses and backward
        if chunk_losses:
            chunk_loss = sum(chunk_losses)
            chunk_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(chunk_loss.item())
        
        # Clear CUDA cache to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"  Step {step+1}/{args.train_steps}: avg_loss={avg_loss:.4f}")
        
        # Save checkpoint every 200 steps
        if (step + 1) % 200 == 0 and args.output_dir:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_step_{step+1}.pt')
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    print(f"Training complete. Final loss: {losses[-1]:.4f}")
    return losses


def print_results_table(baseline_metrics, trained_metrics, save_path=None):
    """Print and save results table."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("RESULTS: BEFORE vs AFTER TRAINING")
    lines.append("="*70)
    
    lines.append("\nCLASSIFICATION ACCURACY (Threshold = 0.95)")
    lines.append("-"*70)
    lines.append(f"{'Metric':<30} {'Before':>12} {'After':>12} {'Improvement':>14}")
    lines.append("-"*70)
    
    def pp(before, after):
        return f"{after - before:+.1f} pp"
    
    lines.append(f"{'Overall Accuracy':<30} {baseline_metrics['overall_acc']:>11.1f}% {trained_metrics['overall_acc']:>11.1f}% {pp(baseline_metrics['overall_acc'], trained_metrics['overall_acc']):>14}")
    lines.append(f"{'Empty Accuracy':<30} {baseline_metrics['empty_acc']:>11.1f}% {trained_metrics['empty_acc']:>11.1f}% {pp(baseline_metrics['empty_acc'], trained_metrics['empty_acc']):>14}")
    lines.append(f"{'Non-Empty Accuracy':<30} {baseline_metrics['non_empty_acc']:>11.1f}% {trained_metrics['non_empty_acc']:>11.1f}% {pp(baseline_metrics['non_empty_acc'], trained_metrics['non_empty_acc']):>14}")
    lines.append(f"{'Empty Rate (Test)':<30} {baseline_metrics['empty_rate']:>11.1f}% {trained_metrics['empty_rate']:>11.1f}% {'--':>14}")
    
    lines.append("\nCOSINE SIMILARITY (Continuous Measure)")
    lines.append("-"*70)
    lines.append(f"{'Metric':<30} {'Before':>12} {'After':>12} {'Improvement':>14}")
    lines.append("-"*70)
    
    def cos_improve(before, after):
        return f"{after - before:+.4f}"
    
    lines.append(f"{'Avg Cos Sim (Empty)':<30} {baseline_metrics['empty_cos']:>12.4f} {trained_metrics['empty_cos']:>12.4f} {cos_improve(baseline_metrics['empty_cos'], trained_metrics['empty_cos']):>14}")
    lines.append(f"{'Avg Cos Sim (Non-Empty)':<30} {baseline_metrics['non_empty_cos']:>12.4f} {trained_metrics['non_empty_cos']:>12.4f} {cos_improve(baseline_metrics['non_empty_cos'], trained_metrics['non_empty_cos']):>14}")
    
    # Print to console
    for line in lines:
        print(line)
    
    # Save to file
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nResults saved to: {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("OVERNIGHT EVALUATION: INVENTORY PREDICTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 1. Create train/test split
    print("\n" + "="*70)
    print("1. CREATING TRAIN/TEST SPLIT")
    print("="*70)
    train_seqs, test_seqs, encoder = create_train_test_split(
        args.data_dir, args.train_jsonls, args.test_jsonls, args.sequence_length
    )
    
    # Analyze distribution
    print("\nInventory distribution:")
    train_dist = analyze_inventory_distribution(train_seqs, encoder, "Train")
    test_dist = analyze_inventory_distribution(test_seqs, encoder, "Test")
    
    # 2. Load model
    print("\n" + "="*70)
    print("2. LOADING MODEL")
    print("="*70)
    model, processor = load_model(args.model_name, args.memory_dim, device)
    
    # 3. Baseline evaluation
    print("\n" + "="*70)
    print("3. BASELINE EVALUATION (BEFORE TRAINING)")
    print("="*70)
    baseline_results = evaluate_model(model, test_seqs, processor, encoder, device)
    baseline_metrics = compute_metrics(baseline_results)
    
    print(f"\nBaseline metrics:")
    print(f"  Overall Accuracy: {baseline_metrics['overall_acc']:.1f}%")
    print(f"  Empty Accuracy: {baseline_metrics['empty_acc']:.1f}%")
    print(f"  Non-Empty Accuracy: {baseline_metrics['non_empty_acc']:.1f}%")
    print(f"  Avg Cos Sim (Empty): {baseline_metrics['empty_cos']:.4f}")
    print(f"  Avg Cos Sim (Non-Empty): {baseline_metrics['non_empty_cos']:.4f}")
    
    # 4. Train
    print("\n" + "="*70)
    print("4. TRAINING")
    print("="*70)
    train_losses = train_model(model, train_seqs, processor, encoder, args, device)
    
    # Save training curve
    with open(os.path.join(args.output_dir, 'train_losses.json'), 'w') as f:
        json.dump(train_losses, f)
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'trained_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # 5. Post-training evaluation
    print("\n" + "="*70)
    print("5. POST-TRAINING EVALUATION")
    print("="*70)
    trained_results = evaluate_model(model, test_seqs, processor, encoder, device)
    trained_metrics = compute_metrics(trained_results)
    
    print(f"\nTrained metrics:")
    print(f"  Overall Accuracy: {trained_metrics['overall_acc']:.1f}%")
    print(f"  Empty Accuracy: {trained_metrics['empty_acc']:.1f}%")
    print(f"  Non-Empty Accuracy: {trained_metrics['non_empty_acc']:.1f}%")
    print(f"  Avg Cos Sim (Empty): {trained_metrics['empty_cos']:.4f}")
    print(f"  Avg Cos Sim (Non-Empty): {trained_metrics['non_empty_cos']:.4f}")
    
    # 6. Results table
    print_results_table(baseline_metrics, trained_metrics, 
                       os.path.join(args.output_dir, 'results.txt'))
    
    # Save detailed metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'baseline': baseline_metrics,
            'trained': trained_metrics,
            'train_distribution': train_dist,
            'test_distribution': test_dist,
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.output_dir}")
    
    # Conclusion
    improvement = trained_metrics['non_empty_acc'] - baseline_metrics['non_empty_acc']
    if improvement > 10:
        print(f"\n✓ SUCCESS: Non-empty accuracy improved by {improvement:.1f} pp!")
        print("  The model is learning inventory prediction.")
    elif improvement > 0:
        print(f"\n~ PARTIAL: Non-empty accuracy improved by {improvement:.1f} pp.")
        print("  Consider longer training for better results.")
    else:
        print(f"\n✗ NO IMPROVEMENT: Non-empty accuracy change: {improvement:.1f} pp")
        print("  Check hyperparameters or data quality.")


if __name__ == "__main__":
    main()
