#!/usr/bin/env python3
"""
Plot training metrics from JSON Lines log file.

Usage:
    python plot_metrics.py /data/vvm33/checkpoints/full_unfrozen_proper/metrics.jsonl
    python plot_metrics.py metrics.jsonl --output training_plot.png
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_file):
    """Load metrics from JSON Lines file."""
    metrics = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metrics


def smooth(data, window=10):
    """Smooth data with moving average."""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(sum(data[start:i+1]) / (i - start + 1))
    return smoothed


def create_plot(metrics, output_file):
    """Create comprehensive training plot."""
    
    # Extract data
    steps = [m['step'] for m in metrics]
    losses = [m.get('loss', 0) for m in metrics]
    inv_losses = [m.get('inventory_loss', 0) for m in metrics]
    cos_sims = [m.get('cosine_similarity', 0) for m in metrics]
    empty_accs = [m.get('empty_accuracy', 0) for m in metrics]
    grad_norms = [m.get('grad_norm', 0) for m in metrics]
    step_times = [m.get('step_time', 0) for m in metrics]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('FSDP Training Metrics - Full Unfrozen Model (8.3B params)', fontsize=14, fontweight='bold')
    
    # 1. Loss curve
    ax = axes[0, 0]
    ax.plot(steps, losses, alpha=0.3, color='blue', linewidth=0.5)
    ax.plot(steps, smooth(losses, 20), color='red', linewidth=2, label='Smoothed (20-step)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations for min loss
    if losses:
        min_idx = losses.index(min(losses))
        ax.annotate(f'Min: {losses[min_idx]:.4f}', 
                    xy=(steps[min_idx], losses[min_idx]),
                    xytext=(steps[min_idx], losses[min_idx] * 2),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=9, color='green')
    
    # 2. Inventory Loss
    ax = axes[0, 1]
    ax.plot(steps, inv_losses, alpha=0.3, color='purple', linewidth=0.5)
    ax.plot(steps, smooth(inv_losses, 20), color='darkviolet', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Inventory Loss')
    ax.set_title('Auxiliary Inventory Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Cosine Similarity
    ax = axes[1, 0]
    ax.plot(steps, cos_sims, alpha=0.3, color='green', linewidth=0.5)
    ax.plot(steps, smooth(cos_sims, 20), color='darkgreen', linewidth=2)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Typical empty threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Predicted vs Target Inventory Similarity')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Empty vs Non-Empty Accuracy
    ax = axes[1, 1]
    ax.plot(steps, empty_accs, alpha=0.3, color='orange', linewidth=0.5)
    ax.plot(steps, smooth(empty_accs, 20), color='darkorange', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Empty/Non-Empty Classification Accuracy')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 5. Gradient Norm
    ax = axes[2, 0]
    ax.plot(steps, grad_norms, alpha=0.3, color='brown', linewidth=0.5)
    ax.plot(steps, smooth(grad_norms, 20), color='darkred', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm (clipped at 1.0)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 6. Step Time
    ax = axes[2, 1]
    ax.plot(steps, step_times, alpha=0.3, color='gray', linewidth=0.5)
    ax.plot(steps, smooth(step_times, 20), color='black', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time per Step')
    ax.grid(True, alpha=0.3)
    
    # Add stats text
    if losses:
        stats_text = (
            f"Steps: {len(steps)}\n"
            f"Initial Loss: {losses[0]:.4f}\n"
            f"Final Loss: {losses[-1]:.4f}\n"
            f"Min Loss: {min(losses):.4f}\n"
            f"Avg Cosine Sim: {np.mean(cos_sims):.4f}\n"
            f"Avg Empty Acc: {np.mean(empty_accs):.2%}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_file', type=str, help='Path to metrics.jsonl')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output PNG file')
    args = parser.parse_args()
    
    if not Path(args.metrics_file).exists():
        print(f"Error: File not found: {args.metrics_file}")
        return
    
    print(f"Loading metrics from {args.metrics_file}...")
    metrics = load_metrics(args.metrics_file)
    print(f"Loaded {len(metrics)} metric entries")
    
    if not metrics:
        print("No metrics found!")
        return
    
    # Default output name
    if args.output is None:
        args.output = str(Path(args.metrics_file).with_suffix('.png'))
    
    create_plot(metrics, args.output)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Steps: {len(metrics)}")
    print(f"  Loss range: {min(m['loss'] for m in metrics):.4f} - {max(m['loss'] for m in metrics):.4f}")
    if 'cosine_similarity' in metrics[0]:
        print(f"  Avg cosine similarity: {np.mean([m['cosine_similarity'] for m in metrics]):.4f}")
    if 'empty_accuracy' in metrics[0]:
        print(f"  Avg empty accuracy: {np.mean([m['empty_accuracy'] for m in metrics]):.2%}")


if __name__ == '__main__':
    main()
