#!/usr/bin/env python3
"""
Plot training metrics from JSON Lines log file.

Usage:
    python plot_metrics.py /data/vvm33/checkpoints/full_unfrozen_fsdp/metrics.jsonl
    python plot_metrics.py metrics.jsonl -o training_plot.png
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_file):
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


def smooth(data, window=20):
    if len(data) < 2:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_vals = [v for v in data[start:i+1] if v is not None and not (isinstance(v, float) and np.isnan(v))]
        smoothed.append(sum(window_vals) / len(window_vals) if window_vals else float('nan'))
    return smoothed


def plot_series(ax, steps, values, label, color, title, ylabel, logy=False, hline=None):
    raw = [v if v is not None else float('nan') for v in values]
    sm  = smooth(raw)
    ax.plot(steps, raw, alpha=0.25, color=color, linewidth=0.6)
    ax.plot(steps, sm,  color=color, linewidth=2, label=f'Smoothed (w=20)')
    if hline is not None:
        ax.axhline(y=hline, linestyle='--', color='red', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        try:
            ax.set_yscale('log')
        except Exception:
            pass
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def make_random_walk(losses, step=1.0):
    """Random walk seeded at the first valid loss value, stepping ±U(0, step) each step."""
    rng = np.random.default_rng()
    rw = [None] * len(losses)
    first = next((i for i, v in enumerate(losses) if v is not None), None)
    if first is None:
        return rw
    rw[first] = losses[first]
    for i in range(first + 1, len(losses)):
        rw[i] = rw[i - 1] + rng.uniform(-step, step)
    return rw


def create_plot(metrics, output_file):
    steps      = [m['step']                        for m in metrics]
    losses     = [m.get('loss')                    for m in metrics]
    lm_losses  = [m['lm_loss']  if m.get('lm_loss',  0) > 0 else None for m in metrics]
    type_losses= [m['type_loss'] if m.get('type_loss', 0) > 0 else None for m in metrics]
    cnt_losses = [m['count_loss'] if m.get('count_loss', 0) > 0 else None for m in metrics]
    step_times = [m.get('step_time', 0)            for m in metrics]

    rw_losses  = make_random_walk(cnt_losses, step=0.3)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle('JarvisVLA Training — LM + type/count inventory heads', fontsize=14, fontweight='bold')

    # Row 0
    plot_series(axes[0,0], steps, losses,      'loss',      'steelblue',  'Total Loss',                    'Loss',    logy=True)
    plot_series(axes[0,1], steps, lm_losses,   'lm_loss',   'darkorange', 'LM Loss (inventory description CE)', 'Loss', logy=False)

    # Row 1
    plot_series(axes[1,0], steps, type_losses, 'type_loss', 'darkorchid', 'Type Loss (InfoNCE)',            'Loss',    logy=False)
    plot_series(axes[1,1], steps, cnt_losses,  'cnt_loss',  'seagreen',   'Count Loss (log-MSE)',           'Loss',    logy=False)

    # Row 2 — grad norm + random walk reference for count loss
    update_steps_gn = [m['step'] for m in metrics if m.get('is_update_step') and m.get('grad_norm', 0) > 0]
    update_gn_vals  = [m['grad_norm'] for m in metrics if m.get('is_update_step') and m.get('grad_norm', 0) > 0]
    ax = axes[2,0]
    if update_steps_gn:
        ax.scatter(update_steps_gn, update_gn_vals, s=12, color='tomato', alpha=0.5, zorder=3)
        ax.plot(update_steps_gn, smooth(update_gn_vals), color='tomato', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Gradient Norm (update steps only)')
    ax.grid(True, alpha=0.3)

    plot_series(axes[2,1], steps, rw_losses, 'rw', 'slategray',
                'Random ±0.3/step Walk ref\n(same start, same axis as count loss)', 'Loss', logy=False)
    axes[2,1].set_ylim(axes[1,1].get_ylim())

    # Row 3 — step time + summary
    plot_series(axes[3,1], steps, step_times, 'time', 'gray', 'Step Time', 'Seconds', logy=False)

    # Update-step markers on loss plot
    update_steps = [m['step'] for m in metrics if m.get('is_update_step')]
    if update_steps:
        update_losses = [losses[steps.index(s)] for s in update_steps if losses[steps.index(s)] is not None]
        axes[0,0].scatter(update_steps, update_losses, s=8, color='red', zorder=5, alpha=0.4, label='Update step')
        axes[0,0].legend(fontsize=8)

    ax = axes[3,0]
    ax.axis('off')
    valid_losses = [v for v in losses if v is not None]
    valid_lm     = [v for v in lm_losses  if v is not None]
    valid_type   = [v for v in type_losses if v is not None]
    valid_cnt    = [v for v in cnt_losses  if v is not None]
    lines = [
        f"Steps logged:     {len(metrics)}",
        f"Loss  init/final: {valid_losses[0]:.4f} → {valid_losses[-1]:.4f}",
        f"Loss  min:        {min(valid_losses):.4f}",
    ]
    if valid_lm:
        lines += [
            f"LMLoss  avg:      {np.mean(valid_lm):.4f}",
            f"LMLoss  min:      {min(valid_lm):.4f}",
        ]
    if valid_type:
        lines += [
            f"TypeLoss avg:     {np.mean(valid_type):.4f}",
            f"TypeLoss min:     {min(valid_type):.4f}",
        ]
    if valid_cnt:
        lines += [
            f"CntLoss  avg:     {np.mean(valid_cnt):.4f}",
            f"CntLoss  min:     {min(valid_cnt):.4f}",
        ]
    if step_times:
        lines.append(f"Avg step time:    {np.mean(step_times):.2f}s")
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_file')
    parser.add_argument('--output', '-o', default=None)
    args = parser.parse_args()

    path = Path(args.metrics_file)
    if not path.exists():
        print(f"Error: {path} not found")
        return

    metrics = load_metrics(path)
    print(f"Loaded {len(metrics)} entries")
    if not metrics:
        print("No metrics found!")
        return

    output = args.output or str(path.with_suffix('.png'))
    create_plot(metrics, output)

    valid = [m for m in metrics if m.get('loss') is not None]
    losses = [m['loss'] for m in valid]
    print(f"\nSummary:")
    print(f"  Steps:  {len(metrics)}")
    print(f"  Loss:   {losses[0]:.4f} → {losses[-1]:.4f}  (min {min(losses):.4f})")


if __name__ == '__main__':
    main()
