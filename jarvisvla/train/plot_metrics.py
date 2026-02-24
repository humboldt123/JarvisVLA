#!/usr/bin/env python3
"""
Plot training metrics from JSON Lines log file.

Usage:
    python jarvisvla/train/plot_metrics.py /data/vvm33/checkpoints/train_inv_head/metrics.jsonl
    python jarvisvla/train/plot_metrics.py metrics.jsonl -o training_plot.png
"""

import json
import math
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


def plot_series(ax, steps, values, label, color, title, ylabel, logy=False):
    raw = [v if v is not None else float('nan') for v in values]
    sm  = smooth(raw)
    ax.plot(steps, raw, alpha=0.2, color=color, linewidth=0.6)
    ax.plot(steps, sm,  color=color, linewidth=2, label=label)
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


def create_plot(metrics, output_file):
    steps      = [m['step']                  for m in metrics]
    type_losses= [m.get('type_loss')         for m in metrics]
    cnt_losses = [m.get('count_loss')        for m in metrics]
    n_ne_vals  = [m.get('n_ne_avg', 0)       for m in metrics]
    n_gui_vals = [m.get('n_gui_frames', 0)   for m in metrics]
    step_times = [m.get('step_time', 0)      for m in metrics]
    mem_norms  = [m.get('memory_norm', 0)    for m in metrics]

    # cos_sim: recovered from type_loss, or stored directly
    cos_all    = [2.0 * math.exp(-m['type_loss']) - 1.0
                  if m.get('type_loss') else None for m in metrics]
    cos_gui    = [m.get('cos_sim_gui')    for m in metrics]
    cos_closed = [m.get('cos_sim_closed') for m in metrics]

    update_steps = [m['step'] for m in metrics if m.get('is_update_step')]
    update_gn    = [m['grad_norm']          for m in metrics if m.get('is_update_step') and m.get('grad_norm', 0) > 0]
    update_gn_s  = [m['step']              for m in metrics if m.get('is_update_step') and m.get('grad_norm', 0) > 0]
    update_hgn   = [m['inv_head_grad_norm'] for m in metrics if m.get('is_update_step')]

    fig, axes = plt.subplots(5, 2, figsize=(16, 25))
    fig.suptitle('JarvisVLA — Inventory Memory Training', fontsize=14, fontweight='bold')

    # ── Row 0: Type loss  |  GUI vs Closed cos_sim (THE key diagnostic) ──────
    plot_series(axes[0,0], steps, type_losses, 'type loss', 'darkorchid',
                'Type Loss  (-log cosine, all 36 slots)', 'Loss')

    ax = axes[0,1]
    cos_all_v    = [v if v is not None else float('nan') for v in cos_all]
    cos_gui_v    = [v if v is not None else float('nan') for v in cos_gui]
    cos_closed_v = [v if v is not None else float('nan') for v in cos_closed]
    ax.plot(steps, cos_all_v,    alpha=0.15, color='gray',        linewidth=0.5)
    ax.plot(steps, smooth(cos_all_v),    color='gray',        linewidth=1.5, label='cos_sim (all)',    linestyle='--')
    ax.plot(steps, cos_gui_v,    alpha=0.15, color='steelblue',   linewidth=0.5)
    ax.plot(steps, smooth(cos_gui_v),    color='steelblue',   linewidth=2,   label='cos_sim GUI open')
    ax.plot(steps, cos_closed_v, alpha=0.15, color='tomato',      linewidth=0.5)
    ax.plot(steps, smooth(cos_closed_v), color='tomato',      linewidth=2,   label='cos_sim GUI closed (memory)')
    ax.axhline(y=0, linestyle=':', color='black', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('cos_sim')
    ax.set_title('GUI-open vs Closed cos_sim\n'
                 'KEY: closed (red) should rise as memory learns to retain inventory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 1: Count loss  |  Global grad norm ───────────────────────────────
    plot_series(axes[1,0], steps, cnt_losses, 'count loss', 'seagreen',
                'Count Loss  (log-MSE, non-empty slots only)', 'Loss')

    ax = axes[1,1]
    if update_gn_s:
        ax.scatter(update_gn_s, update_gn,          s=10, color='tomato', alpha=0.4, zorder=3)
        ax.plot(update_gn_s,    smooth(update_gn),  color='tomato', linewidth=2, label='global gN')
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Global Gradient Norm  (update steps only)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 2: Inv-head grad norm  |  Memory norm ────────────────────────────
    ax = axes[2,0]
    if update_steps:
        ax.scatter(update_steps, update_hgn,           s=10, color='darkorange', alpha=0.4, zorder=3)
        ax.plot(update_steps,    smooth(update_hgn),   color='darkorange', linewidth=2, label='inv-head gN')
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Inventory Head Grad Norm\n(should stay > 0; collapse = head stopped learning)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2,1]
    ax.plot(steps, mem_norms, alpha=0.25, color='mediumpurple', linewidth=0.6)
    ax.plot(steps, smooth(mem_norms), color='mediumpurple', linewidth=2, label='memory L2 norm')
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 norm')
    ax.set_title('Memory Vector Norm\n(explosion/collapse = GRU instability)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 3: Data quality (N_ne, N_gui)  |  Step time ─────────────────────
    ax = axes[3,0]
    ax.plot(steps, n_ne_vals,  alpha=0.25, color='cornflowerblue', linewidth=0.6)
    ax.plot(steps, smooth(n_ne_vals),  color='cornflowerblue', linewidth=2, label='N_ne (non-empty slots)')
    ax.plot(steps, n_gui_vals, alpha=0.25, color='gold', linewidth=0.6)
    ax.plot(steps, smooth(n_gui_vals), color='gold',         linewidth=2, label='N_gui (GUI-open frames in window)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Frames / Slots')
    ax.set_title('Data Quality: Inventory Richness + GUI Frames per Window\n'
                 'N_gui=0 every step → anchor sampling failed, check is_gui_inventory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plot_series(axes[3,1], steps, step_times, 'step time', 'gray',
                'Step Time per Outer Step', 'Seconds')

    # ── Row 4: Memory retention scatter  |  GUI−Closed gap over time ────────
    #
    # Scatter (left): each point is (cos_sim_gui_t, cos_sim_closed_t) coloured
    # by training step (blue=early, red=late).  As co-adaptation improves the
    # GRU memory, points should drift toward the y=x diagonal — meaning the
    # model predicts inventory equally well whether or not the screen is open.
    #
    # Gap (right): (cos_sim_gui − cos_sim_closed) over steps.  Should trend
    # toward 0.  Flat or growing gap means memory isn't improving.
    #
    # Label: "Memory retention: gap between GUI-open and GUI-closed cos_sim
    #          should close over training."
    ax = axes[4, 0]
    valid_pts = [
        (steps[i], cos_gui[i], cos_closed[i])
        for i in range(len(steps))
        if (cos_gui[i] not in (None, 0.0) and cos_closed[i] not in (None, 0.0)
            and not math.isnan(cos_gui[i]) and not math.isnan(cos_closed[i]))
    ]
    if valid_pts:
        sc_steps, sc_g, sc_c = zip(*valid_pts)
        sc = ax.scatter(sc_g, sc_c, c=sc_steps, cmap='coolwarm', s=6, alpha=0.55, zorder=3)
        plt.colorbar(sc, ax=ax, label='Step')
        _lo = min(min(sc_g), min(sc_c))
        _hi = max(max(sc_g), max(sc_c))
        ax.plot([_lo, _hi], [_lo, _hi], 'k--', alpha=0.35, linewidth=1.2, label='y=x  (no gap)')
    ax.set_xlabel('cos_sim GUI-open (visual read)')
    ax.set_ylabel('cos_sim GUI-closed (memory retention)')
    ax.set_title('Memory retention: gap between GUI-open and GUI-closed cos_sim\n'
                 'should close over training  (blue=early steps, red=late steps)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4, 1]
    gap = [
        (cos_gui[i] - cos_closed[i])
        if (cos_gui[i] not in (None, 0.0) and cos_closed[i] not in (None, 0.0)
            and not math.isnan(float(cos_gui[i])) and not math.isnan(float(cos_closed[i])))
        else float('nan')
        for i in range(len(steps))
    ]
    ax.plot(steps, gap, alpha=0.2, color='coral', linewidth=0.6)
    ax.plot(steps, smooth(gap), color='coral', linewidth=2, label='gui − closed gap')
    ax.axhline(y=0, linestyle=':', color='black', alpha=0.4)
    ax.set_xlabel('Step')
    ax.set_ylabel('cos_sim_gui − cos_sim_closed')
    ax.set_title('GUI/Closed gap over training\n'
                 'Should trend → 0  as GRU learns to retain inventory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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

    valid  = [m for m in metrics if m.get('type_loss')]
    t_vals = [m['type_loss'] for m in valid]
    c_vals = [m['count_loss'] for m in valid if m.get('count_loss')]
    print(f"\nSummary ({len(metrics)} steps):")
    if t_vals:
        cos_i = 2.0 * math.exp(-t_vals[0])  - 1.0
        cos_f = 2.0 * math.exp(-t_vals[-1]) - 1.0
        print(f"  cos_sim:  {cos_i:+.4f} → {cos_f:+.4f}  (type_loss {t_vals[0]:.4f} → {t_vals[-1]:.4f})")
    if c_vals:
        print(f"  cnt_loss: {c_vals[0]:.4f} → {c_vals[-1]:.4f}  (min {min(c_vals):.4f})")


if __name__ == '__main__':
    main()
