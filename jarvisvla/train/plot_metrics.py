#!/usr/bin/env python3
"""
Plot training metrics from JSON Lines log file.

Usage:
    python jarvisvla/train/plot_metrics.py /data/vvm33/checkpoints/train_inv_head/metrics.jsonl
    python jarvisvla/train/plot_metrics.py metrics.jsonl -o training_plot.png

Supports both old (cosine-similarity) and new (cross-entropy / accuracy) metric logs.
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
    lm_losses  = [m.get('lm_loss')          for m in metrics]
    rsn_losses = [m.get('reasoning_loss')   for m in metrics]
    n_ne_vals  = [m.get('n_ne_avg', 0)       for m in metrics]
    n_gui_vals = [m.get('n_gui_frames', 0)   for m in metrics]
    mem_norms  = [m.get('memory_norm', 0)    for m in metrics]
    lr_head    = [m.get('lr_head')           for m in metrics]
    lr_backbone= [m.get('lr_backbone')       for m in metrics]

    # cos_sim_gui / cos_sim_closed stores item accuracy (renamed field kept for compat).
    acc_gui    = [m.get('cos_sim_gui')    for m in metrics]
    acc_closed = [m.get('cos_sim_closed') for m in metrics]
    acc_all    = [m.get('item_acc_all')   for m in metrics]

    is_new_style = any(m.get('item_acc_all') is not None for m in metrics)
    y_label = 'Item Top-1 Accuracy' if is_new_style else 'cos_sim / Accuracy'

    update_steps = [m['step'] for m in metrics if m.get('is_update_step')]
    update_gn    = [m['grad_norm']          for m in metrics if m.get('is_update_step') and m.get('grad_norm', 0) > 0]
    update_gn_s  = [m['step']              for m in metrics if m.get('is_update_step') and m.get('grad_norm', 0) > 0]
    update_hgn   = [m['inv_head_grad_norm'] for m in metrics if m.get('is_update_step')]

    # Eval-only metrics (from log_summary lines — not per-step)
    # These are emitted rarely (at eval checkpoints), so we just mark them as points.
    eval_eff_rank_pts  = []
    eval_stable_rank_pts = []
    for m in metrics:
        _e = m.get('eval_eff_rank')
        _s = m.get('eval_stable_rank')
        if _e is not None:
            eval_eff_rank_pts.append((m['step'], _e))
        if _s is not None:
            eval_stable_rank_pts.append((m['step'], _s))

    fig, axes = plt.subplots(6, 2, figsize=(16, 30))
    fig.suptitle('JarvisVLA — VLA Training Metrics', fontsize=14, fontweight='bold')

    # ── Row 0: Item CE loss  |  GUI vs Closed accuracy (THE key diagnostic) ──
    plot_series(axes[0,0], steps, type_losses, 'item CE loss', 'darkorchid',
                'Item Type Loss  (cross-entropy, all 36 slots)', 'CE Loss')

    ax = axes[0,1]
    acc_all_v    = [v if v is not None else float('nan') for v in acc_all]
    acc_gui_v    = [v if v is not None else float('nan') for v in acc_gui]
    acc_closed_v = [v if v is not None else float('nan') for v in acc_closed]
    ax.plot(steps, acc_all_v,    alpha=0.15, color='gray',        linewidth=0.5)
    ax.plot(steps, smooth(acc_all_v),    color='gray',        linewidth=1.5, label='acc (all)',    linestyle='--')
    ax.plot(steps, acc_gui_v,    alpha=0.15, color='steelblue',   linewidth=0.5)
    ax.plot(steps, smooth(acc_gui_v),    color='steelblue',   linewidth=2,   label='acc GUI open')
    ax.plot(steps, acc_closed_v, alpha=0.15, color='tomato',      linewidth=0.5)
    ax.plot(steps, smooth(acc_closed_v), color='tomato',      linewidth=2,   label='acc GUI closed (memory)')
    ax.axhline(y=0, linestyle=':', color='black', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel(y_label)
    ax.set_title('GUI-open vs Closed Item Accuracy\n'
                 'KEY: closed (red) should rise as memory learns to retain inventory')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 1: Count CE loss  |  Global grad norm ────────────────────────────
    plot_series(axes[1,0], steps, cnt_losses, 'count CE loss', 'seagreen',
                'Count Loss  (cross-entropy, non-empty slots only)', 'CE Loss')

    ax = axes[1,1]
    if update_gn_s:
        ax.scatter(update_gn_s, update_gn,          s=10, color='tomato', alpha=0.4, zorder=3)
        ax.plot(update_gn_s,    smooth(update_gn),  color='tomato', linewidth=2, label='global gN')
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Global Gradient Norm  (update steps only)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 2: LM loss + Reasoning loss  |  Inv-head grad norm ──────────────
    ax = axes[2,0]
    _lm_v  = [v if v is not None and v > 0 else float('nan') for v in lm_losses]
    _rsn_v = [v if v is not None and v > 0 else float('nan') for v in rsn_losses]
    ax.plot(steps, _lm_v,  alpha=0.15, color='royalblue',  linewidth=0.5)
    ax.plot(steps, smooth(_lm_v),  color='royalblue',  linewidth=2, label='LM (inventory desc)')
    ax.plot(steps, _rsn_v, alpha=0.15, color='darkorange', linewidth=0.5)
    ax.plot(steps, smooth(_rsn_v), color='darkorange', linewidth=2, label='LM (reasoning, OpenHermes)')
    ax.set_xlabel('Step')
    ax.set_ylabel('CE Loss')
    ax.set_title('Language Modeling Losses\n'
                 'inventory desc (even steps) vs reasoning (odd steps)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2,1]
    if update_steps:
        ax.scatter(update_steps, update_hgn,           s=10, color='indigo', alpha=0.4, zorder=3)
        ax.plot(update_steps,    smooth(update_hgn),   color='indigo', linewidth=2, label='inv-head gN')
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.set_title('Inventory Head Grad Norm\n(should stay > 0; collapse = head stopped learning)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 3: Memory norm  |  LR schedule ──────────────────────────────────
    ax = axes[3,0]
    ax.plot(steps, mem_norms, alpha=0.25, color='mediumpurple', linewidth=0.6)
    ax.plot(steps, smooth(mem_norms), color='mediumpurple', linewidth=2, label='memory L2 norm')
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 norm')
    ax.set_title('Memory Vector Norm\n(explosion/collapse = GRU instability)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3,1]
    _lr_h_v = [v for v in lr_head     if v is not None]
    _lr_s_h = [steps[i] for i, v in enumerate(lr_head)     if v is not None]
    _lr_b_v = [v for v in lr_backbone if v is not None]
    _lr_s_b = [steps[i] for i, v in enumerate(lr_backbone) if v is not None]
    if _lr_h_v:
        ax.plot(_lr_s_h, _lr_h_v, color='steelblue', linewidth=1.5, label='LR head')
    if _lr_b_v:
        ax.plot(_lr_s_b, _lr_b_v, color='tomato',    linewidth=1.5, label='LR backbone')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule  (cosine decay)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 4: Data quality  |  Effective rank ───────────────────────────────
    ax = axes[4,0]
    ax.plot(steps, n_ne_vals,  alpha=0.25, color='cornflowerblue', linewidth=0.6)
    ax.plot(steps, smooth(n_ne_vals),  color='cornflowerblue', linewidth=2, label='N_ne (non-empty slots)')
    ax.plot(steps, n_gui_vals, alpha=0.25, color='gold', linewidth=0.6)
    ax.plot(steps, smooth(n_gui_vals), color='gold',         linewidth=2, label='N_gui (GUI-open frames)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Frames / Slots')
    ax.set_title('Data Quality: Inventory Richness + GUI Frames per Window')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4,1]
    if eval_eff_rank_pts:
        _er_s, _er_v = zip(*eval_eff_rank_pts)
        ax.scatter(_er_s, _er_v, color='firebrick', s=50, zorder=4, label='eff rank (eval)')
        ax.plot(_er_s, _er_v, color='firebrick', linewidth=1.5)
    if eval_stable_rank_pts:
        _sr_s, _sr_v = zip(*eval_stable_rank_pts)
        ax.scatter(_sr_s, _sr_v, color='darkorange', s=30, zorder=3, label='stable rank (eval)', marker='s')
        ax.plot(_sr_s, _sr_v, color='darkorange', linewidth=1, linestyle='--')
    ax.axhline(y=50,  linestyle=':', color='red',  alpha=0.5, linewidth=1.2)
    ax.axhline(y=500, linestyle=':', color='green', alpha=0.5, linewidth=1.2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Rank')
    ax.set_title('Effective Rank of Last-Layer Hidden States  (eval only)\n'
                 'red dashed=collapse threshold(50), green=healthy(500)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Row 5: Memory retention scatter  |  GUI−Closed gap ──────────────────
    ax = axes[5,0]
    valid_pts = [
        (steps[i], acc_gui[i], acc_closed[i])
        for i in range(len(steps))
        if (acc_gui[i] not in (None, 0.0) and acc_closed[i] not in (None, 0.0)
            and not math.isnan(float(acc_gui[i])) and not math.isnan(float(acc_closed[i])))
    ]
    if valid_pts:
        sc_steps, sc_g, sc_c = zip(*valid_pts)
        sc = ax.scatter(sc_g, sc_c, c=sc_steps, cmap='coolwarm', s=6, alpha=0.55, zorder=3)
        plt.colorbar(sc, ax=ax, label='Step')
        _lo = min(min(sc_g), min(sc_c))
        _hi = max(max(sc_g), max(sc_c))
        ax.plot([_lo, _hi], [_lo, _hi], 'k--', alpha=0.35, linewidth=1.2, label='y=x  (no gap)')
    ax.set_xlabel('Accuracy GUI-open (visual read)')
    ax.set_ylabel('Accuracy GUI-closed (memory retention)')
    ax.set_title('Memory retention: GUI-open vs GUI-closed\n'
                 'should close over training  (blue=early, red=late)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[5,1]
    gap = [
        (acc_gui[i] - acc_closed[i])
        if (acc_gui[i] not in (None, 0.0) and acc_closed[i] not in (None, 0.0)
            and not math.isnan(float(acc_gui[i])) and not math.isnan(float(acc_closed[i])))
        else float('nan')
        for i in range(len(steps))
    ]
    ax.plot(steps, gap, alpha=0.2, color='coral', linewidth=0.6)
    ax.plot(steps, smooth(gap), color='coral', linewidth=2, label='gui − closed gap')
    ax.axhline(y=0, linestyle=':', color='black', alpha=0.4)
    ax.set_xlabel('Step')
    ax.set_ylabel('acc_gui − acc_closed')
    ax.set_title('GUI/Closed accuracy gap\nShould trend → 0 as GRU learns to retain inventory')
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
    a_vals = [m['item_acc_all'] for m in valid if m.get('item_acc_all')]
    print(f"\nSummary ({len(metrics)} steps):")
    if t_vals:
        print(f"  item CE loss: {t_vals[0]:.4f} → {t_vals[-1]:.4f}")
    if c_vals:
        print(f"  count CE loss: {c_vals[0]:.4f} → {c_vals[-1]:.4f}  (min {min(c_vals):.4f})")
    if a_vals:
        print(f"  item top-1 acc: {a_vals[0]:.4f} → {a_vals[-1]:.4f}  (max {max(a_vals):.4f})")


if __name__ == '__main__':
    main()
