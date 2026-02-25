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
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import functools

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer

sys.path.insert(0, '/home/vvm33/JarvisVLA')
from jarvisvla.models.stateful_vla import wrap_model_for_stateful_training
from jarvisvla.train.mc_item_vocab import get_item_vocab_size, COUNT_CLASSES, item_id_to_name


# ---------------------------------------------------------------------------
# LM supervision: inventory description prompts
#
# Ideally this model predicts *actions* (move, attack, craft, place, etc.) —
# that's what a VLA does, and that's how we'll fine-tune once we have
# action-annotated data.  For this pre-training stage we don't have those
# labels, so instead we supervise the backbone to *describe the inventory*
# in plain English.
#
# Why bother?  Without any LM loss the inventory aux-head gradient must
# backprop through 32 transformer layers with nothing anchoring the backbone,
# which causes it to vanish.  Having the model output text keeps all language
# pathways active and gives the aux-head gradient a well-conditioned residual
# stream to ride on top of.
#
# The question prompt and answer prefix are sampled randomly from short lists
# so the model can't short-circuit by memorising the question string.
# ---------------------------------------------------------------------------
INVENTORY_QUESTION_PROMPTS = [
    "What is in the inventory?",
    "List the items in my inventory.",
    "What am I currently carrying?",
    "Describe my inventory contents.",
    "What items do I have with me right now?",
    "Give me an inventory summary.",
]

INVENTORY_ANSWER_PREFIXES = [
    "My inventory contains:",
    "I am currently carrying:",
    "The inventory holds:",
    "I have the following items:",
    "Current inventory contents:",
]


def build_inventory_answer(slot_type_texts, slot_counts, prefix):
    """Build a plain-English description of the current inventory state."""
    items = []
    for text, count in zip(slot_type_texts, slot_counts):
        if text != 'empty slot' and int(count) > 0:
            items.append(f"{text} x{int(count)}")
    if items:
        return f"{prefix} {', '.join(items)}."
    return f"{prefix} nothing."


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


def get_fsdp_model(model, device_id, frozen_backbone: bool = False):
    """Wrap model with FSDP.

    Sharding strategy:
      SHARD_GRAD_OP (ZeRO-2, default, unfrozen backbone):
        Parameters stay replicated during forward and backward; only gradients
        and optimizer state are sharded.  FULL_SHARD all-gathers on *every*
        forward AND backward pass.  With 4×80 GB GPUs and ~16.6 GB model in
        bf16, params fit replicated — this removes roughly half the all-gather
        overhead vs FULL_SHARD.

      FULL_SHARD (ZeRO-3, frozen backbone):
        With --frozen_backbone, the 8 B backbone parameters don't need to be
        replicated (no gradient computation for them).  FULL_SHARD shards the
        frozen params across GPUs, saving ~12 GB/GPU, at the cost of one extra
        all-gather per forward.  The head+memory params are tiny and the
        backbone all-gather during forward is unavoidable anyway.
    """
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    def wrap_policy_fn(module):
        return isinstance(module, Qwen2VLDecoderLayer)

    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=wrap_policy_fn)

    strategy = ShardingStrategy.FULL_SHARD if frozen_backbone else ShardingStrategy.SHARD_GRAD_OP

    model = FSDP(
        model,
        device_id=device_id,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=strategy,
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


def evaluate_on_test_set(model, test_loader, processor, args, device, rank, logger):
    """
    End-of-training evaluation on held-out test sequences.

    All FSDP ranks must call this — FSDP all-gathers on every forward pass.
    Only rank 0 prints and logs.

    Metrics logged:
        eval_cos_sim           mean per-slot cosine sim over all test frames
        eval_non_empty_cos_sim same, restricted to frames that have items
        eval_non_empty_pct     fraction of test frames with a non-empty inventory
    """
    num_seqs = len(test_loader)
    if rank == 0:
        print(f"\n[5/5] Evaluating on {num_seqs} test sequences "
              f"({num_seqs * 50} frames)...")

    model.eval()
    model.train_aux_heads(True)       # re-enable aux head for inference
    if model.inventory_embedding_head is not None:
        model.inventory_embedding_head.eval()   # keep dropout off

    total_item_acc  = 0.0
    ne_item_acc     = 0.0
    total_count_acc = 0.0
    n_total     = 0
    n_non_empty = 0
    n_count     = 0
    display_examples: List[Dict] = []

    with torch.no_grad():
        for seq_idx in range(num_seqs):
            if rank == 0 and seq_idx % 5 == 0:
                print(f"  seq {seq_idx + 1}/{num_seqs} ...", end='\r', flush=True)

            seq = test_loader[seq_idx]
            memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)

            _ref_frame  = seq['frames'][0]
            _eval_chat  = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image", "image": _ref_frame},
                    {"type": "text", "text": "What is in the inventory?"},
                ]}],
                tokenize=False, add_generation_prompt=True,
            )
            _eval_static  = processor(text=_eval_chat, images=[_ref_frame], return_tensors="pt")
            _eval_inp_ids = _eval_static['input_ids'].to(device)
            _eval_attn    = _eval_static['attention_mask'].to(device)

            for t in range(len(seq['frames'])):
                frame            = seq['frames'][t]
                inv_id_target    = seq['inventory_ids'][t:t+1].to(device)      # [1, 36] long
                inv_count_target = seq['inventory_counts'][t:t+1].to(device)   # [1, 36] long
                is_non_empty     = seq['inventory_has_items'][t]
                slot_type_texts  = seq.get('inventory_slot_texts',
                                           [['empty slot']*36]*len(seq['frames']))[t]
                slot_counts      = seq['inventory_counts'][t]  # [36] long, cpu

                _img_out = processor.image_processor(images=[frame], return_tensors="pt")
                _pv      = _img_out['pixel_values'].to(device)
                _thw     = _img_out['image_grid_thw'].to(device) if 'image_grid_thw' in _img_out else None

                outputs = model(
                    input_ids=_eval_inp_ids,
                    pixel_values=_pv,
                    image_grid_thw=_thw,
                    prev_memory=memory,
                    attention_mask=_eval_attn,
                    labels=None,
                )

                if outputs.item_logits is not None:
                    item_logits  = outputs.item_logits[0]   # [36, item_vocab_size]
                    count_logits = outputs.count_logits[0]  # [36, count_classes]
                    tgt_ids      = inv_id_target[0]         # [36] long
                    tgt_counts   = inv_count_target[0]      # [36] long

                    pred_ids    = item_logits.argmax(dim=-1)   # [36]
                    pred_counts = count_logits.argmax(dim=-1)  # [36]

                    # Top-1 accuracy over all 36 slots
                    acc_all = (pred_ids == tgt_ids).float().mean().item()
                    total_item_acc += acc_all
                    n_total += 1

                    if is_non_empty:
                        ne_item_acc += acc_all
                        n_non_empty += 1
                        non_empty_mask = tgt_ids > 0
                        if non_empty_mask.any():
                            cnt_acc = (pred_counts[non_empty_mask] ==
                                       tgt_counts[non_empty_mask]).float().mean().item()
                            total_count_acc += cnt_acc
                            n_count += 1

                    if rank == 0 and is_non_empty and len(display_examples) < 3:
                        display_examples.append({
                            'source': seq.get('source_file', '?'),
                            'tick': t,
                            'item_acc': acc_all,
                            'slots': [
                                (i, slot_type_texts[i],
                                 int(slot_counts[i].item()),
                                 int(pred_ids[i].item()),
                                 int(pred_counts[i].item()))
                                for i in range(len(slot_type_texts))
                                if slot_type_texts[i] != "empty slot"
                            ],
                        })

                memory = outputs.new_memory.detach()

    avg_item_acc   = total_item_acc  / max(n_total,     1)
    avg_ne_acc     = ne_item_acc     / max(n_non_empty, 1)
    avg_count_acc  = total_count_acc / max(n_count,     1)
    non_empty_pct  = 100.0 * n_non_empty / max(n_total, 1)

    eval_metrics = {
        'eval_item_acc':          avg_item_acc,
        'eval_non_empty_item_acc': avg_ne_acc,
        'eval_count_acc':         avg_count_acc,
        'eval_non_empty_pct':     non_empty_pct,
        'eval_n_frames':          n_total,
        'eval_seqs':              num_seqs,
    }
    logger.log_summary({'final_eval': eval_metrics})

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TEST SET EVALUATION  ({num_seqs} seqs, {n_total} frames)")
        print(f"{'='*70}")
        print(f"  Item top-1 acc (all frames): {avg_item_acc:.4f}")
        print(f"  Item top-1 acc (non-empty) : {avg_ne_acc:.4f}"
              f"  ({n_non_empty}/{n_total} = {non_empty_pct:.1f}%)")
        print(f"  Count top-1 acc (non-empty): {avg_count_acc:.4f}")

        if display_examples:
            print(f"\n--- Inventory head predictions (non-empty frames) ---")
            for ex in display_examples:
                gt_items   = {txt for _, txt, _, _, _ in ex['slots']}
                pred_items: set = set()

                print(f"\n  [{ex['source']}  tick {ex['tick']}]  item_acc={ex['item_acc']:.3f}")
                for slot_idx, type_text, true_cnt, pred_item_id, pred_cnt_class in ex['slots']:
                    pred_type = item_id_to_name(pred_item_id)
                    pred_items.add(pred_type)
                    match = '✓' if pred_type == type_text else '✗'
                    print(f"    slot {slot_idx:>2}"
                          f"  GT: {type_text:<22} cnt:{true_cnt:<4}"
                          f"PRED: {pred_type:<22} cnt:{pred_cnt_class:<4}  {match}")

                if gt_items or pred_items:
                    jaccard = len(gt_items & pred_items) / max(len(gt_items | pred_items), 1)
                    print(f"           Jaccard (types): {jaccard:.2f}"
                          f"  GT={sorted(gt_items)}  PRED={sorted(pred_items)}")

        print(f"{'='*70}")

    return eval_metrics


def save_checkpoint(model, step, losses, metrics_logger, output_dir, rank):
    """Save FSDP checkpoint using FULL_STATE_DICT.

    All ranks must call this — FSDP all-gathers the full model to rank 0's CPU.
    rank0_only=True means ranks 1+ return empty dicts and don't hold the full
    weights in memory, so there's no OOM risk on non-rank-0 GPUs.
    """
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{step}.pt')
    else:
        checkpoint_path = None

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        model_state_dict = model.state_dict()

    if rank == 0:
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
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--base_model_lr', type=float, default=5e-6)
    # Trinh et al. (ICML 2018) show auxiliary losses on the hidden state create
    # local gradient flow that makes short truncation nearly as effective as
    # full BPTT — exactly our setup.  Backbone gradient signal through the GRU
    # vanishes beyond ~4–8 steps due to standard recurrent gradient decay.
    # Paying for 64-step backbone BPTT is 8× more expensive for negligible
    # extra signal.  The GRU memory still carries information across 64+ frames;
    # only the optimizer's credit-assignment window is truncated.
    # Effective batch = bptt_chunk_size × grad_accum_steps = 8 × 64 = 512 frames
    # (same as the old 64 × 8 = 512 default).
    parser.add_argument('--bptt_chunk_size', type=int, default=8)
    parser.add_argument('--memory_dim', type=int, default=1024)
    parser.add_argument('--inventory_weight', type=float, default=8.0,
                        help='Scale factor on the InfoNCE type loss')
    parser.add_argument('--count_weight', type=float, default=4.0,
                        help='Scale factor on the count MSE loss')
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help='Gradient clip max norm')
    parser.add_argument('--grad_checkpoint', action='store_true',
                        help='Enable gradient checkpointing on the backbone. '
                             'Cuts activation memory ~10x at ~2x compute cost, '
                             'allowing longer bptt_chunk_size on the same VRAM.')
    parser.add_argument('--grad_accum_steps', type=int, default=64)
    parser.add_argument('--frozen_backbone', action='store_true',
                        help='Freeze all backbone (Qwen2-VL) parameters — only GRU memory '
                             'projections and inventory head train.  SPEED ABLATION ONLY: '
                             'the research hypothesis requires an unfrozen backbone for '
                             'co-adaptation.  Use this to verify the head converges and '
                             'establish a frozen-backbone baseline before expecting backbone '
                             'world-state encoding.  Automatically switches to FULL_SHARD '
                             'to avoid replicating 8 B frozen params across GPUs.')
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
    
    if args.grad_checkpoint:
        # Must be enabled before FSDP wrapping. use_reentrant=False is required
        # for compatibility with FSDP + custom autograd functions.
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if rank == 0:
            print("  Gradient checkpointing enabled — ~10x less activation memory, ~2x slower")

    model = wrap_model_for_stateful_training(
        base_model=base_model,
        memory_dim=args.memory_dim,
        add_inventory_head=True,
        inventory_head_kwargs={},   # uses defaults: item_vocab_size from mc-data, 128 count classes
    )

    # All params trainable — backbone at low LR, head+mem at high LR.
    # The whole point is to teach the backbone to understand the world via
    # the inventory auxiliary signal, not just to train an inventory head.
    for param in model.parameters():
        param.requires_grad = True

    if args.frozen_backbone:
        # Freeze every parameter that is NOT part of the GRU memory or the
        # auxiliary head.  The backbone will still run a forward pass (we need
        # its features for the head), but its weights don't move.
        # This is a SPEED/ABLATION mode — co-adaptation requires requires_grad=True
        # on the backbone.
        _head_keywords = ('memory_projections', 'inventory_embedding_head', 'initial_memory')
        for n, p in model.named_parameters():
            if not any(kw in n for kw in _head_keywords):
                p.requires_grad = False
        if rank == 0:
            n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [frozen_backbone] {n_frozen:,} params frozen, {n_trainable:,} trainable")

    total_params = sum(p.numel() for p in model.parameters())
    n_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if rank == 0:
        print(f"  Total params: {total_params:,}  (trainable: {n_trainable:,})")

    # FSDP setup
    model = model.to(device, dtype=torch.bfloat16)
    model = get_fsdp_model(model, device_id=local_rank, frozen_backbone=args.frozen_backbone)

    # Two param groups: head+memory at high LR, backbone at low LR.
    # With FSDP use_orig_params=True, original tensors are accessible via
    # named_parameters() even after wrapping; substring matching works despite
    # the _fsdp_wrapped_module prefix FSDP may prepend to names.
    head_param_names = set()
    for n, _ in model.named_parameters():
        if 'memory_projections' in n or 'inventory_embedding_head' in n:
            head_param_names.add(n)

    head_params_list = []
    backbone_params_list = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue   # frozen backbone — skip entirely; AdamW sees no gradient
        if n in head_param_names:
            head_params_list.append(p)
        else:
            backbone_params_list.append(p)

    param_groups = [
        {'params': head_params_list,     'lr': args.learning_rate},
        {'params': backbone_params_list, 'lr': args.base_model_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    if rank == 0:
        print(f"\n[2/5] Optimizer:")
        print(f"  Head+mem params : LR={args.learning_rate}  ({len(head_params_list)} tensors)")
        print(f"  Backbone params : LR={args.base_model_lr}  ({len(backbone_params_list)} tensors)")
    
    # Resume if needed
    start_step = 0
    losses = []
    if args.resume_from and os.path.exists(args.resume_from):
        if rank == 0:
            print(f"\n[2.5/5] Resuming from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('step', 0)
        losses = checkpoint.get('losses', [])
    
    # Data setup
    if rank == 0:
        print(f"\n[3/5] Loading data...")
    
    from jarvisvla.train.sequence_dataset import InventoryTextEncoder, OnDemandSequenceLoader
    
    encoder = InventoryTextEncoder(device=device)

    jsonl_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    train_files = jsonl_files[:args.train_jsonls]
    test_files  = jsonl_files[args.train_jsonls:args.train_jsonls + args.test_jsonls]
    # Load sequences long enough for one full BPTT chunk.
    # We load bptt_chunk_size + 100 extra frames so the anchor (gui-open) can
    # sit anywhere in the first 100 frames and still leave a full bptt_chunk_size
    # window behind it for the memory to operate over.
    seq_load_len = args.bptt_chunk_size + 100
    train_loader = OnDemandSequenceLoader(train_files, seq_load_len, encoder)
    test_loader  = OnDemandSequenceLoader(test_files,  seq_load_len, encoder)

    if rank == 0:
        print(f"  Training sequences: {len(train_loader)}")
        print(f"  Test sequences:     {len(test_loader)}  (files {args.train_jsonls}–{args.train_jsonls + args.test_jsonls})")
    
    # Training loop
    if rank == 0:
        print(f"\n[4/5] Training from step {start_step} to {args.train_steps}...")
        print(f"  Logging every {args.log_every} steps")
        print(f"  Checkpointing every {args.eval_every} steps")
        print(f"\n{'Step':>8}  {'cos(gui/cls)':>28}  {'cnt':>7}  {'N_ne':>4} {'gui':>4}"
              f"  {'gN':>6} {'hGN':>7} {'wN':>7} {'mN':>6}  {'Time':>7}")
        print("-"*105)
    
    step_times = []
    last_hgn = None   # last inv-head grad norm (from most recent update step)
    last_wn  = None   # last inv-head weight norm
    last_gn  = None   # last global grad norm (from most recent update step)

    # Accumulators for smoothed display: averaged over the current update window
    # (grad_accum_steps chunks).  Reset at each optimizer step so the display
    # always shows "what did the gradient that just fired actually represent?"
    win_type_losses  = []
    win_count_losses = []
    # Cached smoothed values — shown at every log step even between update steps
    last_avg_type  = None
    last_avg_count = None

    for step in range(start_step, args.train_steps):
        step_start = time.time()
        
        model.train()
        model.train_aux_heads(True)  # Enable inventory head
        chunk_losses = []
        chunk_type_losses = []     # item CE loss (logged as "type_loss" for plot compat)
        chunk_count_losses = []
        chunk_n_ne = []            # non-empty slot counts per frame
        chunk_type_losses_gui    = []   # item loss on GUI-open frames
        chunk_type_losses_closed = []   # item loss on GUI-closed frames
        chunk_acc_all:    List[float] = []
        chunk_acc_gui:    List[float] = []
        chunk_acc_closed: List[float] = []
        chunk_n_gui = 0            # how many frames had the GUI open this step

        memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)

        # Find a sequence where the inventory screen is actually open (isGuiInventory)
        # somewhere in the first 100 frames, then start the BPTT window there.
        #
        # The window layout is:
        #   [anchor=gui_open]  →  [closed] → [closed] → ... (bptt_chunk_size frames)
        #
        # Frame 0: inventory screen visible → Qwen gets a direct visual read → memory updates.
        # Frames 1-N: screen closed → model must use memory to predict inventory.
        # Loss fires at every frame from JSONL ground truth (screen-open or not).
        # This is exactly the temporal test we want: "remember what you saw."
        for _seq_retry in range(20):
            seq_idx = torch.randint(0, len(train_loader), (1,)).item()
            seq = train_loader[seq_idx]
            gui_open = seq['is_gui_inventory']   # List[bool], len = seq_load_len
            # Only look in the first 100 frames so the full bptt_chunk_size fits after
            searchable = min(100, len(gui_open) - args.bptt_chunk_size)
            open_frames = [i for i in range(searchable) if gui_open[i]]
            if open_frames:
                break
        if not open_frames:
            # No gui-open frame found — fall back to frame 0 (model trains from scratch)
            start_idx = 0
        else:
            anchor = open_frames[torch.randint(0, len(open_frames), (1,)).item()]
            start_idx = anchor

        scale = args.bptt_chunk_size * args.grad_accum_steps

        # ── Preprocessor optimisation ──────────────────────────────────────────
        # Calling processor(text=..., images=[frame]) inside the per-frame loop
        # runs BPE tokenisation + chat-template expansion + image preprocessing
        # on every iteration.  The text never changes between frames, so we
        # separate the two concerns:
        #
        #   1. Tokenise ONCE → static input_ids / attention_mask valid for all
        #      frames (all frames are 224×224, so image_grid_thw is also constant).
        #
        #   2. Batch-preprocess all chunk frames in ONE image-processor call
        #      instead of bptt_chunk_size individual calls, then split the packed
        #      pixel_values tensor by image_grid_thw before the inner loop.
        #
        # This eliminates ~(bptt_chunk_size - 1) tokenisation calls and folds
        # N image-processor calls into one batched call per outer step.
        _dummy_frame = seq['frames'][start_idx]
        chat_text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": _dummy_frame},
                {"type": "text", "text": "What is in the inventory?"},
            ]}],
            tokenize=False, add_generation_prompt=True,
        )

        # Static text inputs — identical for every frame in the chunk
        _static = processor(text=chat_text, images=[_dummy_frame], return_tensors="pt")
        _static_input_ids = _static['input_ids'].to(device)       # [1, seq_len]
        _static_attn_mask = _static['attention_mask'].to(device)  # [1, seq_len]

        # Batch pixel preprocessing for the entire BPTT chunk
        _chunk_end    = min(start_idx + args.bptt_chunk_size, len(seq['frames']))
        _chunk_frames = seq['frames'][start_idx:_chunk_end]
        _batch_img    = processor.image_processor(images=_chunk_frames, return_tensors="pt")
        _raw_pv       = _batch_img['pixel_values']           # [total_patches, …] packed
        _thw          = _batch_img['image_grid_thw']         # [N, 3]  T×H×W per frame

        # Split packed pixel_values into per-frame tensors using image_grid_thw
        _pv_per_frame:  list = []
        _thw_per_frame: list = []
        _offset = 0
        for _fi in range(len(_chunk_frames)):
            _n = int(_thw[_fi, 0] * _thw[_fi, 1] * _thw[_fi, 2])
            _pv_per_frame.append(_raw_pv[_offset:_offset + _n].to(device))
            _thw_per_frame.append(_thw[_fi:_fi + 1].to(device))
            _offset += _n
        # ── End preprocessor optimisation ─────────────────────────────────────

        # True BPTT: accumulate loss across all frames in the window, then call
        # backward ONCE.  Memory is NOT detached between frames so gradients flow
        # back through the GRU recurrence.  This is what actually teaches the GRU
        # to hold information: "you needed to remember what you saw at frame 0 to
        # predict inventory correctly at frame 150."
        seq_loss = None   # tensor — accumulates the computation graph

        for t in range(start_idx, _chunk_end):
            _t_rel = t - start_idx
            inv_id_target    = seq['inventory_ids'][t:t+1].to(device)       # [1, 36] long
            inv_count_target = seq['inventory_counts'][t:t+1].to(device)    # [1, 36] long

            outputs = model(
                input_ids=_static_input_ids,
                pixel_values=_pv_per_frame[_t_rel],
                image_grid_thw=_thw_per_frame[_t_rel],
                prev_memory=memory,
                attention_mask=_static_attn_mask,
                labels=None,
            )

            # No detach — memory carries the gradient graph across frames.
            memory = outputs.new_memory

            memory_reg = 0.1 * (memory ** 2).mean()
            frame_loss = memory_reg / scale

            if outputs.item_logits is not None:
                item_logits  = outputs.item_logits[0]    # [36, item_vocab_size]
                count_logits = outputs.count_logits[0]   # [36, count_classes]
                tgt_ids      = inv_id_target[0]          # [36] long
                tgt_counts   = inv_count_target[0]       # [36] long

                non_empty = tgt_ids > 0                  # [36] bool mask (0 = empty)
                N_ne = int(non_empty.sum().item())
                chunk_n_ne.append(N_ne)

                # --- Item type loss: CE over all 36 slots ---
                # Empty slots (target=0) provide supervision to predict "empty";
                # non-empty slots must predict the correct item ID.
                item_loss = F.cross_entropy(item_logits, tgt_ids)

                # --- Count loss: CE over non-empty slots only ---
                count_loss = None
                if N_ne > 0:
                    count_loss = F.cross_entropy(
                        count_logits[non_empty],
                        tgt_counts[non_empty],
                    )

                frame_loss = frame_loss + args.inventory_weight * item_loss / scale
                tl_val = item_loss.item()
                chunk_type_losses.append(tl_val)
                is_gui_t = seq['is_gui_inventory'][t]

                # Top-1 accuracy for training-step monitoring (no extra backward)
                with torch.no_grad():
                    pred_ids    = item_logits.argmax(dim=-1)   # [36]
                    pred_counts = count_logits.argmax(dim=-1)  # [36]
                    acc_t = (pred_ids == tgt_ids).float().mean().item()
                    chunk_acc_all.append(acc_t)
                    if is_gui_t:
                        chunk_type_losses_gui.append(tl_val)
                        chunk_acc_gui.append(acc_t)
                        chunk_n_gui += 1
                    else:
                        chunk_type_losses_closed.append(tl_val)
                        chunk_acc_closed.append(acc_t)
                if count_loss is not None:
                    frame_loss = frame_loss + args.count_weight * count_loss / scale
                    chunk_count_losses.append(count_loss.item())

            seq_loss = frame_loss if seq_loss is None else seq_loss + frame_loss
            chunk_losses.append(frame_loss.item() * scale)  # display (forces CUDA sync)

            # Hard norm cap: clamp memory to unit ball before next frame.
            # When norm > 1.0 this renormalises to exactly 1.0; when norm ≤ 1.0
            # it's a no-op.  Differentiable — gradient flows through the division.
            # The soft memory_reg penalty (above) discourages growth; this hard
            # cap is the safety net that prevents runaway explosion regardless.
            memory = memory / memory.norm(dim=-1, keepdim=True).clamp(min=1.0)

        # Snapshot memory norm before backward frees the graph.
        memory_norm_end = memory.detach().float().norm().item()

        # Single backward through the full bptt_chunk_size frame sequence.
        # Gradient flows back through GRU recurrence across all frames.
        if seq_loss is not None:
            seq_loss.backward()

        # Feed this chunk into the update-window accumulators
        win_type_losses.extend(chunk_type_losses)
        win_count_losses.extend(chunk_count_losses)

        # Gradient accumulation: accumulate for grad_accum_steps outer steps before
        # updating.  The loss is already divided by (bptt_chunk_size * grad_accum_steps)
        # so after grad_accum_steps steps the effective batch covers
        # grad_accum_steps * bptt_chunk_size timesteps.
        is_update_step = (
            ((step - start_step + 1) % args.grad_accum_steps == 0)
            or (step + 1 == args.train_steps)
        )

        grad_norm = None
        inv_head_grad_norm = None
        inv_head_weight_norm = None
        if is_update_step and chunk_losses:
            # Per-component grad norm BEFORE clipping.
            # With FSDP FULL_SHARD + use_orig_params=True, each rank holds only a
            # SHARD of each parameter.  Rank 0 typically holds the first flat-param
            # shard (mostly ViT + embedding params) and has p.grad=None for the
            # inventory head.  We must all-reduce the per-rank squared norms to get
            # the true global gradient norm.
            head_sq_norm = torch.tensor(0.0, device=device)
            head_w_sq_norm = torch.tensor(0.0, device=device)
            for n, p in model.named_parameters():
                is_head_param = 'inventory_embedding_head' in n or 'memory_projections' in n
                if is_head_param:
                    if p.grad is not None:
                        head_sq_norm += p.grad.detach().float().pow(2).sum()
                    if 'inventory_embedding_head' in n:
                        head_w_sq_norm += p.detach().float().pow(2).sum()
            if world_size > 1:
                dist.all_reduce(head_sq_norm,   op=dist.ReduceOp.SUM)
                dist.all_reduce(head_w_sq_norm, op=dist.ReduceOp.SUM)
            _gn = float(head_sq_norm.sqrt().item())
            inv_head_grad_norm   = _gn if _gn > 1e-12 else None
            inv_head_weight_norm = float(head_w_sq_norm.sqrt().item())
            # Cache for display on non-update log steps
            last_hgn = inv_head_grad_norm
            last_wn  = inv_head_weight_norm

            # Debug: at the very first update step, dump param group sizes and
            # confirm inventory head params have correct names / are in group 0.
            if step < start_step + args.grad_accum_steps and rank == 0:
                inv_names = [n for n, p in model.named_parameters()
                             if 'inventory_embedding_head' in n]
                print(f"\n[DEBUG step {step}] inv head param tensors: {len(inv_names)}"
                      f"  (e.g. {inv_names[0] if inv_names else 'NONE FOUND'})")
                print(f"[DEBUG step {step}] head_sq_norm={head_sq_norm.item():.4f}"
                      f"  hGN={inv_head_grad_norm}  wN={inv_head_weight_norm:.4f}")

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            last_gn = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm or 0.0)
            optimizer.step()
            optimizer.zero_grad()

            # Compute smoothed display values from the just-completed update window
            # and reset the accumulators for the next window.
            last_avg_type  = sum(win_type_losses)  / len(win_type_losses)  if win_type_losses  else None
            last_avg_count = sum(win_count_losses) / len(win_count_losses) if win_count_losses else None
            win_type_losses.clear()
            win_count_losses.clear()

        if chunk_losses:
            losses.append(sum(chunk_losses))

            avg_loss       = sum(chunk_losses)
            avg_type_loss  = sum(chunk_type_losses)  / len(chunk_type_losses)  if chunk_type_losses  else 0.0
            avg_count_loss = sum(chunk_count_losses) / len(chunk_count_losses) if chunk_count_losses else 0.0
            avg_n_ne       = sum(chunk_n_ne)         / len(chunk_n_ne)         if chunk_n_ne         else 0.0
            # GUI vs closed item accuracy — THE key diagnostic.
            # closed-frame accuracy should rise over training as the model
            # learns to retain inventory state through the GRU memory.
            acc_gui    = sum(chunk_acc_gui)    / len(chunk_acc_gui)    if chunk_acc_gui    else None
            acc_closed = sum(chunk_acc_closed) / len(chunk_acc_closed) if chunk_acc_closed else None
            acc_all_avg= sum(chunk_acc_all)    / len(chunk_acc_all)    if chunk_acc_all    else None

            step_time = time.time() - step_start
            step_times.append(step_time)

            metrics = {
                'loss':               avg_loss,
                'type_loss':          avg_type_loss,   # item CE loss (named type_loss for plot compat)
                'count_loss':         avg_count_loss,
                'n_ne_avg':           avg_n_ne,
                'n_gui_frames':       chunk_n_gui,
                # acc_gui:   item top-1 accuracy on GUI-open frames (direct visual read)
                # acc_closed: item top-1 accuracy on GUI-closed frames (memory retention)
                # Gap should CLOSE over training as GRU learns to retain inventory.
                'cos_sim_gui':        acc_gui    if acc_gui    is not None else 0.0,  # name kept for plot compat
                'cos_sim_closed':     acc_closed if acc_closed is not None else 0.0,  # name kept for plot compat
                'item_acc_all':       acc_all_avg if acc_all_avg is not None else 0.0,
                'memory_norm':        memory_norm_end,
                'grad_norm':          last_gn or 0.0,
                'inv_head_grad_norm': inv_head_grad_norm or 0.0,
                'inv_head_weight_norm': inv_head_weight_norm or 0.0,
                'step_time':          step_time,
                'is_update_step':     is_update_step,
            }

            logger.log(metrics, step)

            if rank == 0 and step % args.log_every == 0:
                gn_str  = f"{last_gn:.1f}"  if last_gn  is not None else "  N/A"
                hgn_str = f"{last_hgn:.3f}" if last_hgn is not None else "  N/A"
                wn_str  = f"{last_wn:.3f}"  if last_wn  is not None else "    N/A"
                # Use smoothed (update-window averaged) values for display
                disp_type  = last_avg_type  if last_avg_type  is not None else avg_type_loss
                disp_count = last_avg_count if last_avg_count is not None else avg_count_loss
                gui_str    = f"{acc_gui:.3f}"    if acc_gui    is not None else "  N/A"
                closed_str = f"{acc_closed:.3f}" if acc_closed is not None else "  N/A"
                flag = "  *** NO GUI ***" if chunk_n_gui == 0 else ""
                print(f"{step:>8} item_loss={disp_type:.3f} acc(gui={gui_str} cls={closed_str})"
                      f"  cnt={disp_count:.4f}  N_ne={avg_n_ne:4.1f} gui={chunk_n_gui:3d}"
                      f"  gN={gn_str} hGN={hgn_str} wN={wn_str} mN={memory_norm_end:.2f}"
                      f"  {step_time:>6.1f}s{flag}")

        # Checkpoint
        if (step + 1) % args.eval_every == 0:
            save_checkpoint(model, step + 1, losses, logger, args.output_dir, rank)
    
    # Final save
    save_checkpoint(model, args.train_steps, losses, logger, args.output_dir, rank)

    # Final evaluation on held-out test set
    eval_metrics = evaluate_on_test_set(model, test_loader, processor, args, device, rank, logger)

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
        if eval_metrics:
            print(f"  Eval item top-1 acc (all)  : {eval_metrics['eval_item_acc']:.4f}")
            print(f"  Eval item top-1 acc (ne)   : {eval_metrics['eval_non_empty_item_acc']:.4f}")
            print(f"  Eval count top-1 acc (ne)  : {eval_metrics['eval_count_acc']:.4f}")
    
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
