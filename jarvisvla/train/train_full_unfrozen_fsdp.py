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
import itertools

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


def build_slot_text_target(slot_type_texts, slot_counts) -> str:
    """Structured slot-by-slot inventory text for LM supervision.

    Format: [slot_N] item_name count:C   (one per line)
    Extensible: add damage:D enchants:[...] in future iterations without
    any architecture changes — just extend this string.

    Used as the assistant answer in teacher-forced LM fine-tuning on
    GUI-open frames.  Forces the language head to stay active and prevents
    backbone representation rank collapse (effective rank staying high
    rather than collapsing to a few dominant directions).
    """
    lines = []
    for i, (text, count) in enumerate(zip(slot_type_texts, slot_counts)):
        if text == 'empty slot' or int(count) == 0:
            lines.append(f"[slot_{i}] empty")
        else:
            lines.append(f"[slot_{i}] {text} count:{int(count)}")
    return "\n".join(lines)


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
    # Accumulate last-layer hidden states (last position) for effective rank computation.
    # Only on rank 0 — FSDP all-gathers during forward, so all ranks have identical
    # hidden state values; computing SVD on rank 0 saves redundant work.
    hidden_states_accum: List[torch.Tensor] = []

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

                # Collect last-layer hidden state at final sequence position for
                # effective rank tracking.  Only on rank 0 to save compute.
                if rank == 0 and outputs.hidden_states is not None:
                    _h = outputs.hidden_states[-1][:, -1, :].float().cpu()  # [1, hidden_dim]
                    hidden_states_accum.append(_h)

                memory = outputs.new_memory.detach()

    avg_item_acc   = total_item_acc  / max(n_total,     1)
    avg_ne_acc     = ne_item_acc     / max(n_non_empty, 1)
    avg_count_acc  = total_count_acc / max(n_count,     1)
    non_empty_pct  = 100.0 * n_non_empty / max(n_total, 1)

    # ── Effective rank of hidden states ──────────────────────────────────────
    # Effective rank = exp(H(p)) where p_i = sigma_i / sum(sigmas) is the
    # normalised singular value distribution.  High erank (close to hidden_dim)
    # means the representation uses many independent directions.  Collapsing
    # representations show dramatic erank drops (e.g., 3584 → <20).
    eval_eff_rank   = None
    eval_stable_rank = None
    if rank == 0 and hidden_states_accum:
        H = torch.cat(hidden_states_accum, dim=0)   # [N, hidden_dim]
        # Subsample to 256 frames max to keep SVD fast (< 1s on CPU)
        if H.shape[0] > 256:
            _idx = torch.randperm(H.shape[0])[:256]
            H = H[_idx]
        # Mean-centre so SVD reflects variance structure, not mean offset
        H = H - H.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(H, full_matrices=False)
            S = S.float().clamp(min=0)
            _p = S / (S.sum() + 1e-10)
            eval_eff_rank    = float(torch.exp(-(_p * torch.log(_p + 1e-10)).sum()).item())
            eval_stable_rank = float((S ** 2).sum() / (S[0] ** 2 + 1e-10))
        except Exception as _svd_err:
            if rank == 0:
                print(f"  [WARNING] SVD for effective rank failed: {_svd_err}")
    # ── End effective rank ────────────────────────────────────────────────────

    eval_metrics = {
        'eval_item_acc':          avg_item_acc,
        'eval_non_empty_item_acc': avg_ne_acc,
        'eval_count_acc':         avg_count_acc,
        'eval_non_empty_pct':     non_empty_pct,
        'eval_n_frames':          n_total,
        'eval_seqs':              num_seqs,
        'eval_eff_rank':          eval_eff_rank,
        'eval_stable_rank':       eval_stable_rank,
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
        if eval_eff_rank is not None:
            print(f"  Effective rank (last layer): {eval_eff_rank:.1f}  "
                  f"stable_rank={eval_stable_rank:.1f}  "
                  f"(collapse if eff_rank < 50)")

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


def save_checkpoint(model, step, losses, metrics_logger, output_dir, rank,
                    optimizer=None, lr_scheduler=None):
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
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
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
    parser.add_argument('--seq_load_len', type=int, default=500,
                        help='Frames loaded per sequence.  The BPTT window slides '
                             'through the sequence in bptt_chunk_size steps, carrying '
                             'detached memory across chunks.  Larger values amortise '
                             'MP4 decode cost across more training steps.  Default 500 '
                             'gives ~60 BPTT chunks per disk read with chunk_size=8.')
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
    parser.add_argument('--context_frames', type=int, default=4,
                        help='Number of consecutive frames passed as visual context per forward '
                             'call (JarvisVLA-style).  Frame t sees [t-ctx+1 … t], padded with '
                             'the earliest available frame.  Larger values give the model richer '
                             'temporal context at the cost of ~context_frames× more image tokens.')
    parser.add_argument('--hotbar_only_steps', type=int, default=5000,
                        help='For this many steps, mask main inventory slots (9-35) from '
                             'classification loss when no GUI-open frame has been seen yet '
                             'in the current sequence.  Prevents false gradients from '
                             'penalizing predictions for items the model cannot observe. '
                             'Set 0 to disable.')
    parser.add_argument('--inv_warmup_steps', type=int, default=500,
                        help='Linear warmup steps for inventory/count loss weights.  '
                             'Ramps from 0 → inventory_weight / count_weight over this many '
                             'steps.  Prevents the auxiliary gradient from destabilising the '
                             'backbone before the head has converged (observed 300k grad-norm '
                             'spikes without warmup).')
    parser.add_argument('--lm_weight', type=float, default=1.0,
                        help='Scale factor for the LM cross-entropy loss (both inventory '
                             'description and reasoning tasks). Default 1.0.')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace token for loading teknium/OpenHermes-2.5 reasoning '
                             'dataset. If None, reasoning alternation is skipped and only '
                             'inventory description LM supervision is used.')
    parser.add_argument('--no_cosine_decay', action='store_true',
                        help='Disable cosine LR decay (keep constant LR throughout training).')
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

    # Cosine LR decay: ramps each param group's LR from its initial value to eta_min
    # over the full training run.  The two-group setup (head 5e-4, backbone 5e-6) decays
    # independently — each group is handled by CosineAnnealingLR's per-group tracking.
    if not args.no_cosine_decay:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.train_steps,
            eta_min=1e-7,
        )
    else:
        lr_scheduler = None

    if rank == 0:
        print(f"\n[2/5] Optimizer:")
        print(f"  Head+mem params : LR={args.learning_rate}  ({len(head_params_list)} tensors)")
        print(f"  Backbone params : LR={args.base_model_lr}  ({len(backbone_params_list)} tensors)")
        print(f"  LR schedule     : {'cosine decay → 1e-7' if not args.no_cosine_decay else 'constant'}")
    
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
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Data setup
    if rank == 0:
        print(f"\n[3/5] Loading data...")
    
    from jarvisvla.train.sequence_dataset import InventoryTextEncoder, OnDemandSequenceLoader
    
    encoder = InventoryTextEncoder(device=device)

    jsonl_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    train_files = jsonl_files[:args.train_jsonls]
    test_files  = jsonl_files[args.train_jsonls:args.train_jsonls + args.test_jsonls]
    # Sequence length for the sliding BPTT window.
    # The window advances by bptt_chunk_size each outer step, carrying detached
    # memory across chunks.  seq_load_len controls how many BPTT chunks we get
    # per disk read: roughly (seq_load_len - first_anchor_pos) / bptt_chunk_size.
    seq_load_len = args.seq_load_len
    train_loader = OnDemandSequenceLoader(train_files, seq_load_len, encoder)
    test_loader  = OnDemandSequenceLoader(test_files,  seq_load_len, encoder)

    if rank == 0:
        print(f"  Training sequences: {len(train_loader)}")
        print(f"  Test sequences:     {len(test_loader)}  (files {args.train_jsonls}–{args.train_jsonls + args.test_jsonls})")

    # ── OpenHermes-2.5 reasoning dataset ──────────────────────────────────────
    # Loaded on all ranks; each rank independently cycles through the same
    # shuffled list so reasoning samples stay in sync across GPUs.
    # Only used when --hf_token is provided; otherwise reasoning steps are skipped.
    reasoning_iter = None
    if args.hf_token:
        if rank == 0:
            print(f"\n[3.5/5] Loading OpenHermes-2.5 reasoning dataset...")
        try:
            from datasets import load_dataset as hf_load_dataset
            _hf_ds = hf_load_dataset(
                "teknium/OpenHermes-2.5",
                split="train",
                token=args.hf_token,
            )
            # Keep only clean 2-turn conversations (human → gpt)
            _reasoning_data = [
                x for x in _hf_ds
                if (len(x.get('conversations', [])) >= 2
                    and x['conversations'][0].get('from') in ('human', 'user')
                    and x['conversations'][1].get('from') in ('gpt', 'assistant')
                    and len(x['conversations'][0].get('value', '')) >= 10
                    and len(x['conversations'][1].get('value', '')) >= 10)
            ]
            random.shuffle(_reasoning_data)
            reasoning_iter = itertools.cycle(_reasoning_data)
            if rank == 0:
                print(f"  {len(_reasoning_data):,} reasoning examples loaded and shuffled.")
        except Exception as _e:
            if rank == 0:
                print(f"  [WARNING] Failed to load OpenHermes-2.5: {_e}")
                print(f"  Continuing without reasoning tasks.")
            reasoning_iter = None
    # ── End OpenHermes loading ─────────────────────────────────────────────────

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
    win_type_losses      = []
    win_count_losses     = []
    win_lm_losses        = []
    win_reasoning_losses = []
    # Cached smoothed values — shown at every log step even between update steps
    last_avg_type      = None
    last_avg_count     = None
    last_avg_lm        = None
    last_avg_reasoning = None

    _cur_seq       = None   # currently loaded sequence dict
    _cur_frame_pos = 0      # start of current BPTT chunk within _cur_seq
    _cur_memory    = None   # detached memory carry-over (None = reset to zeros)
    _had_gui_seq   = False  # True once a GUI-open frame has been seen in current sequence

    for step in range(start_step, args.train_steps):
        step_start = time.time()
        
        model.train()
        model.train_aux_heads(True)  # Enable inventory head
        chunk_losses = []
        chunk_type_losses = []        # item CE loss (logged as "type_loss" for plot compat)
        chunk_count_losses = []
        chunk_lm_losses        = []   # LM CE on GUI-frame inventory description targets
        chunk_reasoning_losses = []   # LM CE on reasoning task targets (OpenHermes)
        chunk_n_ne = []               # non-empty slot counts per frame
        chunk_type_losses_gui    = [] # item loss on GUI-open frames
        chunk_type_losses_closed = [] # item loss on GUI-closed frames
        chunk_acc_all:    List[float] = []
        chunk_acc_gui:    List[float] = []
        chunk_acc_closed: List[float] = []
        chunk_n_gui = 0               # how many frames had the GUI open this step

        # Alternating reasoning/inventory supervision.
        # Odd steps → reasoning task from OpenHermes (if available).
        # Even steps → inventory description (structured or NL format).
        # Memory always updates from visual features regardless of step type.
        _is_reasoning_step = (reasoning_iter is not None) and (step % 2 == 1)
        _reasoning_q = _reasoning_a = None
        if _is_reasoning_step:
            _sample = next(reasoning_iter)
            _reasoning_q = _sample['conversations'][0]['value'][:2000]
            _reasoning_a = _sample['conversations'][1]['value'][:1500]

        # Load a new sequence when the current one is exhausted.
        _need_new_seq = (
            _cur_seq is None
            or _cur_frame_pos + args.bptt_chunk_size > len(_cur_seq['frames'])
        )
        if _need_new_seq:
            seq_idx        = torch.randint(0, len(train_loader), (1,)).item()
            _cur_seq       = train_loader[seq_idx]
            _cur_frame_pos = 0
            _cur_memory    = None   # new sequence → reset memory to zeros
            _had_gui_seq   = False  # reset GUI-seen flag for new sequence

            # If the player carries >9 items (beyond the always-visible hotbar),
            # seek to the first GUI-open frame so the model can read the full
            # inventory before the GUI closes.  The subsequent closed-GUI chunks
            # then create the core memory-retention signal we want to train.
            # For ≤9 items the hotbar is sufficient; no seek needed.
            _n_ne_seq = (_cur_seq['inventory_ids'] > 0).sum(dim=1)  # [seq_len]
            _gui_seq  = _cur_seq['is_gui_inventory']
            _gui_rich = [
                i for i in range(len(_gui_seq))
                if _gui_seq[i]
                and _n_ne_seq[i].item() > 9
                and i + args.bptt_chunk_size <= len(_cur_seq['frames'])
            ]
            if _gui_rich:
                _cur_frame_pos = _gui_rich[0]

        seq           = _cur_seq
        chunk_indices = list(range(_cur_frame_pos, _cur_frame_pos + args.bptt_chunk_size))

        # Memory: zeros at sequence start; detached carry-over within the sequence.
        if _cur_memory is None:
            memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)
        else:
            memory = _cur_memory

        scale = args.bptt_chunk_size * args.grad_accum_steps

        # Auxiliary loss weight: linear warmup from 0 → full over inv_warmup_steps.
        # Prevents 300k+ grad-norm spikes observed when starting at full weight.
        _warmup        = min(1.0, step / max(1, args.inv_warmup_steps))
        eff_inv_weight = args.inventory_weight * _warmup
        eff_cnt_weight = args.count_weight     * _warmup

        # ── Preprocessor optimisation ──────────────────────────────────────────
        # 1. Batch-preprocess all chunk frames in ONE image-processor call;
        #    split into per-frame pixel_value tensors for the context window.
        # 2. Tokenise ONCE for a context_frames-image prompt; the placeholder
        #    structure is identical across all frames in the chunk (same resolution
        #    → same image_grid_thw → same number of image tokens per slot).
        _chunk_frames = [seq['frames'][t] for t in chunk_indices]
        _batch_img    = processor.image_processor(images=_chunk_frames, return_tensors="pt")
        _raw_pv       = _batch_img['pixel_values']   # [total_patches, …] packed
        _thw          = _batch_img['image_grid_thw'] # [N, 3]  T×H×W per frame

        _pv_per_frame:  list = []
        _thw_per_frame: list = []
        _offset = 0
        for _fi in range(len(_chunk_frames)):
            _n = int(_thw[_fi, 0] * _thw[_fi, 1] * _thw[_fi, 2])
            _pv_per_frame.append(_raw_pv[_offset:_offset + _n].to(device))
            _thw_per_frame.append(_thw[_fi:_fi + 1].to(device))
            _offset += _n

        # Static tokenisation: context_frames image placeholders + random question.
        # Reused for every frame in the chunk — only pixel_values differ per step.
        _q_prompt    = random.choice(INVENTORY_QUESTION_PROMPTS)
        _dummy_frame = _chunk_frames[0]
        _chat_text   = processor.apply_chat_template(
            [{"role": "user", "content":
                [{"type": "image", "image": _dummy_frame}] * args.context_frames
                + [{"type": "text", "text": _q_prompt}]
            }],
            tokenize=False, add_generation_prompt=True,
        )
        _static = processor(
            text=_chat_text,
            images=[_dummy_frame] * args.context_frames,
            return_tensors="pt",
        )
        _static_input_ids = _static['input_ids'].to(device)       # [1, seq_len]
        _static_attn_mask = _static['attention_mask'].to(device)  # [1, seq_len]
        # ── End preprocessor optimisation ─────────────────────────────────────

        seq_loss = None   # tensor — accumulates the computation graph

        for _t_rel, t in enumerate(chunk_indices):
            is_gui_t = bool(seq['is_gui_inventory'][t])
            if is_gui_t:
                _had_gui_seq = True  # unlocks main inventory supervision for this sequence

            # Sliding context window: last context_frames frames within this chunk,
            # padding the front with the earliest available frame (index 0).
            # The memory token handles longer-range temporal state across chunks.
            _ctx_idxs = [max(0, _t_rel - args.context_frames + 1 + i)
                         for i in range(args.context_frames)]
            multi_pv  = torch.cat([_pv_per_frame[i]  for i in _ctx_idxs], dim=0)
            multi_thw = torch.cat([_thw_per_frame[i] for i in _ctx_idxs], dim=0)

            inv_id_target    = seq['inventory_ids'][t:t+1].to(device)       # [1, 36] long
            inv_count_target = seq['inventory_counts'][t:t+1].to(device)    # [1, 36] long

            # LM supervision: alternates between inventory description and reasoning.
            #
            # Reasoning step (odd steps, frame 0 only):
            #   Apply one OpenHermes Q&A with the current Minecraft frame as visual
            #   context.  The model sees a game screenshot + unrelated question, which
            #   forces the LM head to remain functional for diverse text tasks and
            #   prevents catastrophic forgetting of language capabilities.
            #   Frames 1+ in the chunk: no LM supervision (classification head still runs).
            #
            # Inventory step (even steps, GUI-open frames only):
            #   Teacher-force the model to generate inventory state as [slot_N] text.
            #   Keeps all language pathways active; prevents backbone rank collapse.
            if _is_reasoning_step and _t_rel == 0:
                # Reasoning task: context_frames images + OpenHermes Q → A
                _rsn_full_chat = processor.apply_chat_template(
                    [{"role": "user", "content":
                        [{"type": "image", "image": _dummy_frame}] * args.context_frames
                        + [{"type": "text", "text": _reasoning_q}]},
                     {"role": "assistant", "content": _reasoning_a}],
                    tokenize=False, add_generation_prompt=False,
                )
                _rsn_q_chat = processor.apply_chat_template(
                    [{"role": "user", "content":
                        [{"type": "image", "image": _dummy_frame}] * args.context_frames
                        + [{"type": "text", "text": _reasoning_q}]}],
                    tokenize=False, add_generation_prompt=True,
                )
                _rsn_full_enc = processor(
                    text=_rsn_full_chat,
                    images=[_dummy_frame] * args.context_frames,
                    return_tensors="pt",
                )
                _rsn_q_len = processor(
                    text=_rsn_q_chat,
                    images=[_dummy_frame] * args.context_frames,
                    return_tensors="pt",
                )['input_ids'].shape[1]
                _cur_input_ids = _rsn_full_enc['input_ids'].to(device)
                _cur_attn_mask = _rsn_full_enc['attention_mask'].to(device)
                _cur_labels    = _cur_input_ids.clone()
                _cur_labels[0, :_rsn_q_len] = -100   # mask question, supervise answer

            elif not _is_reasoning_step and is_gui_t:
                # Inventory description: [slot_N] item_name count:C format (structured)
                _slot_texts_t = seq['inventory_slot_texts'][t]          # List[str], len 36
                _slot_cnts_t  = seq['inventory_counts'][t].tolist()     # List[int], len 36
                _answer_text  = build_slot_text_target(_slot_texts_t, _slot_cnts_t)

                # Build full chat (question + answer) then tokenize.
                # The question prefix in _full_input_ids is identical to _static_input_ids
                # (same images, same prompt) so we can mask it by length.
                _full_chat = processor.apply_chat_template(
                    [{"role": "user", "content":
                        [{"type": "image", "image": _dummy_frame}] * args.context_frames
                        + [{"type": "text", "text": _q_prompt}]},
                     {"role": "assistant", "content": _answer_text}],
                    tokenize=False, add_generation_prompt=False,
                )
                _full_enc = processor(
                    text=_full_chat,
                    images=[_dummy_frame] * args.context_frames,
                    return_tensors="pt",
                )
                _cur_input_ids = _full_enc['input_ids'].to(device)
                _cur_attn_mask = _full_enc['attention_mask'].to(device)
                # Mask question tokens in labels; only supervise the answer portion.
                _cur_labels    = _cur_input_ids.clone()
                _cur_labels[0, :_static_input_ids.shape[1]] = -100

            else:
                # No LM supervision: closed frame on inventory step, or non-first
                # frame on reasoning step.  Classification head still trains.
                _cur_input_ids = _static_input_ids
                _cur_attn_mask = _static_attn_mask
                _cur_labels    = None

            outputs = model(
                input_ids=_cur_input_ids,
                pixel_values=multi_pv,
                image_grid_thw=multi_thw,
                prev_memory=memory,
                attention_mask=_cur_attn_mask,
                labels=_cur_labels,
            )

            # No detach — memory carries the gradient graph across frames.
            memory = outputs.new_memory

            memory_reg = 0.1 * (memory ** 2).mean()
            frame_loss = memory_reg / scale

            # LM loss: scaled by lm_weight for both inventory description and reasoning.
            # Keeps all language pathways active → prevents backbone rank collapse.
            if outputs.loss is not None:
                frame_loss = frame_loss + args.lm_weight * outputs.loss / scale
                if _is_reasoning_step and _t_rel == 0:
                    chunk_reasoning_losses.append(outputs.loss.item())
                else:
                    chunk_lm_losses.append(outputs.loss.item())

            if outputs.item_logits is not None:
                item_logits  = outputs.item_logits[0]    # [36, item_vocab_size]
                count_logits = outputs.count_logits[0]   # [36, count_classes]
                tgt_ids      = inv_id_target[0]          # [36] long
                tgt_counts   = inv_count_target[0]       # [36] long

                non_empty = tgt_ids > 0                  # [36] bool mask (0 = empty)
                N_ne = int(non_empty.sum().item())
                chunk_n_ne.append(N_ne)

                # Hotbar curriculum: before hotbar_only_steps, when no GUI-open frame
                # has been seen in this sequence, mask main inventory slots (9-35).
                # Slots 0-8 (hotbar) are always visible on screen.
                # Slots 9-35 require GUI open — penalising predictions for items the
                # model cannot see creates false gradients toward modal items.
                _supervised = torch.ones(36, dtype=torch.bool, device=device)
                if step < args.hotbar_only_steps and not _had_gui_seq:
                    _supervised[9:] = False

                # Item type loss: focal CE over supervised slots.
                _ce_per_slot = F.cross_entropy(item_logits, tgt_ids, reduction='none')  # [36]
                _pt    = torch.exp(-_ce_per_slot)
                _focal = (1 - _pt) ** 2.0 * _ce_per_slot  # gamma=2
                _slot_w = torch.where(tgt_ids == 0,
                                      torch.full_like(_focal, 0.1),
                                      torch.ones_like(_focal))
                _slot_w = _slot_w * _supervised.float()
                item_loss = (_focal * _slot_w).sum() / _slot_w.sum().clamp(min=1.0)

                # Count loss: focal CE over supervised non-empty slots only.
                count_loss = None
                _sup_ne = non_empty & _supervised
                N_sup_ne = int(_sup_ne.sum().item())
                if N_sup_ne > 0:
                    _cnt_ce = F.cross_entropy(
                        count_logits[_sup_ne],
                        tgt_counts[_sup_ne],
                        reduction='none',
                    )
                    _cnt_pt    = torch.exp(-_cnt_ce)
                    _cnt_focal = (1 - _cnt_pt) ** 2.0 * _cnt_ce
                    _cnt_w = torch.where(
                        tgt_counts[_sup_ne] == 1,
                        torch.full_like(_cnt_focal, 0.1),
                        torch.ones_like(_cnt_focal),
                    )
                    count_loss = (_cnt_focal * _cnt_w).sum() / _cnt_w.sum()

                frame_loss = frame_loss + eff_inv_weight * item_loss / scale
                tl_val = item_loss.item()
                chunk_type_losses.append(tl_val)

                # Top-1 accuracy — split into GUI / closed for the key diagnostic.
                with torch.no_grad():
                    pred_ids = item_logits.argmax(dim=-1)  # [36]
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
                    frame_loss = frame_loss + eff_cnt_weight * count_loss / scale
                    chunk_count_losses.append(count_loss.item())

            seq_loss = frame_loss if seq_loss is None else seq_loss + frame_loss
            chunk_losses.append(frame_loss.item() * scale)  # display (forces CUDA sync)

            # Hard norm cap: clamp memory to unit ball before next frame.
            memory = memory / memory.norm(dim=-1, keepdim=True).clamp(min=1.0)

        # Snapshot memory norm before backward frees the graph.
        memory_norm_end = memory.detach().float().norm().item()

        # Single backward through the full bptt_chunk_size frame sequence.
        # Advance the sliding window BEFORE backward so _cur_memory is a plain
        # data tensor unaffected by graph freeing.
        _cur_memory    = memory.detach()            # carry to next chunk; cut gradient flow
        _cur_frame_pos += args.bptt_chunk_size      # advance sliding window

        # Gradient flows back through GRU recurrence across all frames in this chunk.
        if seq_loss is not None:
            seq_loss.backward()

        # Feed this chunk into the update-window accumulators
        win_type_losses.extend(chunk_type_losses)
        win_count_losses.extend(chunk_count_losses)
        win_lm_losses.extend(chunk_lm_losses)
        win_reasoning_losses.extend(chunk_reasoning_losses)

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
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Compute smoothed display values from the just-completed update window
            # and reset the accumulators for the next window.
            last_avg_type      = sum(win_type_losses)      / len(win_type_losses)      if win_type_losses      else None
            last_avg_count     = sum(win_count_losses)     / len(win_count_losses)     if win_count_losses     else None
            last_avg_lm        = sum(win_lm_losses)        / len(win_lm_losses)        if win_lm_losses        else None
            last_avg_reasoning = sum(win_reasoning_losses) / len(win_reasoning_losses) if win_reasoning_losses else None
            win_type_losses.clear()
            win_count_losses.clear()
            win_lm_losses.clear()
            win_reasoning_losses.clear()

        if chunk_losses:
            losses.append(sum(chunk_losses))

            avg_loss       = sum(chunk_losses)
            avg_type_loss  = sum(chunk_type_losses)  / len(chunk_type_losses)  if chunk_type_losses  else 0.0
            avg_count_loss = sum(chunk_count_losses) / len(chunk_count_losses) if chunk_count_losses else 0.0
            avg_lm_loss        = sum(chunk_lm_losses)        / len(chunk_lm_losses)        if chunk_lm_losses        else 0.0
            avg_reasoning_loss = sum(chunk_reasoning_losses) / len(chunk_reasoning_losses) if chunk_reasoning_losses else 0.0
            avg_n_ne           = sum(chunk_n_ne)             / len(chunk_n_ne)             if chunk_n_ne             else 0.0
            # GUI vs closed item accuracy — THE key diagnostic.
            # closed-frame accuracy should rise over training as the model
            # learns to retain inventory state through the GRU memory.
            acc_gui    = sum(chunk_acc_gui)    / len(chunk_acc_gui)    if chunk_acc_gui    else None
            acc_closed = sum(chunk_acc_closed) / len(chunk_acc_closed) if chunk_acc_closed else None
            acc_all_avg= sum(chunk_acc_all)    / len(chunk_acc_all)    if chunk_acc_all    else None

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Current LR for each param group (only meaningful on update steps)
            _lr_head     = optimizer.param_groups[0]['lr']
            _lr_backbone = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else _lr_head

            metrics = {
                'loss':               avg_loss,
                'type_loss':          avg_type_loss,       # item CE loss (named type_loss for plot compat)
                'count_loss':         avg_count_loss,
                'lm_loss':            avg_lm_loss,         # LM CE on inventory description frames
                'reasoning_loss':     avg_reasoning_loss,  # LM CE on OpenHermes reasoning frames
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
                'lr_head':            _lr_head,
                'lr_backbone':        _lr_backbone,
                'step_time':          step_time,
                'is_update_step':     is_update_step,
            }

            logger.log(metrics, step)

            if rank == 0 and step % args.log_every == 0:
                gn_str  = f"{last_gn:.1f}"  if last_gn  is not None else "  N/A"
                hgn_str = f"{last_hgn:.3f}" if last_hgn is not None else "  N/A"
                wn_str  = f"{last_wn:.3f}"  if last_wn  is not None else "    N/A"
                # Use smoothed (update-window averaged) values for display
                disp_type      = last_avg_type      if last_avg_type      is not None else avg_type_loss
                disp_count     = last_avg_count     if last_avg_count     is not None else avg_count_loss
                disp_lm        = last_avg_lm        if last_avg_lm        is not None else avg_lm_loss
                disp_reasoning = last_avg_reasoning if last_avg_reasoning is not None else avg_reasoning_loss
                gui_str    = f"{acc_gui:.3f}"    if acc_gui    is not None else "  N/A"
                closed_str = f"{acc_closed:.3f}" if acc_closed is not None else "  N/A"
                step_type  = "rsn" if _is_reasoning_step else "inv"
                flag = "  *** NO GUI ***" if chunk_n_gui == 0 and not _is_reasoning_step and avg_n_ne > 9 else ""
                print(f"{step:>8}[{step_type}] item={disp_type:.3f} "
                      f"lm={disp_lm:.3f} rsn={disp_reasoning:.3f} "
                      f"acc(gui={gui_str} cls={closed_str})"
                      f"  cnt={disp_count:.4f}  N_ne={avg_n_ne:4.1f} gui={chunk_n_gui:3d}"
                      f"  gN={gn_str} hGN={hgn_str} wN={wn_str} mN={memory_norm_end:.2f}"
                      f"  {step_time:>6.1f}s{flag}")

        # Checkpoint
        if (step + 1) % args.eval_every == 0:
            save_checkpoint(model, step + 1, losses, logger, args.output_dir, rank,
                            optimizer=optimizer, lr_scheduler=lr_scheduler)

    # Final save
    save_checkpoint(model, args.train_steps, losses, logger, args.output_dir, rank,
                    optimizer=optimizer, lr_scheduler=lr_scheduler)

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
