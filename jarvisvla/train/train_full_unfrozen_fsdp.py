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


def get_fsdp_model(model, device_id):
    """Wrap model with FSDP."""
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    def wrap_policy_fn(module):
        if isinstance(module, Qwen2VLDecoderLayer):
            return True
        return False
    
    auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=wrap_policy_fn)
    
    model = FSDP(
        model,
        device_id=device_id,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
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

    total_cos = 0.0
    non_empty_cos = 0.0
    total_count_mae = 0.0
    n_total = 0
    n_non_empty = 0
    n_count = 0
    vocab_set: set = set()
    display_examples: List[Dict] = []

    with torch.no_grad():
        for seq_idx in range(num_seqs):
            if rank == 0 and seq_idx % 5 == 0:
                print(f"  seq {seq_idx + 1}/{num_seqs} ...", end='\r', flush=True)

            seq = test_loader[seq_idx]
            memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)

            for t in range(len(seq['frames'])):
                frame            = seq['frames'][t]
                inv_type_target  = seq['inventory_embeddings'][t:t+1].to(device, dtype=torch.bfloat16)  # [1,36,768]
                inv_count_target = seq['inventory_counts'][t:t+1].to(device)      # [1,36] long
                is_non_empty     = seq['inventory_has_items'][t]
                slot_type_texts  = seq.get('inventory_slot_texts',
                                           [['empty slot']*36]*len(seq['frames']))[t]
                slot_counts      = seq['inventory_counts'][t]  # [36] long, cpu

                chat_text = processor.apply_chat_template(
                    [{"role": "user", "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": "What is in the inventory?"},
                    ]}],
                    tokenize=False, add_generation_prompt=True,
                )
                inputs = processor(text=chat_text, images=[frame], return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}

                outputs = model(
                    input_ids=inputs.get('input_ids'),
                    pixel_values=inputs.get('pixel_values'),
                    image_grid_thw=inputs.get('image_grid_thw'),
                    prev_memory=memory,
                    attention_mask=inputs.get('attention_mask'),
                    labels=None,
                )

                if outputs.inventory_embedding is not None:
                    pred_type  = outputs.inventory_embedding[0]  # [36, 768]
                    pred_count = outputs.inventory_count[0]       # [36] log(count+1) preds
                    tgt_type   = inv_type_target[0]               # [36, 768]
                    tgt_count  = inv_count_target[0].float()      # [36]

                    per_slot_cos = (pred_type * tgt_type).sum(dim=-1)
                    cos_sim = per_slot_cos.mean().item()
                    total_cos += cos_sim
                    n_total += 1

                    if is_non_empty:
                        non_empty_cos += cos_sim
                        n_non_empty += 1
                        non_empty_mask = tgt_count > 0
                        if non_empty_mask.any():
                            pred_cnt_exp = (torch.exp(pred_count[non_empty_mask]) - 1).clamp(min=0)
                            mae = (pred_cnt_exp - tgt_count[non_empty_mask]).abs().mean().item()
                            total_count_mae += mae
                            n_count += 1

                    if rank == 0:
                        for txt in slot_type_texts:
                            if txt != "empty slot":
                                vocab_set.add(txt)

                        if is_non_empty and len(display_examples) < 3:
                            display_examples.append({
                                'source': seq.get('source_file', '?'),
                                'tick': t,
                                'mean_cos': cos_sim,
                                # (slot_idx, type_text, true_count, pred_type_emb,
                                #  pred_log_count, per_slot_cos)
                                'slots': [
                                    (i, slot_type_texts[i],
                                     int(slot_counts[i].item()),
                                     pred_type[i].float().cpu(),
                                     pred_count[i].float().item(),
                                     per_slot_cos[i].item())
                                    for i in range(len(slot_type_texts))
                                    if slot_type_texts[i] != "empty slot"
                                ],
                            })

                memory = outputs.new_memory.detach()

    avg_cos       = total_cos       / max(n_total,     1)
    avg_ne_cos    = non_empty_cos   / max(n_non_empty, 1)
    avg_count_mae = total_count_mae / max(n_count,     1)
    non_empty_pct = 100.0 * n_non_empty / max(n_total, 1)

    eval_metrics = {
        'eval_cos_sim':           avg_cos,
        'eval_non_empty_cos_sim': avg_ne_cos,
        'eval_count_mae':         avg_count_mae,
        'eval_non_empty_pct':     non_empty_pct,
        'eval_n_frames':          n_total,
        'eval_seqs':              num_seqs,
    }
    logger.log_summary({'final_eval': eval_metrics})

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"TEST SET EVALUATION  ({num_seqs} seqs, {n_total} frames)")
        print(f"{'='*70}")
        print(f"  Type cos-sim (all frames)  : {avg_cos:.4f}")
        print(f"  Type cos-sim (non-empty)   : {avg_ne_cos:.4f}"
              f"  ({n_non_empty}/{n_total} = {non_empty_pct:.1f}%)")
        print(f"  Count MAE  (non-empty slots): {avg_count_mae:.2f} items")

        if display_examples and vocab_set:
            enc = test_loader.encoder
            vocab_list = sorted(vocab_set)
            # Trigram encoder: embed each item name directly (no tokenizer/model attrs)
            vocab_embs_cpu = torch.stack([
                torch.from_numpy(enc._name_to_embedding(name)) for name in vocab_list
            ])  # [vocab_size, 768], already L2-normalised

            print(f"\n--- Inventory head predictions (NN-decoded, non-empty frames) ---")
            for ex in display_examples:
                gt_items   = {txt for _, txt, _, _, _, _ in ex['slots']}
                pred_items: set = set()

                print(f"\n  [{ex['source']}  tick {ex['tick']}]  type cos={ex['mean_cos']:.3f}")
                for slot_idx, type_text, true_cnt, pred_emb, pred_log_cnt, cs in ex['slots']:
                    pred_cnt = max(0, round(math.exp(pred_log_cnt) - 1))
                    sims = (pred_emb.unsqueeze(0) @ vocab_embs_cpu.T).squeeze(0)
                    pred_type = vocab_list[sims.argmax().item()]
                    pred_items.add(pred_type)
                    match = '✓' if pred_type == type_text else '✗'
                    bar = '█' * int(cs * 20) + '░' * (20 - int(cs * 20))
                    print(f"    slot {slot_idx:>2}"
                          f"  GT: {type_text:<20} cnt:{true_cnt:<4}"
                          f"PRED: {pred_type:<20} cnt:{pred_cnt:<4}"
                          f"  {bar} {cs:.3f} {match}")

                if gt_items or pred_items:
                    jaccard = len(gt_items & pred_items) / len(gt_items | pred_items)
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
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--base_model_lr', type=float, default=1e-6)
    parser.add_argument('--bptt_chunk_size', type=int, default=4)
    parser.add_argument('--memory_dim', type=int, default=1024)
    parser.add_argument('--inventory_weight', type=float, default=1.0,
                        help='Scale factor on the InfoNCE type loss')
    parser.add_argument('--count_weight', type=float, default=0.1,
                        help='Scale factor on the count MSE loss')
    parser.add_argument('--inv_temperature', type=float, default=0.5,
                        help='InfoNCE temperature for inventory type loss')
    parser.add_argument('--lm_weight', type=float, default=1.0,
                        help='Weight for LM (inventory description) loss; 0 disables it')
    parser.add_argument('--grad_accum_steps', type=int, default=4)
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
    
    model = wrap_model_for_stateful_training(
        base_model=base_model,
        memory_dim=args.memory_dim,
        add_inventory_head=True,
        inventory_head_kwargs={'output_dim': 768},
    )
    
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    
    if rank == 0:
        print(f"  Total params: {total_params:,}")
        print(f"  All parameters unfrozen (ViT + LLM + memory + inventory)")
    
    # FSDP setup
    model = model.to(device, dtype=torch.bfloat16)
    model = get_fsdp_model(model, device_id=local_rank)
    
    # Optimizer
    param_groups = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.base_model_lr},
    ]
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'memory_projections' in name or 'inventory_embedding_head' in name:
                param_groups[0]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)
    
    optimizer = torch.optim.AdamW(param_groups)
    
    if rank == 0:
        print(f"\n[2/5] Optimizer:")
        print(f"  New params (memory+inventory): LR={args.learning_rate}")
        print(f"  Base params (ViT+LLM): LR={args.base_model_lr}")
    
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
    train_loader = OnDemandSequenceLoader(train_files, 50, encoder)
    test_loader  = OnDemandSequenceLoader(test_files,  50, encoder)

    if rank == 0:
        print(f"  Training sequences: {len(train_loader)}")
        print(f"  Test sequences:     {len(test_loader)}  (files {args.train_jsonls}–{args.train_jsonls + args.test_jsonls})")
    
    # Training loop
    if rank == 0:
        print(f"\n[4/5] Training from step {start_step} to {args.train_steps}...")
        print(f"  Logging every {args.log_every} steps")
        print(f"  Checkpointing every {args.eval_every} steps")
        print(f"\n{'Step':>8} {'Loss':>10} {'LMLoss':>10} {'TypeLoss':>10} {'CntLoss':>10} {'Time':>8}")
        print("-"*66)
    
    step_times = []
    
    for step in range(start_step, args.train_steps):
        step_start = time.time()
        
        model.train()
        model.train_aux_heads(True)  # Enable inventory head
        chunk_losses = []
        chunk_lm_losses = []
        chunk_type_losses = []
        chunk_count_losses = []

        memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)

        seq_idx = torch.randint(0, len(train_loader), (1,)).item()
        seq = train_loader[seq_idx]
        start_idx = torch.randint(0, max(1, len(seq['frames']) - args.bptt_chunk_size), (1,)).item()

        scale = args.bptt_chunk_size * args.grad_accum_steps

        for t in range(start_idx, min(start_idx + args.bptt_chunk_size, len(seq['frames']))):
            frame = seq['frames'][t]
            inv_type_target  = seq['inventory_embeddings'][t:t+1].to(device, dtype=torch.bfloat16)  # [1, 36, 768]
            inv_count_target = seq['inventory_counts'][t:t+1].to(device)      # [1, 36] long

            # Build inventory description for LM supervision.
            # Random question + prefix so the model can't short-circuit on phrasing.
            question      = random.choice(INVENTORY_QUESTION_PROMPTS)
            answer_prefix = random.choice(INVENTORY_ANSWER_PREFIXES)
            slot_type_texts_t = seq.get(
                'inventory_slot_texts',
                [['empty slot'] * 36] * len(seq['frames']),
            )[t]
            slot_counts_cpu = seq['inventory_counts'][t].tolist()
            answer = build_inventory_answer(slot_type_texts_t, slot_counts_cpu, answer_prefix)

            # Full conversation (user + assistant) — model learns to generate the answer.
            chat_text = processor.apply_chat_template(
                [
                    {"role": "user", "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": question},
                    ]},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=False, add_generation_prompt=False,
            )
            inputs = processor(text=chat_text, images=[frame], return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Label mask: -100 for all prompt/image tokens; only supervise answer tokens.
            # We tokenize the answer in isolation to count its tokens, then mask the prefix.
            n_ans  = len(processor.tokenizer(answer, add_special_tokens=False)['input_ids'])
            labels = inputs['input_ids'].clone()
            labels[:, :-n_ans] = -100
            labels = labels.to(device)

            outputs = model(
                input_ids=inputs.get('input_ids'),
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                prev_memory=memory,
                attention_mask=inputs.get('attention_mask'),
                labels=labels,
            )

            memory_reg = 0.01 * (outputs.new_memory ** 2).mean()
            step_loss = memory_reg / scale  # base: always backprop memory reg

            # LM loss: outputs.main_loss is CE only (memory_reg already in step_loss above).
            if outputs.main_loss is not None and args.lm_weight > 0:
                step_loss = step_loss + args.lm_weight * outputs.main_loss / scale
                chunk_lm_losses.append(outputs.main_loss.item())

            if outputs.inventory_embedding is not None:
                pred_type  = outputs.inventory_embedding[0]   # [36, 768]
                pred_count = outputs.inventory_count[0]        # [36]
                tgt_type   = inv_type_target[0]                # [36, 768]
                tgt_count  = inv_count_target[0].float()       # [36]

                non_empty = tgt_count > 0                      # [36] bool mask
                N_ne = int(non_empty.sum().item())

                # --- Type loss: InfoNCE over unique item types in this frame ---
                # pred and tgt are already L2-normalised by the inventory head.
                # We deduplicate by target embedding before forming the similarity
                # matrix: if two slots share the same item type their trigram
                # embeddings are bit-identical, which would create contradictory
                # gradients (push pred[i] toward V AND away from V simultaneously).
                # Dedup removes all but the first occurrence so every off-diagonal
                # entry in the matrix is a genuinely different item type.
                if N_ne >= 2:
                    ne_pred = pred_type[non_empty]   # [N_ne, 768] unit vectors
                    ne_tgt  = tgt_type[non_empty]    # [N_ne, 768] unit vectors

                    # Drop duplicate item types (cos_sim > 0.999 → identical trigram emb)
                    tgt_sim = ne_tgt @ ne_tgt.T
                    unique_mask = torch.ones(N_ne, dtype=torch.bool, device=device)
                    for i in range(1, N_ne):
                        if (tgt_sim[i, :i] > 0.999).any():
                            unique_mask[i] = False
                    ne_pred_u = ne_pred[unique_mask]
                    ne_tgt_u  = ne_tgt[unique_mask]
                    N_u = int(unique_mask.sum())

                    if N_u >= 2:
                        logits   = (ne_pred_u @ ne_tgt_u.T) / args.inv_temperature
                        type_loss = F.cross_entropy(logits, torch.arange(N_u, device=device))
                    else:
                        # Only one unique type after dedup — fall back to cosine
                        type_loss = 1.0 - (ne_pred_u * ne_tgt_u).sum(dim=-1).mean()
                elif N_ne == 1:
                    type_loss = 1.0 - (pred_type[non_empty] * tgt_type[non_empty]).sum(dim=-1).mean()
                else:
                    type_loss = None

                # --- Count loss: MSE on log(count+1) for non-empty slots ---
                if N_ne > 0:
                    count_loss = F.mse_loss(
                        pred_count[non_empty],
                        torch.log(tgt_count[non_empty] + 1).to(pred_count.dtype),
                    )
                else:
                    count_loss = None

                if type_loss is not None:
                    step_loss = step_loss + args.inventory_weight * type_loss / scale
                    chunk_type_losses.append(type_loss.item())
                if count_loss is not None:
                    step_loss = step_loss + args.count_weight * count_loss / scale
                    chunk_count_losses.append(count_loss.item())

            step_loss.backward()
            chunk_losses.append(step_loss.item() * scale)

            memory = outputs.new_memory.detach()
        
        # Gradient accumulation: accumulate for grad_accum_steps outer steps before
        # updating.  The loss is already divided by (bptt_chunk_size * grad_accum_steps)
        # so after grad_accum_steps steps the effective batch covers
        # grad_accum_steps * bptt_chunk_size timesteps.
        is_update_step = (
            ((step - start_step + 1) % args.grad_accum_steps == 0)
            or (step + 1 == args.train_steps)
        )

        grad_norm = None
        if is_update_step and chunk_losses:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if chunk_losses:
            losses.append(sum(chunk_losses))

            avg_loss       = sum(chunk_losses)
            avg_lm_loss    = sum(chunk_lm_losses)    / len(chunk_lm_losses)    if chunk_lm_losses    else 0.0
            avg_type_loss  = sum(chunk_type_losses)  / len(chunk_type_losses)  if chunk_type_losses  else 0.0
            avg_count_loss = sum(chunk_count_losses) / len(chunk_count_losses) if chunk_count_losses else 0.0

            step_time = time.time() - step_start
            step_times.append(step_time)

            metrics = {
                'loss': avg_loss,
                'lm_loss': avg_lm_loss,
                'type_loss': avg_type_loss,
                'count_loss': avg_count_loss,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm or 0.0),
                'step_time': step_time,
                'is_update_step': is_update_step,
            }

            logger.log(metrics, step)

            if rank == 0 and step % args.log_every == 0:
                print(f"{step:>8} {avg_loss:>10.4f} {avg_lm_loss:>10.4f} {avg_type_loss:>10.4f} {avg_count_loss:>10.4f} {step_time:>8.2f}s")

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
            print(f"  Eval type cos-sim (all):     {eval_metrics['eval_cos_sim']:.4f}")
            print(f"  Eval type cos-sim (non-empty): {eval_metrics['eval_non_empty_cos_sim']:.4f}")
            print(f"  Eval count MAE:              {eval_metrics['eval_count_mae']:.2f}")
    
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
