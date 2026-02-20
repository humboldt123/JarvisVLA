#!/usr/bin/env python3
"""
Standalone evaluation script for StatefulJarvisVLA checkpoints.

Loads a saved checkpoint (FULL_STATE_DICT format from train_full_unfrozen_fsdp.py)
and evaluates it on held-out test sequences, printing per-slot NN-decoded predictions.

Usage:
    python jarvisvla/train/eval_checkpoint.py \
        --checkpoint /data/vvm33/checkpoints/full_unfrozen_fsdp/checkpoint_step_1001.pt \
        --data_dir   /data/vvm33/vpt_contractor \
        --train_jsonls 4400 \
        --test_jsonls  10

Runs on a single GPU (no torchrun needed) — bfloat16 keeps the 7B model at ~14 GB VRAM.
"""

import sys
import argparse
import math
import json
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/vvm33/JarvisVLA')
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from jarvisvla.models.stateful_vla import wrap_model_for_stateful_training
from jarvisvla.train.sequence_dataset import InventoryTextEncoder, OnDemandSequenceLoader


def evaluate(model, test_loader, processor, args, device):
    num_seqs = len(test_loader)
    print(f"\nEvaluating on {num_seqs} test sequences ({num_seqs * 50} frames)...")

    model.eval()
    model.train_aux_heads(True)
    if model.inventory_embedding_head is not None:
        model.inventory_embedding_head.eval()

    total_cos = 0.0
    non_empty_cos = 0.0
    total_count_mae = 0.0
    n_total = 0
    n_non_empty = 0
    n_count = 0
    vocab_set: set = set()
    display_examples = []

    with torch.no_grad():
        for seq_idx in range(num_seqs):
            if seq_idx % 5 == 0:
                print(f"  seq {seq_idx + 1}/{num_seqs} ...", end='\r', flush=True)

            seq = test_loader[seq_idx]
            memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)

            for t in range(len(seq['frames'])):
                frame            = seq['frames'][t]
                inv_type_target  = seq['inventory_embeddings'][t:t+1].to(device, dtype=torch.bfloat16)
                inv_count_target = seq['inventory_counts'][t:t+1].to(device)
                is_non_empty     = seq['inventory_has_items'][t]
                slot_type_texts  = seq.get('inventory_slot_texts',
                                           [['empty slot'] * 36] * len(seq['frames']))[t]
                slot_counts      = seq['inventory_counts'][t]

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
                    pred_type  = outputs.inventory_embedding[0]
                    pred_count = outputs.inventory_count[0]
                    tgt_type   = inv_type_target[0]
                    tgt_count  = inv_count_target[0].float()

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

                    for txt in slot_type_texts:
                        if txt != "empty slot":
                            vocab_set.add(txt)

                    if is_non_empty and len(display_examples) < 5:
                        display_examples.append({
                            'source': seq.get('source_file', '?'),
                            'tick': t,
                            'mean_cos': cos_sim,
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

    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION  ({num_seqs} seqs, {n_total} frames)")
    print(f"{'='*70}")
    print(f"  Type cos-sim (all frames)   : {avg_cos:.4f}")
    print(f"  Type cos-sim (non-empty)    : {avg_ne_cos:.4f}"
          f"  ({n_non_empty}/{n_total} = {non_empty_pct:.1f}%)")
    print(f"  Count MAE  (non-empty slots): {avg_count_mae:.2f} items")

    if display_examples and vocab_set:
        enc = test_loader.encoder
        vocab_list = sorted(vocab_set)
        vocab_embs_cpu = torch.stack([
            torch.from_numpy(enc._name_to_embedding(name)) for name in vocab_list
        ])  # [vocab_size, 768], already L2-normalised

        print(f"\n--- NN-decoded slot predictions (up to 5 non-empty frames) ---")
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
                      f"  GT: {type_text:<22} cnt:{true_cnt:<4}"
                      f"PRED: {pred_type:<22} cnt:{pred_cnt:<4}"
                      f"  {bar} {cs:.3f} {match}")

            if gt_items or pred_items:
                jaccard = len(gt_items & pred_items) / len(gt_items | pred_items)
                print(f"           Jaccard (types): {jaccard:.2f}"
                      f"  GT={sorted(gt_items)}  PRED={sorted(pred_items)}")

    print(f"{'='*70}")
    return {
        'eval_cos_sim':           avg_cos,
        'eval_non_empty_cos_sim': avg_ne_cos,
        'eval_count_mae':         avg_count_mae,
        'eval_non_empty_pct':     non_empty_pct,
        'eval_n_frames':          n_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True, help='Path to .pt checkpoint')
    parser.add_argument('--data_dir',      default='/data/vvm33/vpt_contractor')
    parser.add_argument('--train_jsonls',  type=int, default=4400,
                        help='Number of train files (test files start after these)')
    parser.add_argument('--test_jsonls',   type=int, default=10)
    parser.add_argument('--memory_dim',    type=int, default=1024)
    parser.add_argument('--output_json',   default=None,
                        help='Optional path to save eval metrics as JSON')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading Qwen2-VL-7B...")
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
    model = model.to(device, dtype=torch.bfloat16)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Checkpoint step: {ckpt.get('step', '?')}")

    print("Loading test data...")
    encoder = InventoryTextEncoder(device=str(device))
    jsonl_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    test_files  = jsonl_files[args.train_jsonls : args.train_jsonls + args.test_jsonls]
    test_loader = OnDemandSequenceLoader(test_files, 50, encoder)
    print(f"  Test sequences: {len(test_loader)}")

    metrics = evaluate(model, test_loader, processor, args, device)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics saved to {args.output_json}")


if __name__ == '__main__':
    main()
