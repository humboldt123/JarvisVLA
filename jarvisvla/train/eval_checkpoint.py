#!/usr/bin/env python3
"""
Standalone evaluation script for StatefulJarvisVLA checkpoints.

Loads a saved checkpoint (FULL_STATE_DICT format from train_full_unfrozen_fsdp.py)
and evaluates it on held-out test sequences, printing per-slot predictions.

Usage:
    python jarvisvla/train/eval_checkpoint.py \
        --checkpoint /data/vvm33/checkpoints/train_inv_head/checkpoint_step_5000.pt \
        --data_dir   /data/vvm33/vpt_contractor \
        --train_jsonls 4000 \
        --test_jsonls  5 \
        --sequence_length 200

Runs on a single GPU (no torchrun needed) — bfloat16 keeps the 7B model at ~14 GB VRAM.
"""

import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/vvm33/JarvisVLA')
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from jarvisvla.models.stateful_vla import wrap_model_for_stateful_training
from jarvisvla.train.sequence_dataset import InventoryClassEncoder, OnDemandSequenceLoader
from jarvisvla.train.mc_item_vocab import item_id_to_name, get_item_vocab_size


def evaluate(model, test_loader, processor, args, device):
    num_seqs = len(test_loader)
    seq_len  = test_loader.sequence_length
    print(f"\nEvaluating on {num_seqs} test sequences ({num_seqs * seq_len} frames)...")

    model.eval()
    model.train_aux_heads(True)
    if model.inventory_embedding_head is not None:
        model.inventory_embedding_head.eval()

    total_item_acc  = 0.0
    ne_item_acc     = 0.0
    gui_item_acc    = 0.0
    closed_item_acc = 0.0
    total_count_acc = 0.0
    n_total     = 0
    n_non_empty = 0
    n_gui       = 0
    n_closed    = 0
    n_count     = 0
    display_examples = []

    # Build chat template once
    _dummy = test_loader[0]['frames'][0]
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": _dummy},
            {"type": "text", "text": "What is in the inventory?"},
        ]}],
        tokenize=False, add_generation_prompt=True,
    )

    with torch.no_grad():
        for seq_idx in range(num_seqs):
            if seq_idx % 5 == 0:
                print(f"  seq {seq_idx + 1}/{num_seqs} ...", end='\r', flush=True)

            seq = test_loader[seq_idx]
            seq_len = len(seq['frames'])
            memory = torch.zeros(1, args.memory_dim, device=device, dtype=torch.bfloat16)
            is_gui_list = seq.get('is_gui_inventory', [False] * seq_len)

            for t in range(len(seq['frames'])):
                frame            = seq['frames'][t]
                inv_id_target    = seq['inventory_ids'][t:t+1].to(device)      # [1, 36] long
                inv_count_target = seq['inventory_counts'][t:t+1].to(device)   # [1, 36] long
                is_non_empty     = seq['inventory_has_items'][t]
                is_gui_t         = bool(is_gui_list[t])
                slot_type_texts  = seq.get('inventory_slot_texts',
                                           [['empty slot'] * 36] * len(seq['frames']))[t]
                slot_counts      = seq['inventory_counts'][t]

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

                if outputs.item_logits is not None:
                    item_logits  = outputs.item_logits[0]   # [36, item_vocab_size]
                    count_logits = outputs.count_logits[0]  # [36, count_classes]
                    tgt_ids      = inv_id_target[0]         # [36] long
                    tgt_counts   = inv_count_target[0]      # [36] long

                    pred_ids    = item_logits.argmax(dim=-1)   # [36]
                    pred_counts = count_logits.argmax(dim=-1)  # [36]

                    acc_all = (pred_ids == tgt_ids).float().mean().item()
                    total_item_acc += acc_all
                    n_total += 1

                    if is_gui_t:
                        gui_item_acc += acc_all
                        n_gui += 1
                    else:
                        closed_item_acc += acc_all
                        n_closed += 1

                    if is_non_empty:
                        ne_item_acc += acc_all
                        n_non_empty += 1
                        non_empty_mask = tgt_ids > 0
                        if non_empty_mask.any():
                            cnt_acc = (pred_counts[non_empty_mask] ==
                                       tgt_counts[non_empty_mask]).float().mean().item()
                            total_count_acc += cnt_acc
                            n_count += 1

                    # Collect display examples
                    bucket = None
                    if is_gui_t and is_non_empty and not any(e['bucket'] == 'gui' for e in display_examples):
                        bucket = 'gui'
                    elif not is_gui_t and is_non_empty and t >= seq_len // 3 and \
                            sum(1 for e in display_examples if e['bucket'] == 'closed') < 4:
                        bucket = 'closed'
                    if bucket is not None:
                        display_examples.append({
                            'source': seq.get('source_file', '?'),
                            'tick': t,
                            'is_gui': is_gui_t,
                            'bucket': bucket,
                            'item_acc': acc_all,
                            'slots': [
                                (i, slot_type_texts[i],
                                 int(slot_counts[i].item()),
                                 int(pred_ids[i].item()),
                                 int(pred_counts[i].item()),
                                 int(tgt_ids[i].item()))
                                for i in range(len(slot_type_texts))
                                if slot_type_texts[i] != "empty slot"
                            ],
                        })

                memory = outputs.new_memory.detach()

    avg_item_acc   = total_item_acc  / max(n_total,     1)
    avg_ne_acc     = ne_item_acc     / max(n_non_empty, 1)
    avg_gui_acc    = gui_item_acc    / max(n_gui,       1)
    avg_closed_acc = closed_item_acc / max(n_closed,    1)
    avg_count_acc  = total_count_acc / max(n_count,     1)
    non_empty_pct  = 100.0 * n_non_empty / max(n_total, 1)

    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION  ({num_seqs} seqs, {n_total} frames)")
    print(f"{'='*70}")
    print(f"  Item top-1 acc (all frames)      : {avg_item_acc:.4f}")
    print(f"  Item top-1 acc (non-empty slots) : {avg_ne_acc:.4f}"
          f"  ({n_non_empty}/{n_total} = {non_empty_pct:.1f}%)")
    print(f"  Item top-1 acc (GUI open)        : {avg_gui_acc:.4f}  ({n_gui} frames)")
    print(f"  Item top-1 acc (GUI closed/mem)  : {avg_closed_acc:.4f}  ({n_closed} frames)  ← key metric")
    print(f"  GUI-open vs closed gap           : {avg_gui_acc - avg_closed_acc:+.4f}"
          f"  (should shrink as memory improves)")
    print(f"  Count top-1 acc (non-empty slots): {avg_count_acc:.4f}")

    if display_examples:
        print(f"\n--- Slot predictions (up to 5 non-empty frames) ---")
        for ex in display_examples:
            gt_items   = {txt for _, txt, _, _, _, _ in ex['slots']}
            pred_items: set = set()
            gui_label = "[GUI OPEN]" if ex['is_gui'] else "[MEMORY]"

            print(f"\n  [{ex['source']}  tick {ex['tick']}  {gui_label}]  item_acc={ex['item_acc']:.3f}")
            for slot_idx, type_text, true_cnt, pred_item_id, pred_cnt_class, tgt_id in ex['slots']:
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
    return {
        'eval_item_acc':           avg_item_acc,
        'eval_non_empty_item_acc': avg_ne_acc,
        'eval_item_acc_gui':       avg_gui_acc,
        'eval_item_acc_closed':    avg_closed_acc,
        'eval_gui_closed_gap':     avg_gui_acc - avg_closed_acc,
        'eval_count_acc':          avg_count_acc,
        'eval_non_empty_pct':      non_empty_pct,
        'eval_n_frames':           n_total,
        'eval_n_gui_frames':       n_gui,
        'eval_n_closed_frames':    n_closed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',       required=True, help='Path to .pt checkpoint')
    parser.add_argument('--data_dir',         default='/data/vvm33/vpt_contractor')
    parser.add_argument('--train_jsonls',     type=int, default=4000,
                        help='Number of train files (test files start after these)')
    parser.add_argument('--test_jsonls',      type=int, default=5)
    parser.add_argument('--sequence_length',  type=int, default=200,
                        help='Frames per eval sequence')
    parser.add_argument('--memory_dim',       type=int, default=1024)
    parser.add_argument('--output_json',      default=None,
                        help='Optional path to save eval metrics as JSON')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Item vocab size: {get_item_vocab_size()}")

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
        inventory_head_kwargs={},
    )
    model = model.to(device, dtype=torch.bfloat16)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Checkpoint step: {ckpt.get('step', '?')}")

    print("Loading test data...")
    encoder = InventoryClassEncoder(device=str(device))
    jsonl_files = sorted(Path(args.data_dir).glob("*.jsonl"))
    test_files  = jsonl_files[args.train_jsonls : args.train_jsonls + args.test_jsonls]
    test_loader = OnDemandSequenceLoader(test_files, args.sequence_length, encoder)
    print(f"  Test sequences: {len(test_loader)}")

    metrics = evaluate(model, test_loader, processor, args, device)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics saved to {args.output_json}")


if __name__ == '__main__':
    main()
