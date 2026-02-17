'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:35:08
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-05-28 23:10:17
'''
import logging
import os
from contextlib import nullcontext
import pathlib

TRL_USE_RICH = os.getenv("TRL_USE_RICH", False)

from trl.scripts import init_zero_verbose, ScriptArguments, TrlParser
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    get_quantization_config,
    get_kbit_device_map,
)

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler
if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)
    
from datasets import load_dataset,Dataset

import torch
from tqdm.rich import tqdm
from transformers import Qwen2VLProcessor
from transformers import Qwen2VLForConditionalGeneration
from transformers import Trainer

import json
import re

from rich import print,console
from jarvisvla.train.utils_train import (
    print_trainable_parameters,
    seed_everything,
    MoreConfig,
)
from jarvisvla import QWEN_SPECIAL_TOKENS
from jarvisvla.train.data_collator import make_collator
from jarvisvla.train.trainer import VLAInventoryTrainer
from jarvisvla.train.heads import InventoryHead
from jarvisvla.train.item_vocabulary import ItemVocabulary
import torch

tqdm.pandas()    

if __name__ == "__main__":
    
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, MoreConfig))
    sft_script_args, training_args, model_config, more_cfg = parser.parse_args_and_config()

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    seed_everything(training_args.seed)

    ################
    # Model, Tokenizer & Processor
    ################
    
    model_name = model_config.model_name_or_path.lower().replace('-','_')
    
    ### discard: if no chat_template is defined in tokenizer_config.json, use the default one
    DEFAULT_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] + ':\n\n'+ message['content'] + '\n' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"""
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    if 'qwen2_vl' in model_name:
        processor_config = dict(
            do_rescale=False,
            patch_size=14,
            vision_feature_select_strategy="default"
        )
        processor = Qwen2VLProcessor.from_pretrained(model_config.model_name_or_path,**processor_config)
        with open(QWEN_SPECIAL_TOKENS, "r") as file:
            special_token = json.load(file)
        processor.tokenizer.add_special_tokens({"additional_special_tokens":special_token})
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        raise ValueError(f"{model_name} unknown")
    
    if not processor.tokenizer.chat_template:
        raise ValueError("No chat_template found in the tokenizer_config.json, please set the chat_template in the tokenizer_config.json.")
        
    processor.tokenizer.padding_side = "right"
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    fix_refexs = []
    if getattr(more_cfg, 'fix_visual_encoder', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"visual\.blocks.*")
            fix_refexs.append(r"visual\.patch_embed.*")
    if getattr(more_cfg, 'fix_visual_adapter', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"visual\.merger.*")
    if getattr(more_cfg, 'fix_language_backbone', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"model\.embed_tokens.*")
            fix_refexs.append(r"model\.layers.*")
    if getattr(more_cfg, 'fix_lm_head', False):
        if 'qwen2_vl' in model_name:
            fix_refexs.append(r"model\.norm.*")
            fix_refexs.append(r"lm_head.*")
       
    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in fix_refexs):
            param.requires_grad = False
    
    
    ##################
    # DataCollator
    ##################

    # Find image folder
    image_fold = pathlib.Path(sft_script_args.dataset_name).parent
    image_fold = image_fold.parent if image_fold.name=="output" else image_fold
    
    # Prepare collator kwargs
    collator_kwargs = dict(
        processor=processor, 
        model_path=model_name,
        image_folder=image_fold,
        max_seq_length = training_args.max_seq_length,
        min_pixels = more_cfg.min_pixels,
        max_pixels = more_cfg.max_pixels,
    )
    
    # Add inventory-specific args if using inventory collator
    if more_cfg.use_inventory_head and 'Inventory' in more_cfg.collator_type:
        collator_kwargs.update(dict(
            num_slots=more_cfg.inventory_num_slots,
            empty_slot_idx=more_cfg.inventory_empty_slot_idx,
            item_vocabulary=item_vocabulary,
        ))
    
    data_collator = make_collator(more_cfg.collator_type, **collator_kwargs)
    
    ################
    # Dataset
    ################
    
    raw_datasets = load_dataset(sft_script_args.dataset_name)
    
    train_dataset = raw_datasets['train']
    train_dataset_len = train_dataset.num_rows
    train_dataset_len = int(more_cfg.dataset_p*train_dataset_len)
    train_dataset = train_dataset.shuffle(training_args.seed)
    if train_dataset_len < 0:
        select_ids = range(train_dataset.num_rows + train_dataset_len,train_dataset.num_rows)
    else:
        select_ids = range(train_dataset_len)
    train_dataset = train_dataset.select(select_ids) 
    eval_dataset = raw_datasets['valid']
    
    if training_args.local_rank in { 0 ,-1 }:
        print(train_dataset_len,more_cfg.dataset_p,int(more_cfg.dataset_p*train_dataset_len))
    
    ################
    # Optional rich context managers
    ###############
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Inventory Head (if enabled)
    ################
    
    inventory_head = None
    item_vocabulary = None
    similarity_matrix = None
    
    if more_cfg.use_inventory_head:
        # Get the hidden dimension from the model config
        # For Qwen2VL, this is typically the hidden_size of the language model
        if hasattr(model.config, 'hidden_size'):
            hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
            hidden_dim = model.config.text_config.hidden_size
        else:
            # Fallback for Qwen2VL-7B
            hidden_dim = 3584
            if training_args.local_rank in {0, -1}:
                print(f"[yellow]Warning: Could not determine hidden_size from config, using default: {hidden_dim}[/yellow]")
        
        # Load item vocabulary if provided
        if more_cfg.inventory_vocab_path and pathlib.Path(more_cfg.inventory_vocab_path).exists():
            item_vocabulary = ItemVocabulary.load(more_cfg.inventory_vocab_path)
            num_item_classes = item_vocabulary.num_items
            
            # Load or compute similarity matrix
            sim_matrix_path = pathlib.Path(more_cfg.inventory_vocab_path).parent / 'similarity_matrix.pt'
            if sim_matrix_path.exists():
                similarity_matrix = torch.load(sim_matrix_path)
                if training_args.local_rank in {0, -1}:
                    print(f"[green]Loaded similarity matrix from {sim_matrix_path}[/green]")
            else:
                # Compute similarity matrix
                similarity_matrix = item_vocabulary.build_similarity_matrix()
                torch.save(similarity_matrix, sim_matrix_path)
                if training_args.local_rank in {0, -1}:
                    print(f"[green]Computed and saved similarity matrix to {sim_matrix_path}[/green]")
        else:
            num_item_classes = more_cfg.inventory_num_item_classes
            if training_args.local_rank in {0, -1}:
                print(f"[yellow]Warning: No vocabulary provided, using {num_item_classes} classes[/yellow]")
        
        # Create inventory head
        inventory_head = InventoryHead(
            input_dim=hidden_dim,
            num_slots=more_cfg.inventory_num_slots,
            num_item_classes=num_item_classes,
            hidden_dim=more_cfg.inventory_hidden_dim,
            predict_count=more_cfg.inventory_predict_count,
            max_count=more_cfg.inventory_max_count,
            dropout=more_cfg.inventory_dropout,
            use_partial_credit=more_cfg.inventory_use_partial_credit,
            similarity_matrix=similarity_matrix,
        )
        
        # Attach to model
        model.inventory_head = inventory_head
        
        if training_args.local_rank in {0, -1}:
            print(f"[green]Inventory head created:[/green]")
            print(f"  - Input dim: {hidden_dim}")
            print(f"  - Num slots: {more_cfg.inventory_num_slots}")
            print(f"  - Num item classes: {num_item_classes}")
            print(f"  - Loss weight: {more_cfg.inventory_loss_weight}")
            print(f"  - Partial credit: {more_cfg.inventory_use_partial_credit}")
    
    ################
    # Training
    ################
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True
        
    # Ensure use_cache is set to False
    model.config.use_cache = False 

    training_args.dataset_text_field = "text"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    
    # Use custom trainer if inventory head is enabled
    if more_cfg.use_inventory_head:
        trainer = VLAInventoryTrainer(
            model=model,
            args=training_args, 
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor.tokenizer,
            model_init=None,
            compute_metrics=None,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            preprocess_logits_for_metrics=None,
            inventory_head=inventory_head,
            inventory_loss_weight=more_cfg.inventory_loss_weight,
        )
    else:
        trainer = Trainer( 
            model=model,
            args=training_args, 
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor.tokenizer,
            model_init=None,
            compute_metrics=None,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            preprocess_logits_for_metrics=None,
        )
    
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print_trainable_parameters(trainer.model,trainer.optimizer,f"logs/model_structure.json")

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    elif not training_args.do_train and training_args.do_eval:
        trainer.evaluate()

    if training_args.save_strategy != "no":
        trainer.save_model(training_args.output_dir)
        
        # Also save inventory head separately for easy loading
        if more_cfg.use_inventory_head and inventory_head is not None:
            import json
            inventory_config_path = pathlib.Path(training_args.output_dir) / "inventory_config.json"
            inventory_head_path = pathlib.Path(training_args.output_dir) / "inventory_head.pt"
            
            # Save config
            with open(inventory_config_path, 'w') as f:
                json.dump({
                    'num_slots': more_cfg.inventory_num_slots,
                    'num_item_classes': more_cfg.inventory_num_item_classes,
                    'hidden_dim': more_cfg.inventory_hidden_dim,
                    'predict_count': more_cfg.inventory_predict_count,
                    'max_count': more_cfg.inventory_max_count,
                    'dropout': more_cfg.inventory_dropout,
                    'loss_weight': more_cfg.inventory_loss_weight,
                    'empty_slot_idx': more_cfg.inventory_empty_slot_idx,
                }, f, indent=2)
            
            # Save state dict
            torch.save(inventory_head.state_dict(), inventory_head_path)
            
            if training_args.local_rank in {0, -1}:
                print(f"[green]Inventory head saved to {inventory_head_path}[/green]")

