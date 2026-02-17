'''
Author: Muyao 2350076251@qq.com
Date: 2025-03-04 23:31:28
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-05 03:47:11
'''
import random
import torch
import os
import numpy as np
import pathlib
from dataclasses import dataclass, field
from typing import Optional

IGNORE_TOKEN_ID = -100

@dataclass
class MoreConfig:
    dataset_p: float = field(default=1.0, metadata={"help": "Dataset parameter p"})
    collator_type: str = field(default="MultimodalChatDataCollatorforVLM", metadata={"help": "types of collator"})
    fix_visual_encoder: bool = field(default=False, metadata={"help": "fix visual encoder"})
    fix_visual_adapter: bool = field(default=False, metadata={"help": "fix visual adapter layer"})
    fix_language_backbone: bool = field(default=False, metadata={"help": "fix language backbone"})
    fix_lm_head: bool = field(default=False, metadata={"help": "fix language model head"})
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=2048*28*28)
    
    # Inventory prediction head configuration
    use_inventory_head: bool = field(default=False, metadata={"help": "Enable auxiliary inventory prediction head"})
    inventory_num_slots: int = field(default=36, metadata={"help": "Number of inventory slots (9 hotbar + 27 main)"})
    inventory_num_item_classes: int = field(default=256, metadata={"help": "Number of item classes in vocabulary"})
    inventory_hidden_dim: Optional[int] = field(default=None, metadata={"help": "Hidden dimension for inventory head (default: backbone_hidden_dim // 2)"})
    inventory_predict_count: bool = field(default=False, metadata={"help": "Also predict item counts"})
    inventory_max_count: int = field(default=64, metadata={"help": "Maximum item count (Minecraft stack size)"})
    inventory_dropout: float = field(default=0.1, metadata={"help": "Dropout rate for inventory head"})
    inventory_loss_weight: float = field(default=0.1, metadata={"help": "Weight for inventory auxiliary loss"})
    inventory_empty_slot_idx: int = field(default=0, metadata={"help": "Index representing empty slot"})
    inventory_use_partial_credit: bool = field(default=True, metadata={"help": "Use partial credit for similar items"})
    inventory_vocab_path: Optional[str] = field(default=None, metadata={"help": "Path to item vocabulary JSON file"})

def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model:torch.nn.Module,optimizer:torch.optim.Optimizer=None,record_path = None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    model_shapes = []
    for name, parameter in model.named_parameters():
        if optimizer:
            optimizer_group_idx = None
            for idx,param_group in enumerate(optimizer.param_groups):
                for param in param_group["params"]:
                    if parameter is param:
                        optimizer_group_idx = idx
            model_shapes.append([parameter.requires_grad,name,parameter.shape,optimizer_group_idx])
        else:
            model_shapes.append([parameter.requires_grad,name,parameter.shape])
        all_param += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    import json
    if record_path:
        pathlib.Path(record_path).parent.mkdir(parents=True,exist_ok=True)
        with open(record_path,mode="w",encoding="UTF-8") as f:
            json.dump(model_shapes, f, indent=4)
        
        with open(record_path.replace(".json","-scratch.txt"),mode="w",encoding="UTF-8") as f:
            print(optimizer, file=f)
            print(model, file=f)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
