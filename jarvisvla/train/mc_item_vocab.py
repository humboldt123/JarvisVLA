"""
Minecraft 1.16.5 item vocabulary for discrete classification.

Provides stable integer IDs for all ~975 Minecraft 1.16.5 items.
ID 0 is reserved for empty / air slots.
IDs 1..ITEM_VOCAB_SIZE-1 correspond to minecraft-data items sorted by their
canonical numeric ID (same order across all runs).

The alias table maps legacy or variant names that appear in VPT contractor
data to their canonical 1.16.5 names.  All 81 item names seen in 200 sampled
VPT files are already present in minecraft-data 1.16.5, so the alias table
is minimal and provided for safety only.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Alias table: old_name -> canonical_name_in_mc_data_1.16.5
# ---------------------------------------------------------------------------
# None of the VPT contractor files we've sampled need aliases, but having
# this table lets future data or modded content route to the closest vanilla
# item rather than silently mapping to UNKNOWN.
CANONICAL_ALIASES: Dict[str, str] = {
    # 1.16 renames
    "zombie_pigman":          "zombified_piglin",
    "grass":                  "grass_block",
    # British / underscore variants seen in some modpacks
    "grey_wool":              "gray_wool",
    "grey_dye":               "gray_dye",
    "grey_terracotta":        "gray_terracotta",
    "grey_concrete":          "gray_concrete",
    "grey_concrete_powder":   "gray_concrete_powder",
    "grey_stained_glass":     "gray_stained_glass",
    "grey_stained_glass_pane":"gray_stained_glass_pane",
    "grey_banner":            "gray_banner",
    "grey_bed":               "gray_bed",
    "grey_carpet":            "gray_carpet",
    "grey_glazed_terracotta": "gray_glazed_terracotta",
    "grey_shulker_box":       "gray_shulker_box",
    # numeric-prefix legacy names
    "log":                    "oak_log",
    "log2":                   "acacia_log",
    "planks":                 "oak_planks",
    "sapling":                "oak_sapling",
    "leaves":                 "oak_leaves",
    "leaves2":                "acacia_leaves",
    "stone_stairs":           "cobblestone_stairs",
}

# ---------------------------------------------------------------------------
# UNKNOWN / empty sentinel
# ---------------------------------------------------------------------------
UNKNOWN_ITEM_ID = 0  # used for empty slots and unrecognised item names
COUNT_CLASSES = 128  # count classes: 0..127  (vanilla max is 64; 127 covers glitched stacks)


def _load_vocab() -> Dict[str, int]:
    """
    Load the minecraft-data 1.16.5 item list and assign stable integer IDs.

    Returns:
        name2id: dict mapping item name (str) to integer ID (1-indexed).
    """
    try:
        import minecraft_data
        mc = minecraft_data("1.16.5")
        items: List[Dict] = mc.items_list
        # Sort by canonical numeric ID for stability
        items_sorted = sorted(items, key=lambda x: x["id"])
        name2id: Dict[str, int] = {}
        for rank, item in enumerate(items_sorted, start=1):
            name2id[item["name"]] = rank
        return name2id
    except Exception as e:
        raise RuntimeError(
            f"Failed to load minecraft-data 1.16.5 item vocabulary: {e}\n"
            "Install with: pip install minecraft-data"
        ) from e


# Module-level singleton — loaded once on first import.
_NAME2ID: Optional[Dict[str, int]] = None
_ID2NAME: Optional[Dict[int, str]] = None
ITEM_VOCAB_SIZE: int = 0   # set after first load


def get_vocab() -> Dict[str, int]:
    """Return name -> ID mapping (loaded lazily once)."""
    global _NAME2ID, _ID2NAME, ITEM_VOCAB_SIZE
    if _NAME2ID is None:
        _NAME2ID = _load_vocab()
        _ID2NAME = {v: k for k, v in _NAME2ID.items()}
        ITEM_VOCAB_SIZE = len(_NAME2ID) + 1   # +1 for UNKNOWN_ITEM_ID=0
    return _NAME2ID


def item_name_to_id(name: str) -> int:
    """
    Map an item name string to its integer ID.

    Args:
        name: Raw item name from JSONL (may include "minecraft:" prefix,
              may use a legacy alias).

    Returns:
        Integer ID in [0, ITEM_VOCAB_SIZE).  Returns UNKNOWN_ITEM_ID (0)
        if the name is empty, 'air', or unrecognised.
    """
    if not name or name in ("air", ""):
        return UNKNOWN_ITEM_ID

    # Strip namespace prefix
    if ":" in name:
        name = name.split(":", 1)[1]

    name = name.lower().strip()

    # Apply alias table
    name = CANONICAL_ALIASES.get(name, name)

    vocab = get_vocab()
    return vocab.get(name, UNKNOWN_ITEM_ID)


def item_id_to_name(item_id: int) -> str:
    """
    Map an integer ID back to item name.

    Returns 'empty' for ID 0, or 'unknown_<id>' for out-of-range IDs.
    """
    if item_id == UNKNOWN_ITEM_ID:
        return "empty"
    get_vocab()   # ensure _ID2NAME is populated
    return _ID2NAME.get(item_id, f"unknown_{item_id}")


def clamp_count(count: int) -> int:
    """Clamp a raw item count to [0, COUNT_CLASSES-1]."""
    return max(0, min(int(count), COUNT_CLASSES - 1))


def get_item_vocab_size() -> int:
    """Return total vocabulary size (including UNKNOWN_ITEM_ID=0)."""
    get_vocab()
    return ITEM_VOCAB_SIZE


def get_all_item_names() -> List[str]:
    """Return list of all item names, indexed by their ID."""
    vocab = get_vocab()
    get_vocab()  # ensure _ID2NAME is set
    names = ["empty"]   # index 0
    for i in range(1, ITEM_VOCAB_SIZE):
        names.append(_ID2NAME.get(i, f"unknown_{i}"))
    return names
