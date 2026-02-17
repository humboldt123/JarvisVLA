"""
Training module for JarvisVLA.
"""

from jarvisvla.train.auxiliary_heads import (
    InventoryEmbeddingHead,
    MemoryProjections,
    HiddenStateExtractor,
)

from jarvisvla.train.sequence_dataset import (
    InventoryTextEncoder,
    VPTSequenceDataset,
    VPTSequenceDataCollator,
)

from jarvisvla.train.stateful_trainer import (
    StatefulVLATrainer,
    SequenceDataCollator,
)

__all__ = [
    # Auxiliary heads and projections
    'InventoryEmbeddingHead',
    'MemoryProjections',
    'HiddenStateExtractor',
    # Sequence dataset
    'InventoryTextEncoder',
    'VPTSequenceDataset',
    'VPTSequenceDataCollator',
    # Trainer
    'StatefulVLATrainer',
    'SequenceDataCollator',
]
