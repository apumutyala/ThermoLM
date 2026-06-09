"""
Data module for ThermoLM JAX

Contains JAX-compatible data loaders and tokenizers.
"""

from .base_loader import BaseDataLoader
from .wikitext_jax import WikiTextDatasetJAX, create_jax_dataloaders
from .tokenizers import TokenizerManager
from .preprocessing import (
    pad_sequence,
    truncate_sequence,
    create_sliding_windows,
    batch_sequences,
    mask_sequence,
)

__all__ = [
    "BaseDataLoader",
    "WikiTextDatasetJAX",
    "create_jax_dataloaders",
    "TokenizerManager",
    "pad_sequence",
    "truncate_sequence",
    "create_sliding_windows",
    "batch_sequences",
    "mask_sequence",
]
