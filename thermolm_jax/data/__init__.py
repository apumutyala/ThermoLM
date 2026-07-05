"""
Data module for ThermoLM JAX.

Validated: char-level tokenizer and windowing for the Tier-1 LM.
"""

from .char_tokenizer import CharTokenizer, make_windows

__all__ = [
    "CharTokenizer",
    "make_windows",
]
