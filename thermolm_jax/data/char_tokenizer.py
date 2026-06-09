"""
Char-level tokenizer for the small-vocab chain-CRF language model.

A character vocabulary keeps V tiny (~65-100), which is what makes the chain-CRF
pairwise tensor (V x V per edge) cheap and TSU-plausible. Build the vocab from a
corpus, then encode/decode and cut fixed-length windows.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import jax.numpy as jnp


@dataclass
class CharTokenizer:
    """Maps characters <-> integer ids built from a corpus."""

    itos: List[str]
    stoi: dict

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        itos = sorted(set(text))
        stoi = {c: i for i, c in enumerate(itos)}
        return cls(itos=itos, stoi=stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text if c in self.stoi], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def make_windows(ids: np.ndarray, seq_len: int, stride: int) -> jnp.ndarray:
    """Cut a long id stream into overlapping (n, seq_len) windows."""
    starts = range(0, max(len(ids) - seq_len, 0) + 1, stride)
    windows = [ids[s : s + seq_len] for s in starts if s + seq_len <= len(ids)]
    if not windows:
        raise ValueError(f"Corpus too short ({len(ids)}) for seq_len={seq_len}.")
    return jnp.asarray(np.stack(windows))
