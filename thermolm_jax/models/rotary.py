"""
Rotary Positional Embeddings (RoPE) for ThermoLM JAX

Implements rotary positional embeddings as used in NVIDIA's DiT.
Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding".

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple


def apply_rotary_pos_emb(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """
    Apply rotary positional embeddings to input.
    
    Args:
        x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim // 2)
        sin: Sine embeddings of shape (seq_len, head_dim // 2)
    
    Returns:
        Rotated tensor of same shape as x
    """
    # Split into two halves
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    
    # Apply rotation
    x_rotated = jnp.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)
    
    return x_rotated


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate half of the dimensions.
    
    Args:
        x: Input tensor
    
    Returns:
        Rotated tensor
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)


class RotaryEmbedding:
    """
    Rotary positional embedding layer.
    
    Attributes:
        dim: Dimension of the embedding
        base: Base for frequency computation
        seq_len_cached: Cached sequence length
        cos_cached: Cached cosine embeddings
        sin_cached: Cached sine embeddings
    """
    
    def __init__(self, dim: int, base: float = 10000):
        """
        Initialize rotary embedding.
        
        Args:
            dim: Dimension of the embedding
            base: Base for frequency computation (default: 10000)
        """
        self.dim = dim
        self.base = base
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    
    def __call__(self, x: jnp.ndarray, seq_dim: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute rotary embeddings for input.

        Args:
            x: Input tensor
            seq_dim: Sequence dimension

        Returns:
            Tuple of (cos, sin) embeddings
        """
        seq_len = x.shape[seq_dim]

        # Compute if sequence length changed (Python-level cache, only valid outside JIT)
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            self.cos_cached, self.sin_cached = self._compute_embeddings(seq_len)

        return self.cos_cached, self.sin_cached

    def _compute_embeddings(self, seq_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute cosine and sine embeddings.

        Args:
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) embeddings
        """
        # Compute inverse frequencies — use .astype() not .float() (JAX, not PyTorch)
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))

        # Compute position frequencies
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)

        # Compute embeddings — use axis= not dim= (JAX, not PyTorch)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos, sin


# TODO: Add support for variable sequence lengths - Implemented below
# TODO: Add support for multi-dimensional rotary embeddings - Implemented below
# TODO: Test rotary embeddings with attention


class MultiDimensionalRotaryEmbedding(nn.Module):
    """
    Multi-dimensional rotary positional embedding.
    
    Applies rotary embeddings across multiple dimensions for richer positional encoding.
    """
    
    d_model: int
    num_rotary_dims: int = 2
    max_seq_len: int = 8192
    
    def setup(self):
        """Initialize multi-dimensional rotary embedding components."""
        # Compute frequencies for each rotary dimension
        self.freqs = []
        for i in range(self.num_rotary_dims):
            freq = self.compute_frequencies(self.d_model // self.num_rotary_dims, self.max_seq_len)
            self.freqs.append(freq)
    
    def compute_frequencies(self, dim: int, max_len: int) -> jnp.ndarray:
        """Compute rotary frequencies."""
        freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
        freq = jnp.outer(jnp.arange(max_len), freq)
        return freq
    
    def __call__(self, x: jnp.ndarray, seq_len: Optional[int] = None) -> jnp.ndarray:
        """
        Apply multi-dimensional rotary embeddings.
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            seq_len: Optional sequence length (for variable-length sequences)
        
        Returns:
            x: Rotated embeddings
        """
        batch_size, current_seq_len, d_model = x.shape
        
        if seq_len is None:
            seq_len = current_seq_len
        
        # Split into multiple rotary dimensions
        dim_per_rotary = d_model // self.num_rotary_dims
        
        rotated_parts = []
        for i in range(self.num_rotary_dims):
            # Get slice for this dimension
            start_idx = i * dim_per_rotary
            end_idx = (i + 1) * dim_per_rotary
            x_part = x[:, :, start_idx:end_idx]
            
            # Apply rotary embedding
            freq = self.freqs[i][:seq_len]
            cos = jnp.cos(freq)
            sin = jnp.sin(freq)
            
            # Reshape for broadcasting
            x_part = x_part.reshape(batch_size, seq_len, dim_per_rotary // 2, 2)
            
            # Apply rotation
            x_rotated = x_part * cos[None, :, None, None] + jnp.stack([-x_part[..., 1], x_part[..., 0]], axis=-1) * sin[None, :, None, None]
            x_rotated = x_rotated.reshape(batch_size, seq_len, dim_per_rotary)
            
            rotated_parts.append(x_rotated)
        
        # Concatenate all rotated parts
        x_rotated = jnp.concatenate(rotated_parts, axis=-1)
        
        return x_rotated


def apply_rotary_variable_length(
    rotary: RotaryEmbedding,
    x: jnp.ndarray,
    lengths: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply rotary embeddings for variable-length sequences.
    
    Args:
        rotary: Rotary embedding module
        x: Input embeddings (batch, max_len, d_model)
        lengths: Actual lengths (batch,)
    
    Returns:
        x: Rotated embeddings
    """
    batch_size, max_len, d_model = x.shape
    
    # Apply rotary to each sequence individually
    rotated_sequences = []
    for i in range(batch_size):
        seq_len = lengths[i]
        x_seq = x[i:i+1, :seq_len, :]
        x_rotated = rotary(x_seq, seq_len=seq_len)
        
        # Pad back to max_len
        padding = jnp.zeros((1, max_len - seq_len, d_model))
        x_padded = jnp.concatenate([x_rotated, padding], axis=1)
        rotated_sequences.append(x_padded)
    
    x_rotated = jnp.concatenate(rotated_sequences, axis=0)
    
    return x_rotated
