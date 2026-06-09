"""
Unit tests for Rotary Embeddings.

Tests rotary positional embeddings implementation.
"""

import pytest
import jax
import jax.numpy as jnp

from thermolm_jax.models import RotaryEmbedding, apply_rotary_pos_emb


@pytest.mark.unit
def test_rotary_embedding_initialization():
    """Test rotary embedding initialization."""
    rotary = RotaryEmbedding(dim=64, base=10000)
    
    assert rotary.dim == 64
    assert rotary.base == 10000


@pytest.mark.unit
def test_rotary_embedding_computation():
    """Test rotary embedding computation."""
    rotary = RotaryEmbedding(dim=64, base=10000)
    
    x = jnp.ones((2, 10, 8, 32))  # (batch, seq_len, n_heads, head_dim)
    cos, sin = rotary(x)
    
    assert cos.shape == (10, 32)
    assert sin.shape == (10, 32)


@pytest.mark.unit
def test_apply_rotary_pos_emb():
    """Test rotary position embedding application."""
    x = jnp.ones((2, 10, 8, 32))  # (batch, seq_len, n_heads, head_dim)
    cos = jnp.ones((10, 16))  # (seq_len, head_dim // 2)
    sin = jnp.ones((10, 16))
    
    x_rotated = apply_rotary_pos_emb(x, cos, sin)
    
    assert x_rotated.shape == x.shape
    assert not jnp.allclose(x_rotated, x)


# TODO: Add more unit tests
# TODO: Test with different sequence lengths
# TODO: Test with different base values
