"""
Unit tests for preprocessing utilities.

Tests padding, truncation, windowing, and batching functions.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.data.preprocessing import (
    pad_sequence,
    truncate_sequence,
    create_sliding_windows,
    batch_sequences,
    mask_sequence,
)


def test_pad_sequence():
    """Test sequence padding."""
    tokens = [1, 2, 3, 4, 5]
    max_length = 10
    pad_token_id = 0
    
    # Test padding
    padded = pad_sequence(tokens, max_length, pad_token_id)
    assert padded.shape == (10,)
    assert padded[0] == 1
    assert padded[4] == 5
    assert padded[5] == 0  # Padding
    assert padded[-1] == 0
    
    # Test with EOS
    eos_token_id = 50256
    padded_eos = pad_sequence(tokens, max_length, pad_token_id, eos_token_id)
    assert padded_eos[-1] == eos_token_id


def test_truncate_sequence():
    """Test sequence truncation."""
    tokens = list(range(20))
    max_length = 10
    
    # Test truncation
    truncated = truncate_sequence(tokens, max_length)
    assert len(truncated) == 10
    assert truncated[0] == 0
    assert truncated[9] == 9
    
    # Test with EOS
    eos_token_id = 50256
    truncated_eos = truncate_sequence(tokens, max_length, eos_token_id)
    assert truncated_eos[-1] == eos_token_id


def test_create_sliding_windows():
    """Test sliding window creation."""
    tokens = list(range(20))
    max_length = 10
    stride = 5
    
    windows = create_sliding_windows(tokens, max_length, stride)
    
    # Should create windows: [0-9], [5-14], [10-19], [15-19]
    assert len(windows) == 4
    assert len(windows[0]) == 10
    assert windows[0][0] == 0
    assert windows[1][0] == 5
    assert windows[2][0] == 10
    assert windows[3][0] == 15
    
    # Test with min_length
    windows_min = create_sliding_windows(tokens, max_length, stride, min_length=8)
    # Last window has only 5 elements, so it's filtered out
    assert len(windows_min) == 3


def test_batch_sequences():
    """Test sequence batching."""
    sequences = [
        jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32),
        jnp.array([6, 7, 8, 9, 10], dtype=jnp.int32),
        jnp.array([11, 12, 13, 14, 15], dtype=jnp.int32),
        jnp.array([16, 17, 18, 19, 20], dtype=jnp.int32),
    ]
    batch_size = 2
    
    batches = batch_sequences(sequences, batch_size)
    
    assert batches.shape == (2, 2, 5)
    assert batches[0, 0, 0] == 1
    assert batches[0, 1, 0] == 6
    assert batches[1, 0, 0] == 11
    assert batches[1, 1, 0] == 16
    
    # Test with drop_last=False
    sequences_odd = sequences + [jnp.array([21, 22, 23, 24, 25], dtype=jnp.int32)]
    batches_odd = batch_sequences(sequences_odd, batch_size, drop_last=False)
    assert batches_odd.shape == (3, 2, 5)


def test_mask_sequence():
    """Test sequence masking."""
    tokens = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    mask_token_id = 0
    mask_prob = 0.5
    key = jax.random.PRNGKey(42)
    
    masked, mask = mask_sequence(tokens, mask_token_id, mask_prob, key)
    
    assert masked.shape == tokens.shape
    assert mask.shape == tokens.shape
    assert jnp.issubdtype(masked.dtype, jnp.integer)
    assert mask.dtype == jnp.bool_
    
    # Some tokens should be masked
    assert jnp.any(mask)
