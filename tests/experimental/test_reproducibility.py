"""
Unit tests for reproducibility utilities.

Tests random seed management.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.utils.reproducibility import set_seed, get_rng_key, split_rng_key, fork_rng_key


def test_set_seed():
    """Test setting random seed."""
    set_seed(42)
    
    # Test that we can generate consistent results
    key1 = get_rng_key(42)
    key2 = get_rng_key(42)
    
    # Keys should be identical
    assert jnp.array_equal(key1, key2)


def test_get_rng_key():
    """Test getting PRNG key."""
    key = get_rng_key(42)
    
    assert key.shape == (2,)
    assert key.dtype == jnp.uint32


def test_split_rng_key():
    """Test splitting PRNG key."""
    key = get_rng_key(42)
    keys = split_rng_key(key, 3)
    
    assert len(keys) == 3
    assert all(k.shape == (2,) for k in keys)
    assert all(k.dtype == jnp.uint32 for k in keys)


def test_fork_rng_key():
    """Test forking PRNG key."""
    key = get_rng_key(42)
    key1, key2 = fork_rng_key(key)
    
    assert key1.shape == (2,)
    assert key2.shape == (2,)
    assert key1.dtype == jnp.uint32
    assert key2.dtype == jnp.uint32
    
    # Keys should be different
    assert not jnp.array_equal(key1, key2)


def test_reproducibility():
    """Test that same seed produces same results."""
    set_seed(42)
    key1 = get_rng_key(42)
    random1 = jax.random.uniform(key1, shape=(10,))
    
    set_seed(42)
    key2 = get_rng_key(42)
    random2 = jax.random.uniform(key2, shape=(10,))
    
    assert jnp.allclose(random1, random2)
