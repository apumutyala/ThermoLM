"""
Unit tests for EMA.

Tests exponential moving average implementation.
"""

import pytest
import jax
import jax.numpy as jnp

from thermolm_jax.training import ExponentialMovingAverage, apply_ema


@pytest.mark.unit
def test_ema_initialization():
    """Test EMA initialization."""
    ema = ExponentialMovingAverage(decay=0.9999)
    
    assert ema.decay == 0.9999
    assert ema.shadow_params is None


@pytest.mark.unit
def test_ema_update():
    """Test EMA parameter update."""
    ema = ExponentialMovingAverage(decay=0.9)
    
    params = {'w': jnp.array([1.0, 2.0, 3.0])}
    ema.update(params)
    
    assert ema.shadow_params is not None
    assert ema.count == 1


@pytest.mark.unit
def test_ema_copy_to():
    """Test EMA copy to parameters."""
    ema = ExponentialMovingAverage(decay=0.9)
    
    params = {'w': jnp.array([1.0, 2.0, 3.0])}
    ema.update(params)
    
    copied = ema.copy_to(params)
    
    # Check that copied params match shadow params
    assert jnp.allclose(copied['w'], ema.shadow_params['w'])


@pytest.mark.unit
def test_ema_store_restore():
    """Test EMA store and restore."""
    ema = ExponentialMovingAverage(decay=0.9)
    
    params1 = {'w': jnp.array([1.0, 2.0, 3.0])}
    params2 = {'w': jnp.array([4.0, 5.0, 6.0])}
    
    ema.update(params1)
    ema.store(params2)
    restored = ema.restore(params2)
    
    assert jnp.allclose(restored['w'], params2['w'])


@pytest.mark.unit
def test_apply_ema():
    """Test apply_ema function."""
    params = {'w': jnp.array([1.0, 2.0, 3.0])}
    ema_params = {'w': jnp.array([0.0, 0.0, 0.0])}
    decay = 0.9
    
    updated = apply_ema(params, ema_params, decay)
    
    assert updated['w'].shape == params['w'].shape


# TODO: Add more unit tests
# TODO: Test with nested parameter structures
# TODO: Test state_dict save/load
