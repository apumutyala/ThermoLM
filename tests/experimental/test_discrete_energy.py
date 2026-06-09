"""
Unit tests for discrete energy function.

Tests discrete energy function implementation.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models.discrete_energy import (
    DiscreteEnergyFunction,
    DiscreteEnergyLoss,
    DiscreteEnergyConfig,
)


def test_discrete_energy_config():
    """Test discrete energy configuration."""
    config = DiscreteEnergyConfig(
        vocab_size=1000,
        d_model=256,
        d_latent=32,
        n_levels=4,
        num_energy_layers=2,
        num_energy_heads=4,
        max_seq_len=64,
    )
    
    assert config.vocab_size == 1000
    assert config.d_model == 256
    assert config.d_latent == 32
    assert config.n_levels == 4
    assert config.num_energy_layers == 2
    assert config.num_energy_heads == 4


def test_discrete_energy_function_init():
    """Test discrete energy function initialization."""
    config = DiscreteEnergyConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
        num_energy_layers=2,
        num_energy_heads=4,
        max_seq_len=32,
    )
    
    energy_fn = DiscreteEnergyFunction(config)
    assert energy_fn.config.n_levels == 4


def test_discrete_energy_forward():
    """Test discrete energy function forward pass."""
    config = DiscreteEnergyConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
        num_energy_layers=2,
        num_energy_heads=4,
        max_seq_len=32,
    )
    
    energy_fn = DiscreteEnergyFunction(config)
    
    # Create dummy codes
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(4, 10, 16), minval=0, maxval=4)
    
    # Initialize energy function
    mask = jnp.ones((4, 10))
    params = energy_fn.init(key, codes, mask=mask)
    
    # Compute energy
    energy = energy_fn.apply(params, codes, mask=mask)
    
    assert energy.shape == (4,)
    assert energy.dtype == jnp.float32


def test_discrete_energy_no_mask():
    """Test discrete energy function without mask."""
    config = DiscreteEnergyConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
        num_energy_layers=2,
        num_energy_heads=4,
        max_seq_len=32,
    )
    
    energy_fn = DiscreteEnergyFunction(config)
    
    # Create dummy codes
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(4, 10, 16), minval=0, maxval=4)
    
    # Initialize energy function
    params = energy_fn.init(key, codes)
    
    # Compute energy without mask
    energy = energy_fn.apply(params, codes)
    
    assert energy.shape == (4,)


def test_discrete_energy_loss():
    """Test discrete energy loss computation."""
    config = DiscreteEnergyConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
        num_energy_layers=2,
        num_energy_heads=4,
        max_seq_len=32,
    )
    
    loss_fn = DiscreteEnergyLoss(config)
    
    # Create dummy codes
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(4, 10, 16), minval=0, maxval=4)
    mask = jnp.ones((4, 10))
    
    # Initialize loss function
    params = loss_fn.init(key, codes, mask=mask, key=key)
    
    # Compute loss
    loss, energy = loss_fn.apply(params, codes, mask=mask, key=key)
    
    assert loss.shape == ()
    assert energy.shape == ()
    assert loss.dtype == jnp.float32


def test_discrete_energy_loss_no_mask():
    """Test discrete energy loss without mask."""
    config = DiscreteEnergyConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
        num_energy_layers=2,
        num_energy_heads=4,
        max_seq_len=32,
    )
    
    loss_fn = DiscreteEnergyLoss(config)
    
    # Create dummy codes
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(4, 10, 16), minval=0, maxval=4)
    
    # Initialize loss function
    params = loss_fn.init(key, codes, key=key)
    
    # Compute loss without mask
    loss, energy = loss_fn.apply(params, codes, key=key)
    
    assert loss.shape == ()
    assert energy.shape == ()
