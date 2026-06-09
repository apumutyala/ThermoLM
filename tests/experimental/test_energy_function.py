"""
Unit tests for Energy Function module.

Tests individual components of the energy function in isolation.
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models.energy_function import EnergyFunctionJAX, sparse_energy_function


@pytest.mark.unit
def test_energy_function_initialization(model_config):
    """Test energy function initialization."""
    energy_fn = EnergyFunctionJAX(
        d_model=model_config['d_model'],
        d_latent=model_config['d_latent'],
        num_layers=2,  # Small for testing
        num_heads=model_config['num_heads'],
    )
    
    assert energy_fn.d_model == model_config['d_model']
    assert energy_fn.d_latent == model_config['d_latent']


@pytest.mark.unit
def test_energy_function_forward(sample_embeddings):
    """Test energy function forward pass."""
    energy_fn = EnergyFunctionJAX(
        d_model=512,
        d_latent=64,
        num_layers=2,
        num_heads=8,
    )
    
    # Test with same embeddings (should work)
    energy = energy_fn(sample_embeddings, sample_embeddings)
    
    assert energy.shape == (4,)  # batch size
    assert energy.dtype == jnp.float32


@pytest.mark.unit
def test_sparse_energy_function(sample_embeddings):
    """Test sparse energy function."""
    seq_len, d_latent = 128, 64
    z_prev = sample_embeddings[0, :seq_len, :d_latent]
    z_curr = sample_embeddings[1, :seq_len, :d_latent]
    
    params = {
        'W_self': jnp.zeros((seq_len, d_latent)),
        'W_pair': jnp.zeros((seq_len, seq_len, d_latent)),
    }
    
    energy = sparse_energy_function(z_prev, z_curr, params)
    
    assert isinstance(energy, (float, jnp.ndarray))


# TODO: Add more unit tests
# TODO: Test energy gradient computation
# TODO: Test energy function with different configurations
