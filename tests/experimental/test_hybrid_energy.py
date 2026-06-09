"""Unit tests for hybrid energy function."""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models.hybrid_energy import (
    HybridEnergyFunction,
    HybridEnergyLoss,
    HybridEnergyConfig,
)


def test_hybrid_energy_config():
    config = HybridEnergyConfig(vocab_size=1000, d_model=256, d_latent=32, num_energy_layers=4, continuous_weight=0.7, discrete_weight=0.3)
    assert config.vocab_size == 1000
    assert config.continuous_weight == 0.7


def test_hybrid_energy_function_init():
    config = HybridEnergyConfig(vocab_size=100, d_model=128, d_latent=16, num_energy_layers=2)
    energy_fn = HybridEnergyFunction(config)
    assert energy_fn.config.n_levels == 8


def test_hybrid_energy_continuous_only():
    config = HybridEnergyConfig(vocab_size=100, d_model=128, d_latent=16, num_energy_layers=2, continuous_weight=1.0, discrete_weight=0.0)
    energy_fn = HybridEnergyFunction(config)
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    mask = jnp.ones((2, 10))
    params = energy_fn.init(key, latents=latents, mask=mask)
    energy, info = energy_fn.apply(params, latents=latents, mask=mask)
    assert energy.shape == (2,)
    assert 'continuous_energy' in info


def test_hybrid_energy_discrete_only():
    config = HybridEnergyConfig(vocab_size=100, d_model=128, d_latent=16, num_energy_layers=2, continuous_weight=0.0, discrete_weight=1.0)
    energy_fn = HybridEnergyFunction(config)
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(2, 10, 16), minval=0, maxval=8)
    mask = jnp.ones((2, 10))
    params = energy_fn.init(key, codes=codes, mask=mask)
    energy, info = energy_fn.apply(params, codes=codes, mask=mask)
    assert energy.shape == (2,)
    assert 'discrete_energy' in info


def test_hybrid_energy_both():
    config = HybridEnergyConfig(vocab_size=100, d_model=128, d_latent=16, num_energy_layers=2, continuous_weight=0.5, discrete_weight=0.5)
    energy_fn = HybridEnergyFunction(config)
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    codes = jax.random.randint(key, shape=(2, 10, 16), minval=0, maxval=8)
    mask = jnp.ones((2, 10))
    params = energy_fn.init(key, latents=latents, codes=codes, mask=mask)
    energy, info = energy_fn.apply(params, latents=latents, codes=codes, mask=mask)
    assert energy.shape == (2,)
    assert 'continuous_energy' in info
    assert 'discrete_energy' in info


def test_hybrid_energy_loss():
    config = HybridEnergyConfig(vocab_size=100, d_model=128, d_latent=16, num_energy_layers=2)
    loss_fn = HybridEnergyLoss(config)
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    codes = jax.random.randint(key, shape=(2, 10, 16), minval=0, maxval=8)
    mask = jnp.ones((2, 10))
    quantization_info = {'commitment_loss': 0.1}
    # Initialize with quantization_info
    params = loss_fn.init(key, latents=latents, codes=codes, quantization_info=quantization_info, mask=mask)
    loss, info = loss_fn.apply(params, latents, codes, quantization_info, mask, key)
    assert loss.shape == ()
    assert 'total_loss' in info
