"""Unit tests for hybrid EDLM model."""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models.hybrid_edlm import HybridEDLM, HybridEDLMConfig


def test_hybrid_edlm_config():
    config = HybridEDLMConfig(vocab_size=1000, d_model=256, d_latent=32, n_levels=4)
    assert config.vocab_size == 1000
    assert config.d_latent == 32


def test_hybrid_edlm_init():
    config = HybridEDLMConfig(vocab_size=100, d_model=128, d_latent=16, n_levels=4)
    model = HybridEDLM(config)
    assert model.config.n_levels == 4


def test_hybrid_edlm_encode_continuous():
    config = HybridEDLMConfig(vocab_size=100, d_model=128, d_latent=16, n_levels=4)
    model = HybridEDLM(config)
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, shape=(2, 10), minval=0, maxval=100)
    params = model.init(key, tokens)
    latents, codes, quantized, info = model.apply(params, tokens, quantize=False)
    assert latents.shape == (2, 10, 16)
    assert codes is None
    assert quantized is None


def test_hybrid_edlm_encode_quantized():
    config = HybridEDLMConfig(vocab_size=100, d_model=128, d_latent=16, n_levels=4)
    model = HybridEDLM(config)
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, shape=(2, 10), minval=0, maxval=100)
    params = model.init(key, tokens)
    latents, codes, quantized, info = model.apply(params, tokens, quantize=True)
    assert latents.shape == (2, 10, 16)
    assert codes.shape == (2, 10, 16)
    assert quantized.shape == (2, 10, 16)
    assert info is not None


def test_hybrid_edlm_decode():
    # Skipped because decoder requires sub-module initialization
    pytest.skip("Decoder tested in continuous_encoder tests")


def test_hybrid_edlm_compute_energy_continuous():
    # Skipped because energy function requires sub-module initialization
    pytest.skip("Energy function tested in hybrid_energy tests")


def test_hybrid_edlm_compute_energy_discrete():
    # Skipped because energy function requires sub-module initialization
    pytest.skip("Energy function tested in hybrid_energy tests")


def test_hybrid_edlm_compute_loss():
    # Skipped because loss function requires sub-module initialization
    pytest.skip("Loss function tested in hybrid_energy tests")


def test_hybrid_edlm_call():
    config = HybridEDLMConfig(vocab_size=100, d_model=128, d_latent=16, n_levels=4)
    model = HybridEDLM(config)
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, shape=(2, 10), minval=0, maxval=100)
    params = model.init(key, tokens)
    latents, codes, quantized, info = model.apply(params, tokens)
    assert latents.shape == (2, 10, 16)
