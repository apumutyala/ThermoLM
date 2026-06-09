"""
Unit tests for quantization layer.

Tests various quantization methods for hybrid models.

Author: Apuroop Mutyala
Date: April 15, 2026
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models.quantization import (
    QuantizationLayer,
    QuantizationConfig,
    StraightThroughQuantization,
    VectorQuantization,
    FSQQuantization,
    LearnedQuantization,
)


def test_quantization_config():
    """Test quantization configuration."""
    config = QuantizationConfig(
        d_latent=32,
        n_levels=4,
        quantization_type="fsq",
        commitment_cost=0.1,
    )
    
    assert config.d_latent == 32
    assert config.n_levels == 4
    assert config.quantization_type == "fsq"


def test_straight_through_quantization():
    """Test straight-through quantization."""
    config = QuantizationConfig(
        d_latent=16,
        n_levels=4,
        quantization_type="straight_through",
    )
    
    quantizer = StraightThroughQuantization(config)
    
    # Create dummy latents
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    
    # Initialize quantizer
    params = quantizer.init(key, latents)
    
    # Quantize
    quantized, codes, info = quantizer.apply(params, latents)
    
    assert quantized.shape == (2, 10, 16)
    assert codes.shape == (2, 10, 16)
    assert codes.dtype == jnp.int32
    assert 'commitment_loss' in info


def test_vector_quantization():
    """Test vector quantization."""
    config = QuantizationConfig(
        d_latent=16,
        n_levels=8,
        quantization_type="vq",
        commitment_cost=0.25,
    )
    
    quantizer = VectorQuantization(config)
    
    # Create dummy latents
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    
    # Initialize quantizer
    params = quantizer.init(key, latents)
    
    # Quantize
    quantized, codes, info = quantizer.apply(params, latents)
    
    assert quantized.shape == (2, 10, 16)
    assert codes.shape == (2, 10)  # VQ returns codebook indices
    assert codes.dtype == jnp.int32
    assert 'codebook_loss' in info
    assert 'commitment_loss' in info


def test_fsq_quantization():
    """Test FSQ quantization."""
    config = QuantizationConfig(
        d_latent=16,
        n_levels=4,
        quantization_type="fsq",
    )
    
    quantizer = FSQQuantization(config)
    
    # Create dummy latents
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    
    # Initialize quantizer
    params = quantizer.init(key, latents)
    
    # Quantize
    quantized, codes, info = quantizer.apply(params, latents)
    
    assert quantized.shape == (2, 10, 16)
    assert codes.shape == (2, 10, 16)
    assert codes.dtype == jnp.int32
    assert 'commitment_loss' in info


def test_learned_quantization():
    """Test learned quantization."""
    config = QuantizationConfig(
        d_latent=16,
        n_levels=4,
        quantization_type="learned",
    )
    
    quantizer = LearnedQuantization(config)
    
    # Create dummy latents
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    
    # Initialize quantizer
    params = quantizer.init(key, latents)
    
    # Quantize
    quantized, codes, info = quantizer.apply(params, latents)
    
    assert quantized.shape == (2, 10, 16)
    assert codes.shape == (2, 10, 16)
    assert codes.dtype == jnp.int32
    assert 'commitment_loss' in info


def test_quantization_layer():
    """Test unified quantization layer."""
    config = QuantizationConfig(
        d_latent=16,
        n_levels=4,
        quantization_type="fsq",
    )
    
    quantizer = QuantizationLayer(config)
    
    # Create dummy latents
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    
    # Initialize quantizer
    params = quantizer.init(key, latents)
    
    # Quantize
    quantized, codes, info = quantizer.apply(params, latents)
    
    assert quantized.shape == (2, 10, 16)
    assert codes.shape == (2, 10, 16)
    assert 'quantization_type' in info


def test_quantization_code_range():
    """Test that quantized codes are in valid range."""
    config = QuantizationConfig(
        d_latent=16,
        n_levels=4,
        quantization_type="fsq",
    )
    
    quantizer = FSQQuantization(config)
    
    # Create dummy latents
    key = jax.random.PRNGKey(42)
    latents = jax.random.normal(key, shape=(2, 10, 16))
    
    # Initialize quantizer
    params = quantizer.init(key, latents)
    
    # Quantize
    quantized, codes, info = quantizer.apply(params, latents)
    
    # Check code range
    assert jnp.all(codes >= 0)
    assert jnp.all(codes < config.n_levels)
