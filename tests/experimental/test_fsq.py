"""
Unit tests for FSQ encoder.

Tests Finite Scalar Quantization encoder implementation.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models.fsq import FSQEncoder, FSQDecoder, FSQConfig


def test_fsq_config():
    """Test FSQ configuration."""
    config = FSQConfig(
        vocab_size=1000,
        d_model=256,
        d_latent=32,
        n_levels=4,
    )
    
    assert config.vocab_size == 1000
    assert config.d_model == 256
    assert config.d_latent == 32
    assert config.n_levels == 4


def test_fsq_encoder_init():
    """Test FSQ encoder initialization."""
    config = FSQConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
    )
    
    encoder = FSQEncoder(config)
    
    # Check that levels are initialized correctly
    # For n_levels=4, should be [-1.5, -0.5, 0.5, 1.5]
    assert encoder.levels.shape == (4,)
    assert jnp.allclose(encoder.levels, jnp.array([-1.5, -0.5, 0.5, 1.5]))


def test_fsq_encode():
    """Test FSQ encoding."""
    config = FSQConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
    )
    
    encoder = FSQEncoder(config)
    
    # Create dummy tokens
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, shape=(4, 10), minval=0, maxval=100)
    
    # Initialize encoder
    params = encoder.init(key, tokens)
    
    # Encode
    codes, latents = encoder.apply(params, tokens, method=encoder.encode)
    
    assert codes.shape == (4, 10, 16)
    assert latents.shape == (4, 10, 16)
    assert codes.dtype == jnp.int32
    
    # Codes should be in valid range [0, n_levels-1]
    assert jnp.all(codes >= 0)
    assert jnp.all(codes < 4)


def test_fsq_decode():
    """Test FSQ decoding."""
    config = FSQConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
    )
    
    encoder = FSQEncoder(config)
    
    # Create dummy codes
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(4, 10, 16), minval=0, maxval=4)
    
    # Initialize encoder
    tokens = jax.random.randint(key, shape=(4, 10), minval=0, maxval=100)
    params = encoder.init(key, tokens)
    
    # Decode
    embeddings = encoder.apply(params, codes, method=encoder.decode)
    
    assert embeddings.shape == (4, 10, 128)


def test_fsq_roundtrip():
    """Test FSQ encode-decode roundtrip."""
    config = FSQConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
    )
    
    encoder = FSQEncoder(config)
    
    # Create dummy tokens
    key = jax.random.PRNGKey(42)
    tokens = jax.random.randint(key, (4, 10), 0, 100)
    
    # Initialize encoder
    params = encoder.init(key, tokens)
    
    # Encode
    codes, latents = encoder.apply(params, tokens, method=encoder.encode)
    
    # Decode
    recon_embeddings = encoder.apply(params, codes, method=encoder.decode)
    
    assert recon_embeddings.shape == (4, 10, 128)


def test_fsq_quantize():
    """Test FSQ quantization."""
    config = FSQConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
    )
    
    encoder = FSQEncoder(config)
    
    # Create dummy latents
    key = jax.random.PRNGKey(42)
    latents = jax.random.uniform(key, shape=(4, 10, 16), minval=-2.0, maxval=2.0)
    
    # Initialize encoder
    tokens = jax.random.randint(key, (4, 10), 0, 100)
    params = encoder.init(key, tokens)
    
    # Quantize
    codes = encoder.apply(params, latents, method=encoder.quantize)
    
    assert codes.shape == (4, 10, 16)
    assert codes.dtype == jnp.int32
    assert jnp.all(codes >= 0)
    assert jnp.all(codes < 4)


def test_fsq_dequantize():
    """Test FSQ dequantization."""
    config = FSQConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
    )
    
    encoder = FSQEncoder(config)
    
    # Create dummy codes
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(4, 10, 16), minval=0, maxval=4)
    
    # Initialize encoder
    tokens = jax.random.randint(key, shape=(4, 10), minval=0, maxval=100)
    params = encoder.init(key, tokens)
    
    # Dequantize
    latents = encoder.apply(params, codes, method=encoder.dequantize)
    
    assert latents.shape == (4, 10, 16)
    
    # Dequantized values should be close to original levels
    # For n_levels=4, levels are [-1.5, -0.5, 0.5, 1.5]
    valid_values = jnp.array([-1.5, -0.5, 0.5, 1.5])
    # Check that each value is close to one of the valid levels
    for i in range(latents.shape[0]):
        for j in range(latents.shape[1]):
            for k in range(latents.shape[2]):
                assert jnp.any(jnp.isclose(latents[i, j, k], valid_values, atol=0.1))


def test_fsq_decoder():
    """Test standalone FSQ decoder."""
    config = FSQConfig(
        vocab_size=100,
        d_model=128,
        d_latent=16,
        n_levels=4,
    )
    
    decoder = FSQDecoder(config)
    
    # Create dummy codes
    key = jax.random.PRNGKey(42)
    codes = jax.random.randint(key, shape=(4, 10, 16), minval=0, maxval=4)
    
    # Initialize decoder
    params = decoder.init(key, codes)
    
    # Decode
    embeddings = decoder.apply(params, codes)
    
    assert embeddings.shape == (4, 10, 128)
