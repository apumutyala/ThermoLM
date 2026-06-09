"""
Pytest configuration for ThermoLM JAX tests

Provides fixtures and configuration for comprehensive testing.
"""

import pytest
import jax
import jax.numpy as jnp
from typing import Dict, Any


@pytest.fixture(scope="session")
def jax_config():
    """
    Configure JAX for testing.
    
    Returns:
        JAX configuration dictionary
    """
    # Use CPU for consistent testing
    jax.config.update('jax_platform_name', 'cpu')
    
    # Disable JIT for faster test iteration in development
    # Comment out for production tests with JIT enabled
    jax.config.update('jax_disable_jit', True)
    
    return {
        'platform': 'cpu',
        'jit_disabled': True,
    }


@pytest.fixture
def rng_key():
    """
    Provide a random number generator key for JAX.
    
    Returns:
        JAX PRNG key
    """
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_batch(rng_key):
    """
    Provide a sample batch of token IDs for testing.
    
    Args:
        rng_key: PRNG key
    
    Returns:
        Sample batch of shape (batch_size, seq_len)
    """
    batch_size = 4
    seq_len = 128
    vocab_size = 50257  # GPT-2 vocab size
    
    batch = jax.random.randint(
        rng_key, 
        (batch_size, seq_len), 
        0, 
        vocab_size
    )
    
    return batch


@pytest.fixture
def sample_embeddings(rng_key):
    """
    Provide sample embeddings for testing.
    
    Args:
        rng_key: PRNG key
    
    Returns:
        Sample embeddings of shape (batch_size, seq_len, d_model)
    """
    batch_size = 4
    seq_len = 128
    d_model = 512
    
    embeddings = jax.random.normal(
        rng_key,
        (batch_size, seq_len, d_model)
    )
    
    return embeddings


@pytest.fixture
def model_config():
    """
    Provide default model configuration for testing.
    
    Returns:
        Model configuration dictionary
    """
    return {
        'd_model': 512,
        'd_latent': 64,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_len': 128,
        'vocab_size': 50257,
    }


@pytest.fixture
def diffusion_config():
    """
    Provide default diffusion configuration for testing.
    
    Returns:
        Diffusion configuration dictionary
    """
    return {
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.9999,
        'schedule': 'cosine',
    }


@pytest.fixture
def training_config():
    """
    Provide default training configuration for testing.
    
    Returns:
        Training configuration dictionary
    """
    return {
        'batch_size': 32,
        'learning_rate': 0.0001,
        'weight_decay': 0.01,
        'num_epochs': 50,
        'gradient_clip': 1.0,
    }


@pytest.fixture
def tsu_config():
    """
    Provide default TSU configuration for testing.
    
    Returns:
        TSU configuration dictionary
    """
    return {
        'max_energy_per_edge': 1000,
        'max_degree': 4,
        'block_size': 32,
        'num_blocks': 4,
    }


def pytest_configure(config):
    """
    Pytest configuration hook.
    
    Add custom markers and configuration.
    """
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
