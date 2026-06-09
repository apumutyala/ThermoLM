"""
Unit tests for Sampler module.

Tests individual components of the THRML sampler in isolation.
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models.sampler import THRMLSamplerJAX, metropolis_hastings


@pytest.mark.unit
def test_sampler_initialization():
    """Test sampler initialization."""
    sampler = THRMLSamplerJAX(
        d_latent=64,
        num_steps=10,
        temperature=1.0,
    )
    
    assert sampler.d_latent == 64
    assert sampler.num_steps == 10
    assert sampler.temperature == 1.0


@pytest.mark.unit
def test_sampler_block_creation():
    """Test block creation for parallel sampling."""
    sampler = THRMLSamplerJAX(d_latent=64)
    
    # Test without adjacency (default alternating)
    blocks = sampler._create_blocks(10, None)
    
    assert len(blocks) == 2  # Even and odd
    assert len(blocks[0]) + len(blocks[1]) == 10


@pytest.mark.unit
def test_graph_coloring():
    """Test graph coloring algorithm."""
    sampler = THRMLSamplerJAX(d_latent=64)
    
    # Create simple adjacency matrix
    adjacency = jnp.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    
    blocks = sampler._graph_coloring(adjacency)
    
    # Should color with 2 colors for a path graph
    assert len(blocks) == 2


@pytest.mark.unit
def test_metropolis_hastings():
    """Test Metropolis-Hastings acceptance."""
    key = jax.random.PRNGKey(42)
    z = jax.random.normal(key, (10, 64))
    
    def mock_energy(z):
        return jnp.sum(z ** 2)
    
    z_new = metropolis_hastings(z, 0, mock_energy, 1.0, key)
    
    assert z_new.shape == z.shape


# TODO: Add more unit tests
# TODO: Test Gibbs sampling steps
# TODO: Test energy function integration
