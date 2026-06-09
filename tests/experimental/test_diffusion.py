"""
Unit tests for Diffusion Schedule module.

Tests individual components of the diffusion schedule in isolation.
"""

import pytest
import jax.numpy as jnp
from thermolm_jax.models.diffusion_schedule import (
    cosine_schedule,
    linear_schedule,
    sigmoid_schedule,
    compute_alpha_bar,
    q_sample,
    DiffusionSchedule,
)


@pytest.mark.unit
def test_cosine_schedule():
    """Test cosine schedule."""
    T = 1000
    betas = cosine_schedule(T)
    
    assert betas.shape == (T,)
    assert jnp.all(betas >= 0)
    assert jnp.all(betas <= 1)


@pytest.mark.unit
def test_linear_schedule():
    """Test linear schedule."""
    T = 1000
    betas = linear_schedule(T)
    
    assert betas.shape == (T,)
    assert jnp.all(betas >= 0)
    assert jnp.all(betas <= 1)


@pytest.mark.unit
def test_sigmoid_schedule():
    """Test sigmoid schedule."""
    T = 1000
    betas = sigmoid_schedule(T)
    
    assert betas.shape == (T,)
    assert jnp.all(betas >= 0)
    assert jnp.all(betas <= 1)


@pytest.mark.unit
def test_compute_alpha_bar(diffusion_config):
    """Test cumulative alpha computation."""
    betas = cosine_schedule(diffusion_config['num_timesteps'])
    alpha_bar = compute_alpha_bar(betas, 500)
    
    assert isinstance(alpha_bar, (float, jnp.ndarray))
    assert alpha_bar >= 0
    assert alpha_bar <= 1


@pytest.mark.unit
def test_q_sample(sample_embeddings, diffusion_config):
    """Test forward diffusion sampling."""
    betas = cosine_schedule(diffusion_config['num_timesteps'])
    key = jax.random.PRNGKey(42)
    
    x_t, noise = q_sample(sample_embeddings, 500, betas, key)
    
    assert x_t.shape == sample_embeddings.shape
    assert noise.shape == sample_embeddings.shape


@pytest.mark.unit
def test_diffusion_schedule_class(diffusion_config):
    """Test DiffusionSchedule class."""
    schedule = DiffusionSchedule(
        schedule_type='cosine',
        T=diffusion_config['num_timesteps'],
    )
    
    assert schedule.T == diffusion_config['num_timesteps']
    assert schedule.betas.shape == (diffusion_config['num_timesteps'],)


# TODO: Add more unit tests
# TODO: Test different schedule types
# TODO: Test schedule properties (monotonicity, etc.)
