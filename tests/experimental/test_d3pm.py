"""
Unit tests for D3PM Parameterization.

Tests d3pm forward process and loss computation.
"""

import pytest
import jax
import jax.numpy as jnp

from thermolm_jax.models import q_xt, d3pm_parameterization, d3pm_loss, sample_prior


@pytest.mark.unit
def test_q_xt():
    """Test forward process q(x_t|x_0)."""
    key = jax.random.PRNGKey(42)
    x = jnp.array([[0, 1, 2], [3, 4, 5]])
    move_chance = 0.5
    mask_index = 50257
    
    xt = q_xt(x, move_chance, mask_index=mask_index, key=key)
    
    assert xt.shape == x.shape
    # Check that values are either from original x or mask_index
    valid_values = jnp.isin(xt, x) | (xt == mask_index)
    assert jnp.all(valid_values)


@pytest.mark.unit
def test_d3pm_parameterization():
    """Test d3pm parameterization."""
    logits = jnp.ones((2, 10, 50258))  # (batch, seq_len, vocab_size)
    mask_index = 50257
    
    logits_param = d3pm_parameterization(logits, mask_index)
    
    assert logits_param.shape == logits.shape
    assert logits_param[:, :, mask_index].min() < -1e8  # Mask token should have very low log prob


@pytest.mark.unit
def test_sample_prior():
    """Test prior sampling."""
    batch_size = 4
    seq_len = 128
    mask_index = 50257
    
    prior = sample_prior(batch_size, seq_len, mask_index)
    
    assert prior.shape == (batch_size, seq_len)
    assert jnp.all(prior == mask_index)


@pytest.mark.unit
def test_d3pm_loss():
    """Test d3pm loss computation."""
    batch_size = 2
    seq_len = 10
    vocab_size = 50257
    mask_index = 50257
    T = 1000
    
    model_output = jnp.ones((batch_size, seq_len, vocab_size))
    xt = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    x0 = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    t = jnp.array([[0.5], [0.5]])  # Shape (batch_size, 1) for proper broadcasting
    
    loss = d3pm_loss(model_output, xt, x0, t, mask_index, T)
    
    assert loss.shape == (batch_size, seq_len)


# TODO: Add more unit tests
# TODO: Test with different timesteps
# TODO: Test loss gradients
# TODO: Test edge cases
