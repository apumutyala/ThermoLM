"""
Integration tests for EDLM.

Tests end-to-end EDLM workflows with new components.
"""

import pytest
import jax
import jax.numpy as jnp

from thermolm_jax.models import EDLM


@pytest.mark.integration
def test_edlm_initialization(model_config):
    """Test EDLM initialization."""
    model = EDLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['d_model'],
        n_blocks=2,  # Small for testing
    )
    
    assert model.vocab_size == model_config['vocab_size']
    assert model.hidden_size == model_config['d_model']


@pytest.mark.integration
def test_edlm_forward_pass(model_config, sample_batch):
    """Test EDLM forward pass."""
    model = EDLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['d_model'],
        n_blocks=2,
    )
    
    model.set_mask_index(model_config['vocab_size'])
    
    key = jax.random.PRNGKey(42)
    sigma = jnp.array([0.5, 0.5])
    
    logits = model(sample_batch, sigma)
    
    assert logits.shape == (4, 128, model_config['vocab_size'])


@pytest.mark.integration
def test_edlm_compute_loss(model_config, sample_batch):
    """Test EDLM loss computation."""
    model = EDLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['d_model'],
        n_blocks=2,
    )
    
    model.set_mask_index(model_config['vocab_size'])
    
    key = jax.random.PRNGKey(42)
    
    loss_dict = model.compute_loss(sample_batch, key)
    
    assert 'loss' in loss_dict
    assert 'nll' in loss_dict
    assert 'token_mask' in loss_dict
    assert loss_dict['loss'].shape == ()


# TODO: Add more integration tests
# TODO: Test full training step
# TODO: Test EMA integration
# TODO: Test with real data
