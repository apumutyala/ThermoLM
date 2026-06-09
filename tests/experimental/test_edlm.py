"""
Integration tests for EDLM module.

Tests end-to-end EDLM workflows.
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.models import EDLM


@pytest.mark.integration
def test_edlm_forward_pass(model_config, sample_batch):
    """Test EDLM forward pass."""
    model = EDLM(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        d_latent=model_config['d_latent'],
        num_energy_layers=2,  # Small for testing
    )
    
    key = jax.random.PRNGKey(42)
    outputs = model(sample_batch, t=500, key=key)
    
    assert 'embeddings' in outputs
    assert 'energy' in outputs
    assert 'logits' in outputs
    assert outputs['embeddings'].shape == (4, 128, 512)
    assert outputs['logits'].shape == (4, 128, 50257)


@pytest.mark.integration
def test_edlm_initialization(model_config):
    """Test EDLM initialization."""
    model = EDLM(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        d_latent=model_config['d_latent'],
    )
    
    assert model.vocab_size == model_config['vocab_size']
    assert model.d_model == model_config['d_model']


# TODO: Add more integration tests
# TODO: Test full pipeline (data → model → loss)
# TODO: Test training step
# TODO: Test generation
