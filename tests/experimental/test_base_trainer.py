"""
Unit tests for base trainer.

Tests the abstract trainer interface.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.config import BaseConfig
from thermolm_jax.training import BaseTrainer


class MockConfig(BaseConfig):
    """Mock config for testing."""
    learning_rate: float = 1e-4
    batch_size: int = 32


class MockTrainer(BaseTrainer):
    """Mock trainer for testing."""
    
    def train_step(self, params, opt_state, batch, key):
        # Mock training step
        loss = jnp.array(1.0)
        metrics = {'loss': loss}
        return params, opt_state, loss, metrics, key
    
    def validation_step(self, params, batch):
        # Mock validation step
        return {'val_loss': jnp.array(0.5)}
    
    def train(self, train_data, valid_data=None, num_epochs=10):
        # Mock training loop
        return {}, {}


def test_base_trainer_interface():
    """Test that base trainer interface is correctly defined."""
    config = MockConfig()
    trainer = MockTrainer(config)
    
    # Test that trainer has config
    assert trainer.config == config
    
    # Test train_step
    params = {'param': jnp.zeros(10)}
    opt_state = None
    batch = jnp.zeros((32, 128), dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    
    params_out, opt_state_out, loss, metrics, key_out = trainer.train_step(
        params, opt_state, batch, key
    )
    
    assert loss.shape == ()
    assert 'loss' in metrics
    
    # Test validation_step
    val_metrics = trainer.validation_step(params, batch)
    assert 'val_loss' in val_metrics
    
    # Test train
    train_data = jnp.zeros((10, 32, 128), dtype=jnp.int32)
    trained_params, history = trainer.train(train_data)
    assert isinstance(trained_params, dict)
    assert isinstance(history, dict)


def test_base_trainer_abstract():
    """Test that base trainer cannot be instantiated directly."""
    config = MockConfig()
    
    with pytest.raises(TypeError):
        BaseTrainer(config)


def test_get_checkpoint():
    """Test checkpoint creation."""
    config = MockConfig()
    trainer = MockTrainer(config)
    
    params = {'param': jnp.zeros(10)}
    step = 100
    
    checkpoint = trainer.get_checkpoint(params, step)
    
    assert 'params' in checkpoint
    assert 'step' in checkpoint
    assert 'config' in checkpoint
    assert checkpoint['step'] == 100
