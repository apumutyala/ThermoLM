"""
Integration tests for Training module.

Tests training workflows and infrastructure.
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.training import train_step, create_optimizer


@pytest.mark.integration
def test_optimizer_creation():
    """Test optimizer creation."""
    optimizer = create_optimizer(
        learning_rate=0.0001,
        weight_decay=0.01,
    )
    
    assert optimizer is not None


@pytest.mark.integration
def test_train_step(model_config, sample_batch):
    """Test single training step."""
    # TODO: Implement when train_step is fully implemented
    pytest.skip("train_step not yet fully implemented")


# TODO: Add more integration tests
# TODO: Test full training loop
# TODO: Test checkpointing
# TODO: Test distributed training
