"""
Unit tests for Data module.

Tests data loading and tokenization in isolation.
"""

import pytest
import jax.numpy as jnp
from thermolm_jax.data import WikiTextDatasetJAX, create_jax_dataloaders


@pytest.mark.unit
def test_wikitext_dataset_initialization():
    """Test WikiText dataset initialization."""
    dataset = WikiTextDatasetJAX(
        split='train',
        max_length=128,
        stride=64,
    )
    
    assert dataset.max_length == 128
    assert dataset.stride == 64
    assert len(dataset) > 0


@pytest.mark.unit
def test_wikitext_dataset_getitem():
    """Test WikiText dataset item retrieval."""
    dataset = WikiTextDatasetJAX(
        split='train',
        max_length=128,
        stride=64,
    )
    
    example = dataset[0]
    
    assert example.shape == (128,)
    assert example.dtype == jnp.int32


@pytest.mark.unit
def test_create_jax_dataloaders():
    """Test JAX dataloader creation."""
    loaders = create_jax_dataloaders(
        batch_size=4,
        max_length=128,
        stride=64,
    )
    
    assert 'train' in loaders
    assert 'valid' in loaders
    assert loaders['train'].shape[0] > 0
    assert loaders['valid'].shape[0] > 0


# TODO: Add more unit tests
# TODO: Test tokenization
# TODO: Test batching
# TODO: Test edge cases (empty sequences, etc.)
