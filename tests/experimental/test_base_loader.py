"""
Unit tests for base data loader.

Tests the abstract interface and implementation requirements.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import jax.numpy as jnp
from thermolm_jax.data import BaseDataLoader, WikiTextDatasetJAX


class MockDataLoader(BaseDataLoader):
    """Mock implementation for testing."""
    
    def __init__(self, split='train', max_length=128, cache_dir=None, **kwargs):
        self.split = split
        self.max_length = max_length
        self.examples = [jnp.zeros(max_length, dtype=jnp.int32) for _ in range(10)]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_vocab_size(self):
        return 50257
    
    def get_special_tokens(self):
        return {'pad_token_id': 0, 'eos_token_id': 50256}


def test_base_loader_interface():
    """Test that base loader interface is correctly defined."""
    loader = MockDataLoader()
    
    # Test basic methods
    assert len(loader) == 10
    assert loader.get_vocab_size() == 50257
    assert loader.get_special_tokens()['eos_token_id'] == 50256
    
    # Test get_stats
    stats = loader.get_stats()
    assert stats['num_examples'] == 10
    assert stats['max_length'] == 128
    assert stats['vocab_size'] == 50257


def test_wikitext_loader_inherits_base():
    """Test that WikiText loader inherits from base loader."""
    # This test requires actual dataset loading, skip for now
    # TODO: Enable when dataset is available
    pytest.skip("Requires dataset download")


def test_base_loader_abstract():
    """Test that base loader cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseDataLoader()
