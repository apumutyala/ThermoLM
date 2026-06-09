"""
Unit tests for base configuration.

Tests the base config class and serialization.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import json
import tempfile
import os
from dataclasses import dataclass
from thermolm_jax.config import BaseConfig


@dataclass
class MockConfig(BaseConfig):
    """Mock config for testing."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_length: int = 128


def test_base_config_serialization():
    """Test config serialization to dict."""
    config = MockConfig()
    
    config_dict = config.to_dict()
    assert config_dict['learning_rate'] == 1e-4
    assert config_dict['batch_size'] == 32
    assert config_dict['max_length'] == 128


def test_base_config_json():
    """Test config serialization to JSON."""
    config = MockConfig()
    
    json_str = config.to_json()
    assert 'learning_rate' in json_str
    assert '1e-4' in json_str or '0.0001' in json_str


def test_base_config_from_dict():
    """Test config creation from dict."""
    config_dict = {'learning_rate': 2e-4, 'batch_size': 64}
    config = MockConfig.from_dict(config_dict)
    
    assert config.learning_rate == 2e-4
    assert config.batch_size == 64


def test_base_config_from_json():
    """Test config creation from JSON."""
    json_str = '{"learning_rate": 2e-4, "batch_size": 64, "max_length": 256}'
    config = MockConfig.from_json(json_str)
    
    assert config.learning_rate == 2e-4
    assert config.batch_size == 64
    assert config.max_length == 256


def test_base_config_save_load():
    """Test config save and load."""
    config = MockConfig()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save(temp_path)
        loaded_config = MockConfig.load(temp_path)
        
        assert loaded_config.learning_rate == config.learning_rate
        assert loaded_config.batch_size == config.batch_size
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_base_config_str():
    """Test config string representation."""
    config = MockConfig()
    config_str = str(config)
    
    assert 'learning_rate' in config_str
