"""
Default Configuration Module for ThermoLM JAX

Provides default configuration for the entire project.

Author: Apuroop Mutyala
Date: April 2026
"""

from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for ThermoLM JAX.
    
    Returns:
        config: Default configuration dictionary
    """
    config = {
        'model': {
            'vocab_size': 50257,
            'd_model': 512,
            'd_latent': 64,
            'max_seq_len': 128,
            'num_energy_layers': 6,
            'num_energy_heads': 8,
        },
        'diffusion': {
            'num_timesteps': 1000,
            'beta_start': 0.0001,
            'beta_end': 0.9999,
            'schedule': 'cosine',
        },
        'tsu': {
            'max_energy_per_edge': 1000,
            'max_degree': 4,
            'block_size': 32,
            'num_blocks': 4,
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'num_epochs': 50,
            'gradient_clip': 1.0,
        },
        'data': {
            'dataset': 'wikitext-2-raw-v1',
            'max_length': 128,
            'stride': 64,
        },
    }
    
    return config
