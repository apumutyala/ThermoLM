"""
Model Configuration Module for ThermoLM JAX

Provides model-specific configuration.

Author: Apuroop Mutyala
Date: April 2026
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for EDLM model."""
    
    vocab_size: int = 50257
    d_model: int = 512
    d_latent: int = 64
    max_seq_len: int = 128
    num_energy_layers: int = 6
    num_energy_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'd_latent': self.d_latent,
            'max_seq_len': self.max_seq_len,
            'num_energy_layers': self.num_energy_layers,
            'num_energy_heads': self.num_energy_heads,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
        }
