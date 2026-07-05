"""
Diffusion Configuration Module for ThermoLM JAX

Provides diffusion-specific configuration.

Author: Apuroop Mutyala
Date: April 2026
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DiffusionConfig:
    """Configuration for diffusion schedule."""
    
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.9999
    schedule: Literal['cosine', 'linear', 'sigmoid'] = 'cosine'
    sigmoid_scale: float = 6.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'num_timesteps': self.num_timesteps,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'schedule': self.schedule,
            'sigmoid_scale': self.sigmoid_scale,
        }
