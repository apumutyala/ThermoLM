"""
TSU Configuration Module for ThermoLM JAX

Provides TSU-specific configuration for hardware optimization.

Author: Apuroop Mutyala
Date: April 2026
"""

from dataclasses import dataclass


@dataclass
class TSUConfig:
    """Configuration for TSU hardware constraints."""
    
    max_energy_per_edge: float = 1000.0
    max_degree: int = 4
    block_size: int = 32
    num_blocks: int = 4
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'max_energy_per_edge': self.max_energy_per_edge,
            'max_degree': self.max_degree,
            'block_size': self.block_size,
            'num_blocks': self.num_blocks,
        }
