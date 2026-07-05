"""
Training package for ThermoLM JAX.

Validated: contrastive-divergence training of quadratic Ising EBMs.
"""

from .contrastive_divergence import (
    contrastive_divergence_loss,
    contrastive_divergence_step,
    CDConfig,
)

__all__ = [
    "contrastive_divergence_loss",
    "contrastive_divergence_step",
    "CDConfig",
]
