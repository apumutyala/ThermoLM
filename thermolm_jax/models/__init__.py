"""
Models package for ThermoLM JAX.

Validated core: quadratic-Ising EBM and chain-CRF diffusion LM.
See README.md and STATUS.md for scope.
"""

from .quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from .connectivity import generate_connectivity_pattern, get_connectivity_density
from .thrml_quadratic import THRMLQuadraticEBM

__all__ = [
    "QuadraticEBM",
    "QuadraticEBMConfig",
    "generate_connectivity_pattern",
    "get_connectivity_density",
    "THRMLQuadraticEBM",
]
