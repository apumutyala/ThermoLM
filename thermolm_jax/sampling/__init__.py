"""
Sampling utilities for ThermoLM JAX.

Chromatic block Gibbs for Ising EBMs (JAX and THRML paths) and the THRML
chain-MRF sampler used as the diffusion LM's TSU-compatible reverse step.
"""

from .chromatic_gibbs import (
    chromatic_gibbs_sample,
    greedy_coloring,
    color_masks_from_colors,
)
from .chain_mrf_thrml import sample_chain_thrml, sample_chain_thrml_single

__all__ = [
    "chromatic_gibbs_sample",
    "greedy_coloring",
    "color_masks_from_colors",
    "sample_chain_thrml",
    "sample_chain_thrml_single",
]
