"""
Models package for ThermoLM JAX.

Validated DTM / quadratic-Ising components are exported here. The exploratory
energy-diffusion language-model modules (energy_function, diffusion_schedule,
sampler, edlm, rotary, dit_block, d3pm, adaln, timestep, fsq, discrete_energy,
thrml_discrete, discrete_edlm, continuous_encoder, quantization, hybrid_*,
factor_weight_network, binary_autoencoder) are intentionally NOT imported here
because they are unvalidated (see STATUS.md). Import them explicitly by path if
you want to experiment, e.g. ``from thermolm_jax.models.fsq import FSQEncoder``.
"""

from .quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from .connectivity import generate_connectivity_pattern, get_connectivity_density
from .forward_coupling import ForwardCoupling, ForwardCouplingConfig
from .latent_graph import SparseGraph, LatentGraphConfig
from .dtm import DTM, DTMConfig
from .thrml_quadratic import THRMLQuadraticEBM

__all__ = [
    "QuadraticEBM",
    "QuadraticEBMConfig",
    "generate_connectivity_pattern",
    "get_connectivity_density",
    "ForwardCoupling",
    "ForwardCouplingConfig",
    "SparseGraph",
    "LatentGraphConfig",
    "DTM",
    "DTMConfig",
    "THRMLQuadraticEBM",
]
