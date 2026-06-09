"""
ThermoLM JAX

Validated track (this is what the package exposes by default):
  Quadratic Ising energy-based models sampled with chromatic block Gibbs and
  trained by contrastive divergence, with a THRML-backed sampling path. See the
  README "Validated: DTM / quadratic-Ising" section.

Exploratory track (NOT exposed here; see STATUS.md):
  The discrete/hybrid energy-diffusion language-model components under
  ``thermolm_jax.models`` (FSQ, discrete_energy, d3pm, discrete_edlm,
  hybrid_*) and the corresponding trainers are unvalidated research sketches.
  Import them explicitly if you want to experiment, e.g.
  ``from thermolm_jax.models.discrete_edlm import DiscreteEDLM``.
"""

from .models.quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from .models.connectivity import generate_connectivity_pattern
from .models.dtm import DTM, DTMConfig
from .sampling.chromatic_gibbs import chromatic_gibbs_sample, greedy_coloring
from .training.contrastive_divergence import (
    contrastive_divergence_loss,
    contrastive_divergence_step,
    CDConfig,
)

# THRML availability (used by the THRML sampling path).
try:
    import thrml  # noqa: F401
    _has_thrml = True
except ImportError:
    _has_thrml = False

__version__ = "0.1.0"

__all__ = [
    "QuadraticEBM",
    "QuadraticEBMConfig",
    "generate_connectivity_pattern",
    "DTM",
    "DTMConfig",
    "chromatic_gibbs_sample",
    "greedy_coloring",
    "contrastive_divergence_loss",
    "contrastive_divergence_step",
    "CDConfig",
]
