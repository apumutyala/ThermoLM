"""
Training package for ThermoLM JAX.

Two validated trainers for the quadratic Ising EBM, sharing one energy
convention and one THRMLIsingSampler static structure:

- ``contrastive_divergence``: CD-k — negative chain initialised AT the data,
  a few Gibbs sweeps (biased, low variance, fast). Pure-JAX or THRML
  negative phase.
- ``thrml_ml``: fully-visible maximum likelihood on THRML — positive-phase
  moments computed EXACTLY from the data (v0.1.3+), negative phase sampled
  by THRML's IsingSamplingProgram (unbiased in the long-chain limit).
"""

from .contrastive_divergence import (
    contrastive_divergence_loss,
    contrastive_divergence_step,
    CDConfig,
)
from .thrml_ml import (
    fit_ising_ml,
    make_kl_grad_step,
    params_from_ebm,
    params_to_ebm,
)

__all__ = [
    "contrastive_divergence_loss",
    "contrastive_divergence_step",
    "CDConfig",
    "fit_ising_ml",
    "make_kl_grad_step",
    "params_from_ebm",
    "params_to_ebm",
]
