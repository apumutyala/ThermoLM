"""
Models package for ThermoLM JAX.

Validated core: the quadratic-Ising EBM (with its jit/grad-safe THRML
sampler) and the chain-CRF diffusion LM (exact inference + THRML reverse
step). See README.md and STATUS.md for scope and evidence.
"""

from .quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from .connectivity import generate_connectivity_pattern, get_connectivity_density
from .thrml_quadratic import THRMLIsingSampler, THRMLQuadraticEBM
from .chain_crf import (
    chain_log_partition,
    chain_score,
    chain_log_likelihood,
    chain_marginals,
    chain_sample,
)
from .diffusion_lm import (
    DiffusionLMConfig,
    build_net,
    init_params,
    denoising_loss,
    unigram_bits_per_char,
    fit,
    generate,
)
from .forward_coupling import ForwardCoupling, ForwardCouplingConfig

__all__ = [
    "QuadraticEBM",
    "QuadraticEBMConfig",
    "generate_connectivity_pattern",
    "get_connectivity_density",
    "THRMLIsingSampler",
    "THRMLQuadraticEBM",
    "chain_log_partition",
    "chain_score",
    "chain_log_likelihood",
    "chain_marginals",
    "chain_sample",
    "DiffusionLMConfig",
    "build_net",
    "init_params",
    "denoising_loss",
    "unigram_bits_per_char",
    "fit",
    "generate",
    "ForwardCoupling",
    "ForwardCouplingConfig",
]
