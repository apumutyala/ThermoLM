"""
ThermoLM JAX — energy-based models on (simulated) thermodynamic hardware.

Validated track (what this package exposes):
  - Quadratic Ising EBMs sampled with chromatic block Gibbs, trained by
    contrastive divergence or THRML-native maximum likelihood (exact
    positive phase), with a jit/grad-safe THRML sampling path.
  - A chain-CRF discrete-diffusion language model with exact inference
    (forward-backward, FFBS) whose reverse step also runs on THRML.
  See README.md and STATUS.md for the validation evidence and scope.

Exploratory track: legacy research sketches live under the repo-level
``experimental/`` directory (NOT importable from this package); see
STATUS.md for their known defects.
"""

from .models.quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from .models.connectivity import generate_connectivity_pattern
from .models.thrml_quadratic import THRMLIsingSampler
from .models.chain_crf import (
    chain_log_partition,
    chain_log_likelihood,
    chain_marginals,
    chain_sample,
)
from .models.diffusion_lm import DiffusionLMConfig, fit, generate
from .data.char_tokenizer import CharTokenizer, make_windows
from .sampling.chromatic_gibbs import (
    chromatic_gibbs_sample,
    greedy_coloring,
    color_masks_from_colors,
)
from .training.contrastive_divergence import (
    contrastive_divergence_loss,
    contrastive_divergence_step,
    CDConfig,
)

__version__ = "0.2.0"

__all__ = [
    "QuadraticEBM",
    "QuadraticEBMConfig",
    "THRMLIsingSampler",
    "generate_connectivity_pattern",
    "chain_log_partition",
    "chain_log_likelihood",
    "chain_marginals",
    "chain_sample",
    "DiffusionLMConfig",
    "fit",
    "generate",
    "CharTokenizer",
    "make_windows",
    "chromatic_gibbs_sample",
    "greedy_coloring",
    "color_masks_from_colors",
    "contrastive_divergence_loss",
    "contrastive_divergence_step",
    "CDConfig",
]
