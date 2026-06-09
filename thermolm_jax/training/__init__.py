"""
Training package for ThermoLM JAX.

Validated: contrastive-divergence training of quadratic Ising EBMs
(``contrastive_divergence``) and the DTM trainer (``train_dtm``).

The other trainers (train_edlm/train_discrete_edlm/train_hybrid_edlm,
distributed, thrml_training, thrml_flax_coexistence, acp, total_correlation,
hybrid_training, ema, base_trainer, checkpoint, scheduler, optimizer) belong to
the unvalidated exploratory track (see STATUS.md) and are not imported here.
Import them explicitly by path if needed.
"""

from .contrastive_divergence import (
    contrastive_divergence_loss,
    contrastive_divergence_step,
    CDConfig,
)
from .train_dtm import DTMTrainer, TrainingConfig, train_dtm

__all__ = [
    "contrastive_divergence_loss",
    "contrastive_divergence_step",
    "CDConfig",
    "DTMTrainer",
    "TrainingConfig",
    "train_dtm",
]
