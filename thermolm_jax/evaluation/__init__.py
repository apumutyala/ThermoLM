"""
Evaluation module for ThermoLM JAX

Contains metrics, benchmarking, and analysis tools.
"""

from .metrics import compute_perplexity
from .analysis import (
    compute_energy_statistics,
    analyze_energy_landscape,
    compute_mode,
    visualize_energy_landscape,
)

__all__ = [
    "compute_perplexity",
    "compute_energy_statistics",
    "analyze_energy_landscape",
    "compute_mode",
    "visualize_energy_landscape",
]
