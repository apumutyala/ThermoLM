"""
Benchmark Module for ThermoLM JAX

Provides generic benchmarking utilities.

Note: This module provides only generic utility functions. Benchmarking against
external baselines (NVIDIA EDLM, DTM) is not implemented as it would require:
- Loading external model checkpoints
- Cross-framework compatibility (PyTorch to JAX)
- Access to proprietary model weights

Author: Apuroop Mutyala
Date: April 2026
"""

import jax.numpy as jnp
from typing import Dict, Any


def evaluate_model(
    model: Any,
    test_data: Any,
    n_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        model: Model to evaluate
        test_data: Test dataset
        n_samples: Number of samples to generate

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    import time

    # Generate samples
    start_time = time.time()
    samples = model.sample(n_samples)
    generation_time = time.time() - start_time

    # Compute metrics
    metrics = {
        'time': generation_time,
        'samples_per_second': n_samples / generation_time,
    }

    return metrics


def compare_metrics(
    our_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Compare two sets of metrics.

    Args:
        our_metrics: Our model's metrics
        baseline_metrics: Baseline model's metrics

    Returns:
        comparison: Dictionary of comparison ratios
    """
    comparison = {}
    for key in our_metrics:
        if key in baseline_metrics:
            comparison[key] = our_metrics[key] / baseline_metrics[key]
    return comparison
