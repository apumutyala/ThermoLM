"""
Energy Landscape Analysis for DTM

Implements energy landscape analysis to understand model behavior.
Critical for understanding mixing properties and mode collapse.

Design Decision: Comprehensive Energy Analysis
- Rationale: Understanding energy landscape is crucial for EBMs
- Impact: Identifies training issues and mode collapse
- Trade-off: Requires sampling time
- Downstream: Enables model debugging

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict
from dataclasses import dataclass


@dataclass
class EnergyLandscapeConfig:
    """Configuration for energy landscape analysis."""
    n_samples: int = 1000  # Number of samples to analyze
    n_bins: int = 50  # Number of histogram bins


def analyze_energy_landscape(
    energy_fn: Callable,
    samples: jnp.ndarray,
    config: EnergyLandscapeConfig
) -> Dict[str, float]:
    """
    Analyze energy landscape properties.
    
    Args:
        energy_fn: Energy function
        samples: Sampled states, shape (n_samples, n_vars)
        config: Analysis configuration
    
    Returns:
        metrics: Dictionary with energy landscape metrics
    """
    # Compute energies
    energies = energy_fn(samples)
    
    # Basic statistics
    metrics = {
        "mean_energy": float(jnp.mean(energies)),
        "std_energy": float(jnp.std(energies)),
        "min_energy": float(jnp.min(energies)),
        "max_energy": float(jnp.max(energies)),
        "energy_range": float(jnp.max(energies) - jnp.min(energies)),
        "energy_variance": float(jnp.var(energies)),
        "median_energy": float(jnp.median(energies)),
    }
    
    # Energy distribution metrics
    q25, q75 = jnp.percentile(energies, jnp.array([25, 75]))
    metrics["q25_energy"] = float(q25)
    metrics["q75_energy"] = float(q75)
    metrics["iqr_energy"] = float(q75 - q25)
    
    # Skewness (approximate)
    mean = metrics["mean_energy"]
    std = metrics["std_energy"]
    if std > 1e-10:
        skew = jnp.mean(((energies - mean) / std) ** 3)
        metrics["skewness"] = float(skew)
    else:
        metrics["skewness"] = 0.0
    
    # Kurtosis (approximate)
    if std > 1e-10:
        kurtosis = jnp.mean(((energies - mean) / std) ** 4) - 3
        metrics["kurtosis"] = float(kurtosis)
    else:
        metrics["kurtosis"] = 0.0
    
    return metrics


def energy_histogram(
    energy_fn: Callable,
    samples: jnp.ndarray,
    config: EnergyLandscapeConfig
) -> Dict[str, jnp.ndarray]:
    """
    Compute energy histogram.
    
    Args:
        energy_fn: Energy function
        samples: Sampled states
        config: Analysis configuration
    
    Returns:
        histogram_data: Dictionary with histogram data
    """
    energies = energy_fn(samples)
    
    # Compute histogram
    hist, bin_edges = jnp.histogram(
        energies,
        bins=config.n_bins,
        density=True
    )
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
    }


def detect_mode_collapse(
    energy_fn: Callable,
    samples: jnp.ndarray,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Detect mode collapse in energy landscape.
    
    Mode collapse occurs when most samples cluster around a single mode.
    
    Args:
        energy_fn: Energy function
        samples: Sampled states
        threshold: Threshold for collapse detection
    
    Returns:
        collapse_metrics: Dictionary with collapse metrics
    """
    energies = energy_fn(samples)
    
    # Compute energy histogram
    hist, bin_edges = jnp.histogram(energies, bins=20, density=True)
    
    # Find dominant mode
    max_bin = jnp.argmax(hist)
    max_prob = hist[max_bin]
    
    # Mode collapse if dominant mode has > threshold probability
    is_collapsed = max_prob > threshold
    
    # Entropy of energy distribution
    hist_normalized = hist / (jnp.sum(hist) + 1e-10)
    entropy = -jnp.sum(hist_normalized * jnp.log(hist_normalized + 1e-10))
    max_entropy = jnp.log(len(hist))
    entropy_ratio = entropy / max_entropy
    
    collapse_metrics = {
        "is_collapsed": bool(is_collapsed),
        "max_prob": float(max_prob),
        "entropy": float(entropy),
        "max_entropy": float(max_entropy),
        "entropy_ratio": float(entropy_ratio),
    }
    
    return collapse_metrics


def energy_gradient_norm(
    energy_fn: Callable,
    samples: jnp.ndarray
) -> Dict[str, float]:
    """
    Compute energy gradient norms to understand landscape smoothness.
    
    Args:
        energy_fn: Energy function
        samples: Sampled states
    
    Returns:
        gradient_metrics: Dictionary with gradient metrics
    """
    def compute_grad(x):
        return jax.grad(energy_fn)(x)
    
    # Compute gradients for each sample
    grad_fn = jax.vmap(compute_grad)
    grads = grad_fn(samples)
    
    # Compute gradient norms
    grad_norms = jnp.linalg.norm(grads, axis=-1)
    
    gradient_metrics = {
        "mean_grad_norm": float(jnp.mean(grad_norms)),
        "std_grad_norm": float(jnp.std(grad_norms)),
        "max_grad_norm": float(jnp.max(grad_norms)),
        "min_grad_norm": float(jnp.min(grad_norms)),
    }
    
    return gradient_metrics


def test_energy_landscape():
    """Test energy landscape analysis."""
    print("Testing energy landscape analysis...")
    
    # Simple energy function
    def simple_energy(x):
        return -jnp.sum(x, axis=-1)
    
    config = EnergyLandscapeConfig(n_samples=100, n_bins=20)
    
    key = jax.random.PRNGKey(0)
    samples = jax.random.randint(key, (100, 64), minval=0, maxval=2) * 2 - 1
    
    # Test energy landscape analysis
    metrics = analyze_energy_landscape(simple_energy, samples, config)
    
    print(f"Energy landscape metrics: {metrics}")
    assert "mean_energy" in metrics
    assert "std_energy" in metrics
    
    # Test histogram
    hist_data = energy_histogram(simple_energy, samples, config)
    print(f"Histogram bins: {len(hist_data['hist'])}")
    assert len(hist_data["hist"]) == config.n_bins
    
    # Test mode collapse detection
    collapse_metrics = detect_mode_collapse(simple_energy, samples)
    print(f"Mode collapse metrics: {collapse_metrics}")
    assert "is_collapsed" in collapse_metrics
    
    # Test gradient norms
    grad_metrics = energy_gradient_norm(simple_energy, samples)
    print(f"Gradient metrics: {grad_metrics}")
    assert "mean_grad_norm" in grad_metrics
    
    print("[SUCCESS] Energy landscape analysis test passed!")


if __name__ == "__main__":
    test_energy_landscape()
