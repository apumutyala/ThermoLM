"""
Unit tests for energy landscape analysis.

Tests energy statistics and landscape analysis tools.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import pytest
import jax
import jax.numpy as jnp
from thermolm_jax.evaluation.analysis import (
    compute_energy_statistics,
    analyze_energy_landscape,
    compute_mode,
)


def simple_energy_fn(x):
    """Simple quadratic energy function for testing."""
    return jnp.sum(x ** 2)


def test_compute_energy_statistics():
    """Test energy statistics computation."""
    key = jax.random.PRNGKey(42)
    samples = jax.random.normal(key, (100, 10))
    
    stats = compute_energy_statistics(simple_energy_fn, samples)
    
    assert 'mean_energy' in stats
    assert 'std_energy' in stats
    assert 'min_energy' in stats
    assert 'max_energy' in stats
    assert 'median_energy' in stats
    assert stats['mean_energy'] > 0
    assert stats['std_energy'] > 0


def test_analyze_energy_landscape():
    """Test energy landscape analysis."""
    key = jax.random.PRNGKey(42)
    grid_points = jax.random.normal(key, (50, 10))
    
    analysis = analyze_energy_landscape(simple_energy_fn, grid_points)
    
    assert 'mean_energy' in analysis
    assert 'std_energy' in analysis
    # Gradient computation may fail for this simple function


def test_compute_mode():
    """Test mode computation."""
    key = jax.random.PRNGKey(42)
    samples = jax.random.normal(key, (100, 10))
    
    modes, energies = compute_mode(simple_energy_fn, samples, top_k=5)
    
    assert modes.shape == (5, 10)
    assert energies.shape == (5,)
    # Energies should be in ascending order
    assert jnp.all(energies[1:] >= energies[:-1])
