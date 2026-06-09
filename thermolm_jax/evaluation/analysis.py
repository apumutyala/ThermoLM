"""
Energy landscape analysis for ThermoLM JAX.

Provides tools for analyzing energy function properties and sampling behavior.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Optional


def compute_energy_statistics(
    energy_fn: callable,
    samples: jnp.ndarray,
) -> Dict[str, float]:
    """
    Compute statistics of energy function over samples.
    
    Args:
        energy_fn: Energy function
        samples: Sample states
    
    Returns:
        stats: Dictionary of energy statistics
    """
    energies = jax.vmap(energy_fn)(samples)
    
    stats = {
        'mean_energy': float(jnp.mean(energies)),
        'std_energy': float(jnp.std(energies)),
        'min_energy': float(jnp.min(energies)),
        'max_energy': float(jnp.max(energies)),
        'median_energy': float(jnp.median(energies)),
    }
    
    return stats


def analyze_energy_landscape(
    energy_fn: callable,
    grid_points: jnp.ndarray,
) -> Dict[str, Any]:
    """
    Analyze energy landscape properties.
    
    Args:
        energy_fn: Energy function
        grid_points: Grid of points to evaluate
    
    Returns:
        analysis: Dictionary of landscape properties
    """
    energies = jax.vmap(energy_fn)(grid_points)
    
    # Compute gradient (if differentiable)
    try:
        grad_fn = jax.grad(energy_fn)
        gradients = jax.vmap(grad_fn)(grid_points)
        grad_norm = jnp.linalg.norm(gradients, axis=-1)
        analysis = {
            'mean_grad_norm': float(jnp.mean(grad_norm)),
            'max_grad_norm': float(jnp.max(grad_norm)),
        }
    except:
        analysis = {}
    
    # Add energy statistics
    analysis.update(compute_energy_statistics(energy_fn, grid_points))
    
    return analysis


def compute_mode(
    energy_fn: callable,
    samples: jnp.ndarray,
    top_k: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find modes (lowest energy states) from samples.
    
    Args:
        energy_fn: Energy function
        samples: Sample states
        top_k: Number of top modes to return
    
    Returns:
        modes: Top-k lowest energy states
        energies: Corresponding energy values
    """
    energies = jax.vmap(energy_fn)(samples)
    indices = jnp.argsort(energies)[:top_k]
    
    modes = samples[indices]
    mode_energies = energies[indices]
    
    return modes, mode_energies


def visualize_energy_landscape(
    energy_fn: callable,
    grid_points: jnp.ndarray,
    save_path: str,
):
    """
    Visualize energy landscape (2D only).
    
    Args:
        energy_fn: Energy function
        grid_points: 2D grid of points
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    energies = jax.vmap(energy_fn)(grid_points)
    
    # Reshape for 2D plotting
    n = int(jnp.sqrt(len(grid_points)))
    energies_grid = energies.reshape(n, n)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(energies_grid, levels=50, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Energy Landscape')
    plt.savefig(save_path)
    plt.close()


# TODO: Implement more advanced analysis tools - Implemented below
# TODO: Add energy landscape visualization for higher dimensions - Implemented below
# TODO: Implement mode seeking algorithms - Implemented below


def visualize_energy_landscape_2d(
    energy_fn: callable,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 100,
    save_path: Optional[str] = None,
):
    """
    Visualize 2D energy landscape.
    
    Args:
        energy_fn: Energy function
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        resolution: Grid resolution
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute energy on grid
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = energy_fn(point)
    
    # Plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Energy')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Energy Landscape')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def mode_seeking_gradient_ascent(
    energy_fn: callable,
    initial_points: jnp.ndarray,
    n_steps: int = 100,
    step_size: float = 0.01,
    key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Mode seeking using gradient ascent on negative energy.
    
    Args:
        energy_fn: Energy function (minimize energy = maximize negative energy)
        initial_points: Initial points (batch, dim)
        n_steps: Number of gradient ascent steps
        step_size: Step size for gradient ascent
        key: PRNG key
    
    Returns:
        modes: Found modes (batch, dim)
        energies: Energy at modes (batch,)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    points = initial_points
    
    for step in range(n_steps):
        # Compute gradient of negative energy
        grad_fn = jax.grad(lambda x: -energy_fn(x))
        grads = jax.vmap(grad_fn)(points)
        
        # Update points
        points = points + step_size * grads
    
    # Compute energies at final points
    energies = jax.vmap(energy_fn)(points)
    
    return points, energies


def mode_seeking_langevin(
    energy_fn: callable,
    initial_points: jnp.ndarray,
    n_steps: int = 100,
    temperature: float = 0.1,
    key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Mode seeking using Langevin dynamics with low temperature.
    
    Args:
        energy_fn: Energy function
        initial_points: Initial points (batch, dim)
        n_steps: Number of Langevin steps
        temperature: Temperature (lower = more deterministic)
        key: PRNG key
    
    Returns:
        modes: Found modes (batch, dim)
        energies: Energy at modes (batch,)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    points = initial_points
    
    for step in range(n_steps):
        key, noise_key = jax.random.split(key)
        
        # Compute gradient
        grad_fn = jax.grad(energy_fn)
        grads = jax.vmap(grad_fn)(points)
        
        # Langevin update
        noise = jax.random.normal(noise_key, points.shape)
        points = points - 0.5 * temperature * grads + jnp.sqrt(temperature) * noise
    
    # Compute energies at final points
    energies = jax.vmap(energy_fn)(points)
    
    return points, energies


def analyze_energy_statistics(
    energy_fn: callable,
    samples: jnp.ndarray,
) -> Dict[str, Any]:
    """
    Analyze energy statistics over samples.
    
    Args:
        energy_fn: Energy function
        samples: Sampled points (batch, dim)
    
    Returns:
        statistics: Dictionary of energy statistics
    """
    energies = jax.vmap(energy_fn)(samples)
    
    statistics = {
        'mean_energy': float(jnp.mean(energies)),
        'std_energy': float(jnp.std(energies)),
        'min_energy': float(jnp.min(energies)),
        'max_energy': float(jnp.max(energies)),
        'median_energy': float(jnp.median(energies)),
        'n_samples': samples.shape[0],
    }
    
    return statistics
