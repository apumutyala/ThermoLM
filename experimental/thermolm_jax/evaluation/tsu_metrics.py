"""
TSU Efficiency Metrics for DTM

Implements TSU energy consumption estimation from Extropic.pdf Appendix D.
Critical for quantifying the energy efficiency advantage of TSU hardware.

Design Decision: TSU Energy Estimation
- Rationale: Quantifies energy efficiency advantage over GPUs
- Impact: Demonstrates TSU hardware value
- Trade-off: Estimates based on paper parameters
- Downstream: Enables cost-benefit analysis

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import Dict
from dataclasses import dataclass


@dataclass
class TSUMetricsConfig:
    """Configuration for TSU efficiency metrics."""
    # Energy per sampling cell (from Extropic.pdf Appendix D)
    E_cell: float = 1e-15  # Joules per sample per cell
    
    # Connectivity densities from Extropic.pdf Table I
    connectivity_densities: Dict[str, float] = None
    
    def __post_init__(self):
        if self.connectivity_densities is None:
            self.connectivity_densities = {
                "G8": 0.125,
                "G12": 0.188,
                "G16": 0.25,
                "G20": 0.313,
                "G24": 0.375,
            }


def estimate_tsu_energy_consumption(
    n_vars: int,
    connectivity_pattern: str,
    n_samples: int,
    mixing_time: int,
    config: TSUMetricsConfig
) -> Dict[str, float]:
    """
    Estimate TSU energy consumption from Extropic.pdf Appendix D.
    
    Energy = E_cell * n_active_cells * n_samples * mixing_time
    
    where:
    - n_active_cells = n_vars * connectivity_density
    - mixing_time = number of Gibbs steps to reach equilibrium
    
    Args:
        n_vars: Number of variables
        connectivity_pattern: Connectivity pattern (G8, G12, etc.)
        n_samples: Number of samples to generate
        mixing_time: Mixing time (Gibbs steps)
        config: TSU metrics configuration
    
    Returns:
        energy_metrics: Dictionary with energy consumption metrics
    """
    # Get connectivity density
    density = config.connectivity_densities.get(
        connectivity_pattern,
        0.25  # Default to G16 density
    )
    
    # Number of active cells
    n_active_cells = n_vars * density
    
    # Total energy consumption
    total_energy = (
        config.E_cell *
        n_active_cells *
        n_samples *
        mixing_time
    )
    
    # Energy per sample
    energy_per_sample = total_energy / n_samples
    
    # Energy per variable
    energy_per_var = total_energy / (n_samples * n_vars)
    
    energy_metrics = {
        "total_energy_joules": float(total_energy),
        "energy_per_sample_joules": float(energy_per_sample),
        "energy_per_var_joules": float(energy_per_var),
        "n_active_cells": float(n_active_cells),
        "connectivity_density": float(density),
        "mixing_time": float(mixing_time),
    }
    
    return energy_metrics


def compare_tsu_vs_gpu(
    tsu_energy_joules: float,
    gpu_power_watts: float = 300.0,
    gpu_time_seconds: float = 1.0
) -> Dict[str, float]:
    """
    Compare TSU energy consumption vs GPU.
    
    Args:
        tsu_energy_joules: TSU energy consumption in joules
        gpu_power_watts: GPU power consumption in watts (default 300W for RTX 3090)
        gpu_time_seconds: GPU computation time in seconds
    
    Returns:
        comparison_metrics: Dictionary with comparison metrics
    """
    # GPU energy consumption
    gpu_energy_joules = gpu_power_watts * gpu_time_seconds
    
    # Energy efficiency ratio
    energy_ratio = gpu_energy_joules / tsu_energy_joules
    
    # Percentage savings
    energy_savings_percent = (1 - tsu_energy_joules / gpu_energy_joules) * 100
    
    comparison_metrics = {
        "tsu_energy_joules": float(tsu_energy_joules),
        "gpu_energy_joules": float(gpu_energy_joules),
        "energy_ratio": float(energy_ratio),
        "energy_savings_percent": float(energy_savings_percent),
        "gpu_power_watts": float(gpu_power_watts),
        "gpu_time_seconds": float(gpu_time_seconds),
    }
    
    return comparison_metrics


def estimate_tpu_flops(
    n_vars: int,
    connectivity_pattern: str,
    n_samples: int,
    mixing_time: int,
    flops_per_cell: float = 1e6  # FLOPs per cell per sample
) -> Dict[str, float]:
    """
    Estimate computational complexity in FLOPs.
    
    Args:
        n_vars: Number of variables
        connectivity_pattern: Connectivity pattern
        n_samples: Number of samples
        mixing_time: Mixing time
        flops_per_cell: FLOPs per cell per sample
    
    Returns:
        flop_metrics: Dictionary with FLOP metrics
    """
    config = TSUMetricsConfig()
    density = config.connectivity_densities.get(connectivity_pattern, 0.25)
    n_active_cells = n_vars * density
    
    total_flops = flops_per_cell * n_active_cells * n_samples * mixing_time
    flops_per_sample = total_flops / n_samples
    
    flop_metrics = {
        "total_flops": float(total_flops),
        "flops_per_sample": float(flops_per_sample),
        "n_active_cells": float(n_active_cells),
    }
    
    return flop_metrics


def test_tsu_metrics():
    """Test TSU efficiency metrics."""
    print("Testing TSU efficiency metrics...")
    
    config = TSUMetricsConfig()
    
    # Test energy estimation
    energy_metrics = estimate_tsu_energy_consumption(
        n_vars=1024,
        connectivity_pattern="G16",
        n_samples=1000,
        mixing_time=100,
        config=config
    )
    
    print(f"TSU energy metrics: {energy_metrics}")
    assert "total_energy_joules" in energy_metrics
    assert "energy_per_sample_joules" in energy_metrics
    
    # Test TSU vs GPU comparison
    comparison = compare_tsu_vs_gpu(
        tsu_energy_joules=energy_metrics["total_energy_joules"],
        gpu_power_watts=300.0,
        gpu_time_seconds=10.0
    )
    
    print(f"TSU vs GPU comparison: {comparison}")
    assert "energy_savings_percent" in comparison
    assert comparison["energy_savings_percent"] > 0  # TSU should be more efficient
    
    # Test FLOP estimation
    flop_metrics = estimate_tpu_flops(
        n_vars=1024,
        connectivity_pattern="G16",
        n_samples=1000,
        mixing_time=100
    )
    
    print(f"FLOP metrics: {flop_metrics}")
    assert "total_flops" in flop_metrics
    
    # Test different connectivity patterns
    for pattern in ["G8", "G12", "G16", "G20", "G24"]:
        energy = estimate_tsu_energy_consumption(
            n_vars=1024,
            connectivity_pattern=pattern,
            n_samples=1000,
            mixing_time=100,
            config=config
        )
        print(f"{pattern} energy: {energy['total_energy_joules']:.2e} J")
    
    print("[SUCCESS] TSU efficiency metrics test passed!")


if __name__ == "__main__":
    test_tsu_metrics()
