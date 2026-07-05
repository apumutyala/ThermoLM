"""
Adaptive Correlation Penalty (ACP) for DTM Training

Implements ACP from Extropic.pdf Section IV.
Dynamically controls mixing time by penalizing autocorrelation.

Design Decision: Full ACP Implementation
- Rationale: Critical for TSU efficiency per Extropic.pdf
- Impact: Controls mixing time dynamically
- Trade-off: Requires autocorrelation estimation
- Downstream: Enables efficient TSU sampling

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from dataclasses import dataclass


@dataclass
class ACPConfig:
    """Configuration for Adaptive Correlation Penalty."""
    target_autocorr: float = 0.03  # Target autocorrelation (from Extropic.pdf)
    update_factor: float = 0.2  # Update factor for lambda
    lambda_min: float = 0.0001  # Minimum lambda value
    lambda_max: float = 1.0  # Maximum lambda value
    lag: int = 100  # Lag for autocorrelation estimation (K in paper)


class AdaptiveCorrelationPenalty(eqx.Module):
    """
    Adaptive Correlation Penalty from Extropic.pdf Section IV.
    
    ACP dynamically adjusts the penalty strength λ based on autocorrelation
    to control mixing time. If autocorrelation is too high (mixing too slow),
    increase penalty. If too low (mixing too fast), decrease penalty.
    
    Update rule:
    - If autocorr < target: λ_new = (1 - update_factor) * λ_current
    - If autocorr >= target: λ_new = (1 + update_factor) * λ_current
    
    Args:
        config: ACP configuration
    """
    
    config: ACPConfig
    lambda_current: float
    
    def __init__(self, config: ACPConfig, initial_lambda: float = 0.01):
        """Initialize ACP with configuration."""
        self.config = config
        self.lambda_current = initial_lambda
    
    def update(
        self,
        samples: jnp.ndarray
    ) -> float:
        """
        Update lambda based on autocorrelation of samples.
        
        Args:
            samples: Sampled states, shape (n_samples, n_vars)
        
        Returns:
            lambda_new: Updated lambda value
        """
        # Estimate autocorrelation at lag K
        autocorr = estimate_autocorrelation(samples, lag=self.config.lag)
        
        # Update lambda based on autocorrelation
        if autocorr < self.config.target_autocorr:
            # Mixing too fast: decrease penalty
            lambda_new = (1 - self.config.update_factor) * self.lambda_current
        else:
            # Mixing too slow: increase penalty
            lambda_new = (1 + self.config.update_factor) * self.lambda_current
        
        # Clamp to range
        lambda_new = jnp.clip(lambda_new, self.config.lambda_min, self.config.lambda_max)
        
        import equinox as eqx
        self = eqx.tree_at(lambda m: m.lambda_current, self, float(lambda_new))
        
        return self
    
    def compute_penalty(
        self,
        energy: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute ACP penalty term.
        
        Args:
            energy: Energy values, shape (batch,)
        
        Returns:
            penalty: ACP penalty, shape (batch,)
        """
        return self.lambda_current * jnp.mean(energy)
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_current


def estimate_autocorrelation(samples: jnp.ndarray, lag: int) -> float:
    """
    Estimate autocorrelation at given lag.
    
    For binary spins, computes autocorrelation of the mean spin value.
    
    Args:
        samples: Sampled states, shape (n_samples, n_vars)
        lag: Lag for autocorrelation estimation
    
    Returns:
        autocorr: Estimated autocorrelation
    """
    if len(samples) <= lag:
        return 0.0
    
    # Compute mean spin per sample
    mean_spins = jnp.mean(samples, axis=-1)  # (n_samples,)
    
    # Split into lag-separated samples
    samples_lag = mean_spins[:-lag]
    samples_current = mean_spins[lag:]
    
    # Compute mean and variance
    mean = jnp.mean(samples_current)
    var = jnp.var(samples_current)
    
    if var < 1e-10:
        return 0.0
    
    # Compute autocorrelation
    autocorr = jnp.mean((samples_lag - mean) * (samples_current - mean)) / var
    
    return float(autocorr)


def estimate_autocorrelation_per_variable(
    samples: jnp.ndarray,
    lag: int
) -> jnp.ndarray:
    """
    Estimate autocorrelation per variable.
    
    Args:
        samples: Sampled states, shape (n_samples, n_vars)
        lag: Lag for autocorrelation estimation
    
    Returns:
        autocorr_per_var: Autocorrelation per variable, shape (n_vars,)
    """
    if len(samples) <= lag:
        return jnp.zeros(samples.shape[-1])
    
    n_vars = samples.shape[-1]
    autocorrs = []
    
    for i in range(n_vars):
        var_samples = samples[:, i]
        samples_lag = var_samples[:-lag]
        samples_current = var_samples[lag:]
        
        mean = jnp.mean(samples_current)
        var = jnp.var(samples_current)
        
        if var < 1e-10:
            autocorrs.append(0.0)
        else:
            autocorr = jnp.mean((samples_lag - mean) * (samples_current - mean)) / var
            autocorrs.append(float(autocorr))
    
    return jnp.array(autocorrs)


def test_acp():
    """Test Adaptive Correlation Penalty implementation."""
    print("Testing Adaptive Correlation Penalty...")
    
    config = ACPConfig(target_autocorr=0.03, lag=10)
    acp = AdaptiveCorrelationPenalty(config, initial_lambda=0.01)
    
    # Test initial lambda
    assert acp.get_lambda() == 0.01
    print(f"Initial lambda: {acp.get_lambda()}")
    
    # Test autocorrelation estimation
    # Generate samples with high autocorrelation
    key = jax.random.PRNGKey(0)
    samples = jax.random.randint(key, (100, 64), minval=0, maxval=2) * 2 - 1
    
    autocorr = estimate_autocorrelation(samples, lag=10)
    print(f"Autocorrelation: {autocorr}")
    
    # Test per-variable autocorrelation
    autocorr_per_var = estimate_autocorrelation_per_variable(samples, lag=10)
    print(f"Autocorrelation per variable: {autocorr_per_var[:5]}...")
    assert autocorr_per_var.shape == (64,)
    
    # Test lambda update
    lambda_new = acp.update(samples)
    print(f"Updated lambda: {lambda_new}")
    assert lambda_new == acp.get_lambda()
    
    # Test penalty computation
    energy = jnp.array([1.0, 2.0, 3.0])
    penalty = acp.compute_penalty(energy)
    expected_penalty = acp.get_lambda() * jnp.mean(energy)
    assert jnp.allclose(penalty, expected_penalty)
    print(f"Penalty: {penalty}")
    
    # Test lambda clamping
    config_clamp = ACPConfig(lambda_min=0.1, lambda_max=0.5)
    acp_clamp = AdaptiveCorrelationPenalty(config_clamp, initial_lambda=0.01)
    
    # Try to decrease below min
    samples_fast = jax.random.randint(key, (100, 64), minval=0, maxval=2) * 2 - 1
    lambda_clamped = acp_clamp.update(samples_fast)
    assert lambda_clamped >= config_clamp.lambda_min
    print(f"Clamped lambda (min): {lambda_clamped}")
    
    print("[SUCCESS] ACP test passed!")


if __name__ == "__main__":
    test_acp()
