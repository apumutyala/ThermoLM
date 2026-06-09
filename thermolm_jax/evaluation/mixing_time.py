"""
Mixing Time Analysis for DTM

Implements mixing time estimation for evaluating sampling efficiency.
Following Extropic.pdf, mixing time is critical for TSU hardware efficiency.

Design Decision: Autocorrelation-Based Mixing Time
- Rationale: Standard metric for MCMC efficiency
- Impact: Quantifies TSU sampling efficiency
- Trade-off: Requires sampling time
- Downstream: Enables TSU efficiency optimization

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional, Dict
from dataclasses import dataclass

from ..training.acp import estimate_autocorrelation


@dataclass
class MixingTimeConfig:
    """Configuration for mixing time analysis."""
    max_steps: int = 10000  # Maximum sampling steps
    target_autocorr: float = 0.1  # Target autocorrelation threshold
    lag: int = 100  # Lag for autocorrelation estimation


def estimate_mixing_time(
    energy_fn: Callable,
    init_state: jnp.ndarray,
    config: MixingTimeConfig,
    key: jax.random.PRNGKey
) -> Dict[str, float]:
    """
    Estimate mixing time from autocorrelation decay.
    
    Mixing time is the time when autocorrelation drops below threshold.
    
    Args:
        energy_fn: Energy function
        init_state: Initial state, shape (batch_size, n_vars)
        config: Mixing time configuration
        key: Random key
    
    Returns:
        metrics: Dictionary with mixing time metrics
    """
    from ..sampling.chromatic_gibbs import chromatic_gibbs_sample
    
    batch_size = init_state.shape[0]
    state = init_state
    
    # Sample trajectory
    energies = []
    samples = []
    
    for step in range(config.max_steps):
        key_step = jax.random.split(key)[0]
        state, _ = chromatic_gibbs_sample(
            energy_fn,
            state,
            n_steps=1,
            key=key_step,
            temperature=1.0
        )
        
        # Record samples for autocorrelation
        samples.append(state)
        
        # Record energy
        energy = jnp.mean(energy_fn(state))
        energies.append(energy)
    
    samples = jnp.stack(samples, axis=0)  # (max_steps, batch_size, n_vars)
    energies = jnp.array(energies)
    
    # Compute autocorrelation over time
    autocorrs = []
    for lag in range(1, config.max_steps // 10):
        if lag >= len(samples):
            break
        autocorr = estimate_autocorrelation(samples[:lag], lag=config.lag)
        autocorrs.append(autocorr)
    
    autocorrs = jnp.array(autocorrs)
    
    # Find mixing time (when autocorr drops below threshold)
    if len(autocorrs) > 0:
        mixing_time = jnp.argmax(autocorrs < config.target_autocorr)
        # If never drops below threshold, use max
        if autocorrs[mixing_time] >= config.target_autocorr:
            mixing_time = len(autocorrs)
    else:
        mixing_time = config.max_steps
    
    metrics = {
        "mixing_time": float(mixing_time),
        "final_autocorr": float(autocorrs[-1]) if len(autocorrs) > 0 else 0.0,
        "energy_mean": float(jnp.mean(energies)),
        "energy_std": float(jnp.std(energies)),
        "n_steps": config.max_steps,
    }
    
    return metrics


def integrated_autocorrelation_time(
    samples: jnp.ndarray,
    max_lag: Optional[int] = None
) -> float:
    """
    Compute integrated autocorrelation time (IACT).
    
    IACT is a more robust measure of mixing time.
    
    Args:
        samples: Sampled states, shape (n_samples, n_vars)
        max_lag: Maximum lag to consider
    
    Returns:
        iact: Integrated autocorrelation time
    """
    n_samples = samples.shape[0]
    
    if max_lag is None:
        max_lag = n_samples // 10
    
    # Compute autocorrelation at different lags
    autocorrs = []
    for lag in range(max_lag):
        if lag >= n_samples:
            break
        autocorr = estimate_autocorrelation(samples, lag=lag)
        autocorrs.append(autocorr)
    
    autocorrs = jnp.array(autocorrs)
    
    # IACT = 1 + 2 * Σ ρ(k)
    iact = 1.0 + 2.0 * jnp.sum(autocorrs)
    
    return float(iact)


def effective_sample_size(
    samples: jnp.ndarray,
    iact: Optional[float] = None
) -> float:
    """
    Compute effective sample size (ESS).
    
    ESS = n_samples / IACT
    
    Args:
        samples: Sampled states, shape (n_samples, n_vars)
        iact: Pre-computed IACT (if None, computes it)
    
    Returns:
        ess: Effective sample size
    """
    n_samples = samples.shape[0]
    
    if iact is None:
        iact = integrated_autocorrelation_time(samples)
    
    ess = n_samples / iact
    
    return float(ess)


def test_mixing_time():
    """Test mixing time analysis."""
    print("Testing mixing time analysis...")
    
    # Simple energy function
    def simple_energy(x):
        return -jnp.sum(x, axis=-1)
    
    config = MixingTimeConfig(max_steps=100, lag=10)
    
    key = jax.random.PRNGKey(0)
    init_state = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1
    
    # Test mixing time estimation
    metrics = estimate_mixing_time(simple_energy, init_state, config, key)
    
    print(f"Mixing time metrics: {metrics}")
    assert "mixing_time" in metrics
    assert "final_autocorr" in metrics
    
    # Test IACT
    samples = jax.random.randint(key, (100, 64), minval=0, maxval=2) * 2 - 1
    iact = integrated_autocorrelation_time(samples, max_lag=20)
    print(f"IACT: {iact}")
    assert iact > 0
    
    # Test ESS
    ess = effective_sample_size(samples, iact)
    print(f"ESS: {ess}")
    assert ess > 0
    assert ess < samples.shape[0]  # ESS should be less than total samples
    
    print("[SUCCESS] Mixing time analysis test passed!")


if __name__ == "__main__":
    test_mixing_time()
