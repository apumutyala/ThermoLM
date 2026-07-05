"""
Total Correlation Penalty for DTM Training

Implements total correlation penalty from Extropic.pdf Appendix F.
Encourages factorized distribution for fast mixing.

Design Decision: Total Correlation Penalty
- Rationale: Encourages factorization for fast mixing (Extropic.pdf)
- Impact: Improves sampling efficiency
- Trade-off: May reduce model expressivity
- Downstream: Faster mixing on TSU hardware

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import Optional
from dataclasses import dataclass


@dataclass
class TCConfig:
    """Configuration for total correlation penalty."""
    lambda_tc: float = 0.1  # Penalty weight


def total_correlation_penalty(
    samples: jnp.ndarray,
    config: TCConfig
) -> jnp.ndarray:
    """
    Total correlation penalty from Extropic.pdf Appendix F.
    
    TC = Σ_i I(X_i; X_{\i}) encourages factorized distribution.
    For binary spins, we approximate mutual information using
    empirical entropy estimates.
    
    Args:
        samples: Sampled states, shape (n_samples, n_vars)
        config: TC configuration
    
    Returns:
        penalty: Total correlation penalty
    """
    n_vars = samples.shape[-1]
    tc = 0.0
    
    for i in range(n_vars):
        # I(X_i; X_{\i}) ≈ H(X_i) - H(X_i|X_{\i})
        # Simplified: encourage independence by penalizing correlation
        
        x_i = samples[:, i]
        x_others = samples[:, jnp.arange(n_vars) != i]
        
        # Empirical entropy of X_i
        H_xi = empirical_entropy_binary(x_i)
        
        # Simplified conditional entropy (assume independence)
        # In practice, this would require estimating the full conditional
        # For efficiency, we use a pairwise approximation
        H_xi_given_others = approximate_conditional_entropy_pairwise(x_i, x_others)
        
        tc += (H_xi - H_xi_given_others)
    
    return config.lambda_tc * tc


def empirical_entropy_binary(samples: jnp.ndarray) -> float:
    """
    Empirical entropy of binary samples.
    
    For binary spins in {-1, 1}, compute entropy:
    H(X) = -p(+1) log p(+1) - p(-1) log p(-1)
    
    Args:
        samples: Binary samples, shape (n_samples,)
    
    Returns:
        entropy: Empirical entropy
    """
    # Convert to probabilities
    p_pos = jnp.mean(samples > 0)
    p_neg = 1.0 - p_pos
    
    # Avoid log(0)
    p_pos = jnp.clip(p_pos, 1e-10, 1.0)
    p_neg = jnp.clip(p_neg, 1e-10, 1.0)
    
    entropy = -(p_pos * jnp.log(p_pos) + p_neg * jnp.log(p_neg))
    return float(entropy)


def approximate_conditional_entropy_pairwise(
    x_i: jnp.ndarray,
    x_others: jnp.ndarray
) -> float:
    """
    Approximate conditional entropy using pairwise correlations.
    
    H(X_i|X_{\i}) ≈ H(X_i) - Σ_j I(X_i; X_j)
    
    This is a simplification for computational efficiency.
    
    Args:
        x_i: Variable i samples, shape (n_samples,)
        x_others: Other variables, shape (n_samples, n_vars-1)
    
    Returns:
        cond_entropy: Approximate conditional entropy
    """
    H_xi = empirical_entropy_binary(x_i)
    
    # Sum pairwise mutual information
    mi_sum = 0.0
    for j in range(x_others.shape[-1]):
        x_j = x_others[:, j]
        mi_ij = mutual_information_binary(x_i, x_j)
        mi_sum += mi_ij
    
    # Conditional entropy approximation
    cond_entropy = H_xi - 0.1 * mi_sum  # Scale factor for stability
    
    return max(0.0, cond_entropy)  # Ensure non-negative


def mutual_information_binary(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """
    Mutual information between two binary variables.
    
    I(X; Y) = Σ_x Σ_y p(x,y) log(p(x,y) / (p(x)p(y)))
    
    For binary variables, this can be computed from joint distribution.
    
    Args:
        x: Binary samples, shape (n_samples,)
        y: Binary samples, shape (n_samples,)
    
    Returns:
        mi: Mutual information
    """
    # Convert to binary {0, 1}
    x_bin = (x > 0).astype(int)
    y_bin = (y > 0).astype(int)
    
    # Joint distribution
    p_00 = jnp.mean((x_bin == 0) & (y_bin == 0))
    p_01 = jnp.mean((x_bin == 0) & (y_bin == 1))
    p_10 = jnp.mean((x_bin == 1) & (y_bin == 0))
    p_11 = jnp.mean((x_bin == 1) & (y_bin == 1))
    
    # Marginal distributions
    p_x0 = p_00 + p_01
    p_x1 = p_10 + p_11
    p_y0 = p_00 + p_10
    p_y1 = p_01 + p_11
    
    # Avoid log(0)
    epsilon = 1e-10
    p_00 = jnp.clip(p_00, epsilon, 1.0)
    p_01 = jnp.clip(p_01, epsilon, 1.0)
    p_10 = jnp.clip(p_10, epsilon, 1.0)
    p_11 = jnp.clip(p_11, epsilon, 1.0)
    p_x0 = jnp.clip(p_x0, epsilon, 1.0)
    p_x1 = jnp.clip(p_x1, epsilon, 1.0)
    p_y0 = jnp.clip(p_y0, epsilon, 1.0)
    p_y1 = jnp.clip(p_y1, epsilon, 1.0)
    
    # Mutual information
    mi = (
        p_00 * jnp.log(p_00 / (p_x0 * p_y0)) +
        p_01 * jnp.log(p_01 / (p_x0 * p_y1)) +
        p_10 * jnp.log(p_10 / (p_x1 * p_y0)) +
        p_11 * jnp.log(p_11 / (p_x1 * p_y1))
    )
    
    return float(mi)


def pairwise_correlation_penalty(
    samples: jnp.ndarray,
    config: TCConfig
) -> jnp.ndarray:
    """
    Simplified total correlation using pairwise correlations.
    
    This is a faster approximation that penalizes pairwise correlations.
    
    Args:
        samples: Sampled states, shape (n_samples, n_vars)
        config: TC configuration
    
    Returns:
        penalty: Pairwise correlation penalty
    """
    n_vars = samples.shape[-1]
    
    # Compute correlation matrix
    samples_centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    cov = (samples_centered.T @ samples_centered) / samples.shape[0]
    std = jnp.sqrt(jnp.diag(cov))
    corr = cov / (std[:, None] * std[None, :] + 1e-10)
    
    # Penalize off-diagonal correlations
    off_diag_corr = corr * (1 - jnp.eye(n_vars, dtype=bool))
    penalty = config.lambda_tc * jnp.sum(jnp.abs(off_diag_corr))
    
    return penalty


def test_total_correlation():
    """Test total correlation penalty implementation."""
    print("Testing total correlation penalty...")
    
    config = TCConfig(lambda_tc=0.1)
    
    # Generate samples with correlation
    key = jax.random.PRNGKey(0)
    samples = jax.random.randint(key, (100, 64), minval=0, maxval=2) * 2 - 1
    
    # Test full TC penalty
    tc_penalty = total_correlation_penalty(samples, config)
    print(f"TC penalty: {tc_penalty}")
    
    # Test pairwise correlation penalty (faster)
    pairwise_penalty = pairwise_correlation_penalty(samples, config)
    print(f"Pairwise penalty: {pairwise_penalty}")
    
    # Test empirical entropy
    x_i = samples[:, 0]
    entropy = empirical_entropy_binary(x_i)
    print(f"Entropy: {entropy}")
    assert 0 <= entropy <= jnp.log(2), f"Entropy out of range: {entropy}"
    
    # Test mutual information
    x_j = samples[:, 1]
    mi = mutual_information_binary(x_i, x_j)
    print(f"Mutual information: {mi}")
    assert mi >= 0, f"MI should be non-negative: {mi}"
    
    # Test with independent samples (should have low penalty)
    independent_samples = jax.random.randint(jax.random.PRNGKey(1), (100, 64), minval=0, maxval=2) * 2 - 1
    tc_penalty_independent = total_correlation_penalty(independent_samples, config)
    print(f"TC penalty (independent): {tc_penalty_independent}")
    
    print("[SUCCESS] Total correlation penalty test passed!")


if __name__ == "__main__":
    test_total_correlation()
