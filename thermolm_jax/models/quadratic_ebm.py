"""
Quadratic Energy-Based Model for DTM

Implements quadratic energy-based model following Extropic.pdf Eq. 10:
E(x) = -β(Σ_{i≠j} x_i J_ij x_j + Σ_i h_i x_i)

This is the core energy function for Denoising Thermodynamic Models,
compatible with THRML library and TSU hardware.

Design Decision: Quadratic EBMs
- Rationale: Required by Extropic.pdf for TSU hardware compatibility
- Impact: Replaces neural network energy functions with quadratic form
- Trade-off: Limited expressivity vs hardware compatibility
- Downstream: Enables true THRML integration

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class QuadraticEBMConfig:
    """Configuration for quadratic EBM."""
    n_vars: int = 1024  # Number of binary variables
    connectivity_pattern: str = "G8"  # Connectivity pattern (G8, G12, G16, G20, G24)
    beta: float = 1.0  # Inverse temperature
    init_scale: float = 0.01  # Initialization scale for weights


class QuadraticEBM(eqx.Module):
    """
    Quadratic energy-based model following Extropic.pdf Eq. 10.
    
    Energy function: E(x) = -β(Σ_{i≠j} x_i J_ij x_j + Σ_i h_i x_i)
    
    where:
    - x_i ∈ {-1, 1} are binary spin variables
    - J_ij are pairwise interaction weights (sparse)
    - h_i are unary biases
    - β is the inverse temperature
    
    Args:
        J: Pairwise weights matrix, shape (n_vars, n_vars)
        h: Unary biases, shape (n_vars,)
        beta: Inverse temperature parameter
        connectivity_mask: Boolean mask for sparse connectivity
    """
    
    J: jnp.ndarray  # Pairwise weights
    h: jnp.ndarray  # Unary biases
    beta: float  # Inverse temperature
    connectivity_mask: jnp.ndarray  # Sparse connectivity mask
    
    def __init__(
        self,
        config: QuadraticEBMConfig,
        key: jax.random.PRNGKey
    ):
        """Initialize quadratic EBM with random weights."""
        self.beta = config.beta
        
        # Initialize pairwise weights with sparse connectivity
        key_J, key_h = jax.random.split(key)
        
        # Initialize J with small random values
        self.J = jax.random.normal(
            key_J, 
            (config.n_vars, config.n_vars)
        ) * config.init_scale
        
        # Initialize h with small random values
        self.h = jax.random.normal(
            key_h, 
            (config.n_vars,)
        ) * config.init_scale
        
        # Connectivity mask: default fully-connected with no self-loops.
        # Set a sparse pattern explicitly via set_connectivity().
        self.connectivity_mask = jnp.ones((config.n_vars, config.n_vars), dtype=bool) & ~jnp.eye(
            config.n_vars, dtype=bool
        )

    def _effective_couplings(self) -> jnp.ndarray:
        """Symmetric, strictly-upper-triangular coupling matrix.

        Each unordered pair {i, j} is represented exactly once (i < j), with no
        diagonal (self-coupling) term. This is what makes the einsum below
        evaluate ``Σ_{i<j} J_ij x_i x_j`` rather than double-counting pairs or
        adding a constant ``Σ_i J_ii`` term (recall ``x_i^2 = 1`` for spins).
        """
        J_sym = 0.5 * (self.J + self.J.T)
        return jnp.triu(J_sym * self.connectivity_mask, k=1)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute energy for binary spin configuration.

        E(x) = -β( Σ_{i<j} J_ij x_i x_j + Σ_i h_i x_i ),  x_i ∈ {-1, +1}

        Args:
            x: Binary spins, shape (..., n_vars) with values in {-1, +1}.
            mask: Optional {0,1} mask, shape (..., n_vars); masked-out spins
                contribute nothing to either term (treated as absent).

        Returns:
            energy: Energy per configuration, shape (...).
        """
        if mask is not None:
            # Zero out masked spins so they drop out of both sums. This does not
            # renormalise the surviving energy (the previous implementation
            # rescaled by n_valid, which silently changed the distribution).
            x = x * mask

        J_eff = self._effective_couplings()
        pairwise = jnp.einsum('...i,ij,...j->...', x, J_eff, x)
        unary = jnp.einsum('...i,i->...', x, self.h)

        return -self.beta * (pairwise + unary)

    def set_connectivity(self, connectivity_mask: jnp.ndarray):
        """
        Set sparse connectivity mask (returns a NEW module — Equinox is immutable).

        The mask is symmetrised and its diagonal cleared so that the energy is a
        well-formed pairwise Ising form. Usage:  ``ebm = ebm.set_connectivity(m)``.

        Args:
            connectivity_mask: Boolean mask, shape (n_vars, n_vars).
        """
        mask = connectivity_mask.astype(bool)
        mask = (mask | mask.T) & ~jnp.eye(mask.shape[0], dtype=bool)
        self = eqx.tree_at(lambda m: m.connectivity_mask, self, mask)
        # Zero out weights for disconnected edges (keeps stored J interpretable).
        J_masked = self.J * mask.astype(self.J.dtype)
        self = eqx.tree_at(lambda m: m.J, self, J_masked)
        return self

    def get_energy_components(
        self,
        x: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get separate (pairwise, unary) energy components for analysis.

        Args:
            x: Binary spins, shape (..., n_vars) with values in {-1, +1}.

        Returns:
            pairwise_energy, unary_energy — each shape (...).
        """
        J_eff = self._effective_couplings()
        pairwise = -self.beta * jnp.einsum('...i,ij,...j->...', x, J_eff, x)
        unary = -self.beta * jnp.einsum('...i,i->...', x, self.h)
        return pairwise, unary
    
    def hinton_initialize(
        self,
        key: jax.random.PRNGKey,
        batch_size: int = 1
    ) -> jnp.ndarray:
        """
        Initialize state from the bias marginal (Hinton's heuristic).

        Each unit is sampled independently as P(S_i = +1) = σ(β h_i) — the
        couplings play no role, so this is three lines of JAX. (An earlier
        version built a full THRML IsingEBM with an O(n²) Python edge loop
        just to call thrml's hinton_init; equivalent, but wasteful.)

        Args:
            key: JAX random key
            batch_size: Number of samples to initialize

        Returns:
            initialized_state: spins in {-1, +1}, shape (batch_size, n_vars)
        """
        p_plus = jax.nn.sigmoid(self.beta * self.h)
        bern = jax.random.bernoulli(key, p_plus, shape=(batch_size, len(self.h)))
        return 2 * bern.astype(jnp.int8) - 1


def test_quadratic_ebm():
    """Test quadratic EBM implementation."""
    print("Testing QuadraticEBM...")
    
    config = QuadraticEBMConfig(n_vars=64, connectivity_pattern="G8")
    key = jax.random.PRNGKey(0)
    
    ebm = QuadraticEBM(config, key)
    
    # Test energy computation
    x = jax.random.randint(key, (4, 64), minval=0, maxval=2) * 2 - 1  # {-1, 1}
    energy = ebm(x)
    
    assert energy.shape == (4,), f"Expected shape (4,), got {energy.shape}"
    print(f"Energy shape: {energy.shape}")
    print(f"Energy values: {energy}")
    
    # Test energy components
    pairwise, unary = ebm.get_energy_components(x)
    assert pairwise.shape == (4,), f"Expected shape (4,), got {pairwise.shape}"
    assert unary.shape == (4,), f"Expected shape (4,), got {unary.shape}"
    print(f"Pairwise energy: {pairwise}")
    print(f"Unary energy: {unary}")
    
    # Test with mask
    mask = jnp.ones((4, 64))
    mask = mask.at[:, 32:].set(0)
    energy_masked = ebm(x, mask)
    print(f"Masked energy: {energy_masked}")
    
    print("[SUCCESS] QuadraticEBM test passed!")


if __name__ == "__main__":
    test_quadratic_ebm()
