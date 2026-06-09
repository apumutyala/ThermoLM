"""
THRML Quadratic EBM Wrapper

Wraps quadratic EBM for THRML library integration.
Converts quadratic EBM to THRML factor format for block Gibbs sampling.

Design Decision: THRML Factor Decomposition
- Rationale: THRML requires factor-based energy representation
- Impact: Enables true THRML library integration
- Trade-off: Factor decomposition adds complexity
- Downstream: Enables TSU-compatible sampling

Author: Apuroop Mutyala
Date: April 16, 2026
"""

import jax
import jax.numpy as jnp
from typing import List, Optional

from thrml.models.discrete_ebm import SpinEBMFactor
from thrml.pgm import SpinNode
from thrml.block_management import Block


class THRMLQuadraticEBM:
    """
    Wrap quadratic EBM for THRML sampling.
    
    Converts quadratic EBM: E(x) = -β(Σ_{i≠j} x_i J_ij x_j + Σ_i h_i x_i)
    to THRML factor format: E = s_1 * ... * s_M * W[c_1, ..., c_N]
    
    Decomposition:
    - Unary factors (h_i x_i): Single-spin factors
    - Pairwise factors (J_ij x_i x_j): Two-spin factors
    
    Args:
        J: Pairwise weights matrix, shape (n_vars, n_vars)
        h: Unary biases, shape (n_vars,)
        beta: Inverse temperature
        connectivity_mask: Sparse connectivity mask
    """
    
    def __init__(
        self,
        J: jnp.ndarray,
        h: jnp.ndarray,
        beta: float,
        connectivity_mask: jnp.ndarray
    ):
        self.J = J
        self.h = h
        self.beta = beta
        self.connectivity_mask = connectivity_mask
    
    def to_thrml_factors(
        self,
        nodes: Optional[List] = None
    ) -> List:
        """
        Convert quadratic EBM to THRML factors using IsingEBM pattern.
        
        Uses sparse edge representation (not full matrix) to match THRML's IsingEBM.
        This is the correct approach per the integration plan.
        
        Args:
            nodes: List of THRML SpinNode objects (if None, creates them)
        
        Returns:
            factors: List of THRML SpinEBMFactor objects
        """
        from thrml.models import IsingEBM
        
        n_vars = len(self.J)
        
        # Create nodes if not provided
        if nodes is None:
            nodes = [SpinNode() for _ in range(n_vars)]
        
        # Create sparse edges for IsingEBM (CRITICAL: only non-zero entries)
        edges = []
        edge_weights = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if self.connectivity_mask[i, j]:
                    edges.append((nodes[i], nodes[j]))
                    edge_weights.append(self.J[i, j])
        
        # Create IsingEBM which will generate proper factors
        ising_ebm = IsingEBM(
            nodes=nodes,
            edges=edges,
            biases=self.h,
            weights=jnp.array(edge_weights),
            beta=jnp.array(self.beta)
        )
        
        return ising_ebm.factors
    
    def compute_energy_from_factors(
        self,
        x: jnp.ndarray,
        factors: List
    ) -> jnp.ndarray:
        """
        Compute energy directly from the quadratic form (single-counted).

        E(x) = -β( Σ_{i<j} J_ij x_i x_j + Σ_i h_i x_i ). Used to cross-check the
        THRML factor energy in tests.

        Args:
            x: Binary spins, shape (..., n_vars).
            factors: Unused (kept for signature compatibility).

        Returns:
            energy: Computed energy, shape (...).
        """
        J_sym = 0.5 * (self.J + self.J.T)
        J_eff = jnp.triu(J_sym * self.connectivity_mask, k=1)
        pairwise = jnp.einsum('...i,ij,...j->...', x, J_eff, x)
        unary = jnp.einsum('...i,i->...', x, self.h)
        return -self.beta * (pairwise + unary)

    def sample(
        self,
        init_state: jnp.ndarray,
        colors,
        n_steps: int,
        key: jax.random.PRNGKey,
        temperature: float = 1.0,
    ):
        """
        Sample from the quadratic EBM using THRML's validated Ising sampler.

        Builds a THRML ``IsingEBM`` and samples it with ``IsingSamplingProgram``,
        whose free super-blocks are the colour classes (independent sets). This
        is the TSU-compatible path; correctness is guaranteed by THRML provided
        the colouring is valid.

        Temperature enters exactly: sampling at T from exp(-E) is equivalent to
        sampling at T=1 from a model with β → β/T.

        Args:
            init_state: (batch, n_vars) spins in {-1, +1}.
            colors: (n_vars,) integer colour per node (e.g. from
                ``greedy_coloring`` of the connectivity mask).
            n_steps: Warmup sweeps.
            key: PRNG key.
            temperature: Sampling temperature.

        Returns:
            (samples_pm1, info): samples in {-1, +1}, shape (batch, n_vars).
        """
        import numpy as np
        from thrml import Block, SamplingSchedule, sample_states
        from thrml.pgm import SpinNode
        from thrml.models import IsingEBM
        from thrml.models.ising import IsingSamplingProgram

        batch, n = init_state.shape
        nodes = [SpinNode() for _ in range(n)]

        mask = np.asarray(self.connectivity_mask).astype(bool)
        J = np.asarray(self.J)
        edges, weights = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    edges.append((nodes[i], nodes[j]))
                    weights.append(0.5 * (J[i, j] + J[j, i]))

        ebm = IsingEBM(
            nodes=nodes,
            edges=edges,
            biases=jnp.asarray(self.h),
            weights=jnp.asarray(weights),
            beta=jnp.asarray(self.beta / temperature),
        )

        colors = np.asarray(colors)
        n_colors = int(colors.max()) + 1
        block_indices = [np.nonzero(colors == c)[0] for c in range(n_colors)]
        free_blocks = [Block([nodes[int(i)] for i in idx]) for idx in block_indices]

        program = IsingSamplingProgram(ebm, free_blocks, [])
        schedule = SamplingSchedule(n_warmup=n_steps, n_samples=1, steps_per_sample=1)

        init_bool = init_state > 0  # {-1,+1} -> bool
        init_per_block = [init_bool[:, idx] for idx in block_indices]  # each (batch, |c|)

        def sample_single(inits_single, k):
            return sample_states(k, program, schedule, list(inits_single), [], free_blocks)

        keys = jax.random.split(key, batch)
        per_block_samples = jax.vmap(sample_single)(init_per_block, keys)
        # per_block_samples[c]: (batch, n_samples=1, |c|) bool

        out = jnp.zeros((batch, n), dtype=jnp.float32)
        for c, idx in enumerate(block_indices):
            last = per_block_samples[c][:, -1, :]  # (batch, |c|) bool
            out = out.at[:, idx].set(2.0 * last.astype(jnp.float32) - 1.0)

        info = {
            "n_steps": n_steps,
            "temperature": temperature,
            "n_colors": n_colors,
            "sampling_method": "THRML IsingSamplingProgram",
        }
        return out, info


def test_thrml_quadratic_ebm():
    """Test THRML quadratic EBM wrapper."""
    print("Testing THRMLQuadraticEBM...")
    
    n_vars = 16
    key = jax.random.PRNGKey(0)
    
    # Create random weights
    J = jax.random.normal(key, (n_vars, n_vars)) * 0.01
    h = jax.random.normal(key, (n_vars,)) * 0.01
    beta = 1.0
    connectivity_mask = jnp.ones((n_vars, n_vars), dtype=bool)
    
    wrapper = THRMLQuadraticEBM(J, h, beta, connectivity_mask)
    
    # Test energy computation (fallback)
    x = jax.random.randint(key, (4, n_vars), minval=0, maxval=2) * 2 - 1
    energy = wrapper.compute_energy_from_factors(x, [])
    
    assert energy.shape == (4,), f"Expected shape (4,), got {energy.shape}"
    print(f"Energy shape: {energy.shape}")
    print(f"Energy values: {energy}")
    
    # Test THRML factor conversion
    nodes = [SpinNode() for _ in range(n_vars)]
    factors = wrapper.to_thrml_factors(nodes)
    print(f"Number of factors: {len(factors)}")
    
    # Verify factor count
    n_unary = n_vars
    n_pairwise = jnp.sum(connectivity_mask) // 2
    expected_factors = n_unary + n_pairwise
    assert len(factors) == expected_factors, f"Expected {expected_factors} factors, got {len(factors)}"
    
    print("[SUCCESS] THRMLQuadraticEBM test passed!")


if __name__ == "__main__":
    test_thrml_quadratic_ebm()
