"""
THRML-backed block Gibbs sampling for quadratic Ising EBMs.

The core object is ``THRMLIsingSampler``: a *setup-once* sampler that separates
static structure from traced arrays, so it is safe to call inside ``jit`` /
``grad`` (e.g. as the negative phase of a training step):

- **Static structure (built once, NumPy/Python):** the edge list derived from
  the connectivity mask, the ``SpinNode`` objects, and the colour-class
  ``Block``s. These are Python-level objects fixed at trace time.
- **Traced arrays (flow through transforms):** ``J``, ``h``, ``beta``, the
  sampling state, and PRNG keys. Edge weights are gathered from the traced
  ``J`` with ``jnp`` indexing — never ``np.asarray`` on a traced value.

An earlier version derived the edge structure from the (traced) coupling
matrix with NumPy inside the call, which raised ``TracerArrayConversionError``
under ``eqx.filter_value_and_grad`` — i.e. THRML-backed *training* was
impossible. This rewrite is the fix.

``THRMLQuadraticEBM`` is kept as a thin back-compat wrapper (eager use only).

Temperature enters exactly: sampling at T from exp(-E) is equivalent to
sampling at T=1 from a model with beta -> beta/T.
"""

from typing import List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from thermolm_jax.sampling.chromatic_gibbs import greedy_coloring


class THRMLIsingSampler:
    """Setup-once THRML block-Gibbs sampler for a quadratic Ising EBM.

    Build it ONCE from a concrete (non-traced) connectivity mask; then call
    :meth:`sample` with (possibly traced) parameters. The same instance should
    be reused across calls/steps — a new instance forces a retrace under jit
    (the jit cache is keyed on object identity).

    Args:
        connectivity_mask: (n, n) boolean adjacency (NumPy or concrete array).
            Symmetrised; diagonal ignored.
        colors: Optional (n,) integer colouring. Derived via
            ``greedy_coloring`` when omitted. Each colour class becomes one
            THRML free ``Block`` (a valid coloring is what makes the parallel
            block update correct).
    """

    def __init__(self, connectivity_mask, colors=None):
        mask = np.array(connectivity_mask, dtype=bool, copy=True)
        mask = mask | mask.T
        np.fill_diagonal(mask, False)
        self.n_vars = n = mask.shape[0]

        if colors is None:
            colors = greedy_coloring(mask)
        colors = np.asarray(colors)

        # Edge structure: one entry per unordered pair {i, j}, i < j.
        ei, ej = np.nonzero(np.triu(mask, 1))
        self._ei = ei  # kept as NumPy: the sampler object stays fully static
        self._ej = ej

        self.nodes = [SpinNode() for _ in range(n)]
        self.edges = [
            (self.nodes[i], self.nodes[j]) for i, j in zip(ei.tolist(), ej.tolist())
        ]

        # One free block per non-empty colour class. Empty classes (possible
        # with a gapped user-supplied colouring) are dropped: THRML's
        # BlockSpec rejects empty blocks.
        n_colors = int(colors.max()) + 1 if n > 0 else 0
        self.block_indices = [
            idx for idx in (np.nonzero(colors == c)[0] for c in range(n_colors))
            if len(idx) > 0
        ]
        self.n_colors = len(self.block_indices)
        self.free_blocks = [
            Block([self.nodes[int(i)] for i in idx]) for idx in self.block_indices
        ]

    def edge_weights(self, J: jnp.ndarray) -> jnp.ndarray:
        """Per-edge couplings gathered from a (possibly traced) J.

        Matches the energy convention in ``QuadraticEBM._effective_couplings``:
        weight of edge (i, j) is ``0.5 * (J[i, j] + J[j, i])``.
        """
        J_sym = 0.5 * (J + J.T)
        return J_sym[self._ei, self._ej]

    def build_ebm(self, J, h, beta, temperature: float = 1.0) -> IsingEBM:
        """THRML ``IsingEBM`` with traced parameters and static structure."""
        w = self.edge_weights(J)
        return IsingEBM(self.nodes, self.edges, h, w, jnp.asarray(beta) / temperature)

    def sample(
        self,
        J: jnp.ndarray,
        h: jnp.ndarray,
        beta,
        init_state: jnp.ndarray,
        key: jax.Array,
        n_steps: int,
        temperature: float = 1.0,
    ) -> Tuple[jnp.ndarray, dict]:
        """Run THRML block Gibbs from ``init_state`` for ``n_steps`` sweeps.

        Args:
            J, h, beta: model parameters (may be traced).
            init_state: (batch, n_vars) spins in {-1, +1}.
            key: PRNG key.
            n_steps: warmup sweeps before the single readout sample.
            temperature: sampling temperature T (beta -> beta/T).

        Returns:
            (samples, info): samples in {-1, +1}, shape (batch, n_vars).
        """
        ebm = self.build_ebm(J, h, beta, temperature)
        program = IsingSamplingProgram(ebm, self.free_blocks, [])
        schedule = SamplingSchedule(n_warmup=n_steps, n_samples=1, steps_per_sample=1)

        batch = init_state.shape[0]
        init_bool = init_state > 0  # {-1,+1} -> bool (THRML spin state is bool)
        init_per_block = [init_bool[:, idx] for idx in self.block_indices]

        def sample_single(inits_single, k):
            return sample_states(
                k, program, schedule, list(inits_single), [], self.free_blocks
            )

        keys = jax.random.split(key, batch)
        per_block_samples = jax.vmap(sample_single)(init_per_block, keys)
        # per_block_samples[c]: (batch, n_samples=1, |c|) bool

        out = jnp.zeros((batch, self.n_vars), dtype=jnp.float32)
        for c, idx in enumerate(self.block_indices):
            last = per_block_samples[c][:, -1, :]
            out = out.at[:, idx].set(2.0 * last.astype(jnp.float32) - 1.0)

        info = {
            "n_steps": n_steps,
            "temperature": temperature,
            "n_colors": self.n_colors,
            "sampling_method": "THRML IsingSamplingProgram",
        }
        return out, info


class THRMLQuadraticEBM:
    """Back-compat wrapper: quadratic EBM parameters + eager THRML sampling.

    For anything inside ``jit``/``grad`` (training), build a
    ``THRMLIsingSampler`` once and call it with traced parameters instead —
    this class converts its parameters with NumPy and is eager-only.

    Args:
        J: Pairwise weights matrix, shape (n_vars, n_vars).
        h: Unary biases, shape (n_vars,).
        beta: Inverse temperature.
        connectivity_mask: Sparse connectivity mask.
    """

    def __init__(
        self,
        J: jnp.ndarray,
        h: jnp.ndarray,
        beta: float,
        connectivity_mask: jnp.ndarray,
    ):
        self.J = J
        self.h = h
        self.beta = beta
        self.connectivity_mask = connectivity_mask

    def to_thrml_factors(self, nodes: Optional[List] = None) -> List:
        """Convert the quadratic EBM to THRML factors via ``IsingEBM``.

        Uses the sparse edge representation (only edges present in the
        connectivity mask), matching THRML's ``IsingEBM``.
        """
        n_vars = len(self.J)
        if nodes is None:
            nodes = [SpinNode() for _ in range(n_vars)]

        mask = np.asarray(self.connectivity_mask).astype(bool)
        J = np.asarray(self.J)
        edges, edge_weights = [], []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if mask[i, j]:
                    edges.append((nodes[i], nodes[j]))
                    edge_weights.append(J[i, j])

        ising_ebm = IsingEBM(
            nodes=nodes,
            edges=edges,
            biases=self.h,
            weights=jnp.array(edge_weights),
            beta=jnp.array(self.beta),
        )
        return ising_ebm.factors

    def compute_energy_from_factors(self, x: jnp.ndarray, factors: List) -> jnp.ndarray:
        """Single-counted quadratic energy (cross-check for factor energies).

        E(x) = -beta( sum_{i<j} J_ij x_i x_j + sum_i h_i x_i ).
        """
        J_sym = 0.5 * (self.J + self.J.T)
        J_eff = jnp.triu(J_sym * self.connectivity_mask, k=1)
        pairwise = jnp.einsum("...i,ij,...j->...", x, J_eff, x)
        unary = jnp.einsum("...i,i->...", x, self.h)
        return -self.beta * (pairwise + unary)

    def sample(
        self,
        init_state: jnp.ndarray,
        colors,
        n_steps: int,
        key: jax.Array,
        temperature: float = 1.0,
    ):
        """Eager THRML sampling (delegates to ``THRMLIsingSampler``)."""
        sampler = THRMLIsingSampler(np.asarray(self.connectivity_mask), colors)
        return sampler.sample(
            self.J, self.h, self.beta, init_state, key, n_steps, temperature
        )


def test_thrml_quadratic_ebm():
    """Smoke test for the wrapper and the setup-once sampler."""
    print("Testing THRMLQuadraticEBM / THRMLIsingSampler...")

    n_vars = 16
    key = jax.random.PRNGKey(0)

    J = jax.random.normal(key, (n_vars, n_vars)) * 0.01
    h = jax.random.normal(key, (n_vars,)) * 0.01
    beta = 1.0
    connectivity_mask = jnp.ones((n_vars, n_vars), dtype=bool)

    wrapper = THRMLQuadraticEBM(J, h, beta, connectivity_mask)

    x = jax.random.randint(key, (4, n_vars), minval=0, maxval=2) * 2 - 1
    energy = wrapper.compute_energy_from_factors(x, [])
    assert energy.shape == (4,), f"Expected shape (4,), got {energy.shape}"

    nodes = [SpinNode() for _ in range(n_vars)]
    factors = wrapper.to_thrml_factors(nodes)
    n_pairwise = int(np.triu(np.asarray(connectivity_mask), 1).sum())
    assert len(factors) == 2  # bias factor + pairwise factor

    sampler = THRMLIsingSampler(np.asarray(connectivity_mask))
    assert len(sampler.edges) == n_pairwise
    init = (jax.random.randint(key, (4, n_vars), 0, 2) * 2 - 1).astype(jnp.float32)
    out, info = sampler.sample(J, h, beta, init, key, n_steps=5)
    assert out.shape == (4, n_vars)
    assert set(np.unique(np.asarray(out))).issubset({-1.0, 1.0})

    print("[SUCCESS] THRMLQuadraticEBM test passed!")


if __name__ == "__main__":
    test_thrml_quadratic_ebm()
