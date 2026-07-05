"""
Chromatic (block) Gibbs sampling for quadratic Ising EBMs.

Chromatic Gibbs sampling colours the interaction graph so that each colour
class is an *independent set* (no edges within a class). All spins in one class
are then conditionally independent given the rest, so they can be resampled
simultaneously — the parallelism that maps onto TSU hardware. Correctness rests
entirely on the colouring being a valid graph colouring (Hammersley–Clifford /
Markov-random-field conditional independence); see ``greedy_coloring`` below.

For a spin model E(x) = -β( Σ_{i<j} J_ij x_i x_j + Σ_i h_i x_i ), x_i ∈ {-1,+1},
the single-site conditional is

    P(x_i = +1 | x_{-i}) = σ( 2β f_i / T ),   f_i = Σ_j J_ij x_j + h_i

where f_i is the local field. (The factor 2β is the energy gap ΔE = 2β f_i
between x_i = ±1; a previous version applied an *extra* ×2 to the full ΔE, i.e.
sampled at half the requested temperature — that bug is fixed here.)

A correct THRML-backed path is available via ``use_thrml=True`` for quadratic
EBMs; it delegates to THRML's validated ``IsingSamplingProgram``. Inside
``jit``/``grad`` (e.g. a training step), pass a prebuilt ``thrml_sampler``
(``thermolm_jax.models.thrml_quadratic.THRMLIsingSampler``) so the edge
structure is static and only arrays are traced.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple


def greedy_coloring(adjacency: jnp.ndarray) -> np.ndarray:
    """Greedy (Welsh–Powell) graph colouring.

    Returns an integer colour per node such that adjacent nodes never share a
    colour, i.e. each colour class is an independent set. Computed in NumPy at
    setup time (not inside ``jit``); the resulting colour arrays are then used
    by the jittable sampler.

    Args:
        adjacency: (n, n) boolean/0-1 symmetric adjacency (diagonal ignored).

    Returns:
        colors: (n,) int array of colour indices.
    """
    adj = np.asarray(adjacency).astype(bool)
    n = adj.shape[0]
    np.fill_diagonal(adj, False)
    degrees = adj.sum(axis=1)
    order = np.argsort(-degrees)  # high degree first
    colors = np.full(n, -1, dtype=np.int64)
    for v in order:
        used = set(colors[u] for u in np.nonzero(adj[v])[0] if colors[u] >= 0)
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


def _full_couplings(ebm) -> jnp.ndarray:
    """Symmetric, zero-diagonal coupling matrix J_ij used for local fields."""
    j_upper = ebm._effective_couplings()  # strict upper triangle
    return j_upper + j_upper.T


def color_masks_from_colors(colors, n_vars: int, dtype=jnp.float32) -> jnp.ndarray:
    """Build a (n_colors, n_vars) one-hot-per-class mask from a colouring.

    Hoisting this (and ``greedy_coloring``) out of the sampler lets the sampler
    be ``jit``-compiled: the colouring is Python/NumPy work done once, and only
    the resulting array is threaded into the traced region.
    """
    colors = np.asarray(colors)
    n_colors = int(colors.max()) + 1
    return jnp.stack([jnp.asarray(colors == c, dtype=dtype) for c in range(n_colors)])


def chromatic_gibbs_sample(
    energy_fn: Callable,
    init_state: jnp.ndarray,
    n_steps: int,
    key: jax.random.PRNGKey,
    temperature: float = 1.0,
    use_thrml: bool = False,
    color_masks: Optional[jnp.ndarray] = None,
    thrml_sampler=None,
) -> Tuple[jnp.ndarray, dict]:
    """Chromatic Gibbs sampling.

    For a quadratic EBM (an object exposing ``J``, ``h``, ``beta``,
    ``connectivity_mask``) this runs true chromatic block Gibbs using local
    fields. For a generic ``energy_fn`` it falls back to systematic single-site
    Gibbs, which is correct for any graph.

    Args:
        energy_fn: Either a quadratic EBM module, or a callable
            ``energy_fn(state) -> (...)`` energy.
        init_state: (batch, n_vars) spins in {-1, +1}.
        n_steps: Number of full Gibbs sweeps.
        key: PRNG key.
        temperature: Sampling temperature T.
        use_thrml: If True and ``energy_fn`` is quadratic, sample via THRML's
            ``IsingSamplingProgram`` instead of the JAX implementation.
        color_masks: Optional precomputed (n_colors, n_vars) colour masks (see
            ``color_masks_from_colors``). Pass this to keep the quadratic
            sampler ``jit``-able (avoids running graph colouring in-trace). If
            omitted it is derived from the EBM's connectivity.
        thrml_sampler: Optional prebuilt
            ``thermolm_jax.models.thrml_quadratic.THRMLIsingSampler``. REQUIRED
            for the THRML path inside ``jit``/``grad`` (the fallback builds the
            edge structure from the concrete mask, which fails on tracers).
            Build it once at setup and reuse it.

    Returns:
        (final_state, info)
    """
    is_quadratic = hasattr(energy_fn, "J") and hasattr(energy_fn, "h")

    if use_thrml and is_quadratic:
        if thrml_sampler is None:
            # Eager-only fallback: derives structure from the concrete mask.
            from thermolm_jax.models.thrml_quadratic import THRMLIsingSampler

            thrml_sampler = THRMLIsingSampler(np.asarray(energy_fn.connectivity_mask))
        return thrml_sampler.sample(
            energy_fn.J, energy_fn.h, energy_fn.beta, init_state, key, n_steps, temperature
        )

    if is_quadratic:
        if color_masks is None:
            colors = greedy_coloring(energy_fn.connectivity_mask)
            color_masks = color_masks_from_colors(colors, init_state.shape[-1], init_state.dtype)
        return _chromatic_gibbs_quadratic(
            energy_fn, init_state, n_steps, key, temperature, color_masks
        )

    return _gibbs_single_site(energy_fn, init_state, n_steps, key, temperature)


def _chromatic_gibbs_quadratic(
    ebm,
    init_state: jnp.ndarray,
    n_steps: int,
    key: jax.random.PRNGKey,
    temperature: float,
    color_masks: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    """Vectorised chromatic block Gibbs for a quadratic Ising EBM."""
    J_full = _full_couplings(ebm)  # (n, n) symmetric, zero diagonal
    h = ebm.h
    beta = ebm.beta

    n_colors = color_masks.shape[0]  # static under jit

    def sweep(state, key):
        for c in range(n_colors):
            key, sub = jax.random.split(key)
            local_field = state @ J_full + h  # (batch, n)
            p_plus = jax.nn.sigmoid(2.0 * beta * local_field / temperature)
            u = jax.random.uniform(sub, state.shape)
            proposal = jnp.where(u < p_plus, 1.0, -1.0)
            cmask = color_masks[c]  # (n,)
            state = state * (1.0 - cmask) + proposal * cmask
        return state, key

    def scan_fn(carry, _):
        state, key = carry
        state, key = sweep(state, key)
        return (state, key), None

    (final_state, _), _ = jax.lax.scan(
        scan_fn, (init_state.astype(jnp.float32), key), None, length=n_steps
    )

    info = {
        "n_steps": n_steps,
        "temperature": temperature,
        "n_colors": n_colors,
        "sampling_method": "JAX chromatic block Gibbs (quadratic)",
    }
    return final_state, info


def _gibbs_single_site(
    energy_fn: Callable,
    init_state: jnp.ndarray,
    n_steps: int,
    key: jax.random.PRNGKey,
    temperature: float,
) -> Tuple[jnp.ndarray, dict]:
    """Systematic single-site Gibbs for a generic energy function.

    Updates one spin at a time using the exact conditional
    ``P(x_i=+1) = σ((E(x_i=-1) - E(x_i=+1)) / T)``. Correct for any graph; no
    colouring assumptions are made.
    """
    batch_size, n_vars = init_state.shape

    def update_site(state, idx, key):
        x_plus = state.at[..., idx].set(1.0)
        x_minus = state.at[..., idx].set(-1.0)
        energy_diff = energy_fn(x_minus) - energy_fn(x_plus)  # (batch,)
        p_plus = jax.nn.sigmoid(energy_diff / temperature)
        u = jax.random.uniform(key, (batch_size,))
        new_vals = jnp.where(u < p_plus, 1.0, -1.0)
        return state.at[..., idx].set(new_vals)

    def sweep(state, key):
        def body(state, idx):
            k = jax.random.fold_in(key, idx)
            return update_site(state, idx, k), None

        state, _ = jax.lax.scan(body, state, jnp.arange(n_vars))
        return state

    def scan_fn(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        state = sweep(state, sub)
        return (state, key), None

    (final_state, _), _ = jax.lax.scan(
        scan_fn, (init_state.astype(jnp.float32), key), None, length=n_steps
    )

    info = {
        "n_steps": n_steps,
        "temperature": temperature,
        "sampling_method": "JAX systematic single-site Gibbs (generic)",
    }
    return final_state, info
