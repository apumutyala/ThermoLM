"""
JAX chromatic (block) Gibbs sampler for linear-chain categorical CRFs.

The chain CRF is a bipartite graph: even positions are independent of each other
given odd positions, and vice versa. This sampler updates a whole colour class in
parallel — the same graph-colouring idea that makes sparse EBMs fast on GPUs and
that underlies THRML's TSU block-Gibbs programs.

For a chain with unary log-potentials u(i,v) and pairwise log-potentials w(i,v,v'),
the conditional for position i is

    p(x_i = v | x_{-i}) ∝ exp( u[i,v] + w[i-1, x_{i-1}, v] + w[i, v, x_{i+1}] )

with the w terms omitted at the boundaries. Even and odd positions are sampled
alternately; within one colour class all updates are independent and
parallelisable.

This is the fair GPU baseline for the THRML block-Gibbs sampler in
``chain_mrf_thrml.py``: both implement the same algorithm on the same model, but
this version runs on the GPU through JAX while the THRML version uses THRML's
SamplingProgram / sample_states (the hardware-shaped abstraction).
"""

import jax
import jax.numpy as jnp


def sample_chain_jax_gibbs(
    unary: jnp.ndarray,
    pairwise: jnp.ndarray,
    key: jax.Array,
    n_chains: int = 512,
    n_warmup: int = 100,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Sample ``n_chains`` independent chains via JAX chromatic block Gibbs.

    Args:
        unary: (L, V) unary log-potentials.
        pairwise: (L-1, V, V) pairwise log-potentials.
        key: PRNG key.
        n_chains: number of independent chains to draw.
        n_warmup: number of full even/odd sweeps.
        temperature: temperature scaling (potentials are divided by T).

    Returns:
        (n_chains, L) int32 samples.
    """
    L, V = unary.shape
    u = unary / temperature
    w = pairwise / temperature

    even_mask = jnp.arange(L) % 2 == 0
    odd_mask = ~even_mask

    def sample_color(state, mask, k):
        # state: (L,) int32; mask: (L,) bool
        left = jnp.zeros((L, V))
        left = left.at[1:].set(w[jnp.arange(L - 1), state[:-1], :])
        right = jnp.zeros((L, V))
        right = right.at[:-1].set(w[jnp.arange(L - 1), :, state[1:]])
        logits = u + left + right
        proposal = jax.random.categorical(k, logits, axis=1).astype(jnp.int32)
        return jnp.where(mask, proposal, state)

    def sweep(state, key):
        k1, k2 = jax.random.split(key)
        state = sample_color(state, even_mask, k1)
        state = sample_color(state, odd_mask, k2)
        return state, None

    def one(key_i):
        init = jax.random.randint(key_i, (L,), 0, V, dtype=jnp.int32)
        final, _ = jax.lax.scan(sweep, init, jax.random.split(key_i, n_warmup))
        return final

    return jax.vmap(one)(jax.random.split(key, n_chains))
