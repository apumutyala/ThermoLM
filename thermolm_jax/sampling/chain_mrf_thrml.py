"""
Sample a linear-chain categorical MRF on THRML (the TSU-compatible path).

The chain CRF defined in ``thermolm_jax.models.chain_crf`` (unary (L,V) +
nearest-neighbour pairwise (L-1,V,V)) is turned into a THRML factor graph and
sampled with block (chromatic) Gibbs. The chain is 2-coloured by even/odd
position, so each colour class is an independent set — the requirement for a
correct parallel block update (this fixes the single-block bug in the old
``thrml_discrete.py``).

THRML's categorical factor energy is ``-Σ weights[gathered]``, i.e. it samples
``p ∝ exp(Σ unary + Σ pairwise)`` — exactly the chain-CRF score — so the same
potential tensors are passed through unchanged (temperature folded in as / T).
"""

import jax
import jax.numpy as jnp

from thrml import (
    Block,
    BlockGibbsSpec,
    CategoricalNode,
    FactorSamplingProgram,
    SamplingSchedule,
    sample_states,
)
from thrml.models import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    SquareCategoricalEBMFactor,
)


def _build_chain_program(unary, pairwise, n_levels, temperature):
    """Build (program, free_blocks, even_idx, odd_idx) for one chain model."""
    L = unary.shape[0]
    u = unary / temperature
    w = pairwise / temperature

    nodes = [CategoricalNode() for _ in range(L)]
    factors = [CategoricalEBMFactor(node_groups=[Block(nodes)], weights=u)]
    if L > 1:
        factors.append(
            SquareCategoricalEBMFactor(
                node_groups=[Block(nodes[:-1]), Block(nodes[1:])], weights=w
            )
        )

    even_idx = list(range(0, L, 2))
    odd_idx = list(range(1, L, 2))
    # Drop the odd block when L == 1 — THRML's BlockSpec rejects empty blocks.
    free_blocks = [Block([nodes[i] for i in even_idx])]
    if odd_idx:
        free_blocks.append(Block([nodes[i] for i in odd_idx]))

    node_sds = {CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8)}
    spec = BlockGibbsSpec(free_blocks, [], node_sds)
    program = FactorSamplingProgram(
        spec,
        [CategoricalGibbsConditional(n_categories=n_levels) for _ in free_blocks],
        factors,
        [],
    )
    return program, free_blocks, even_idx, odd_idx


def _reassemble(samp, even_idx, odd_idx, L):
    out = jnp.zeros((L,), dtype=jnp.int32)
    out = out.at[jnp.array(even_idx)].set(samp[0][-1].astype(jnp.int32))
    if odd_idx:
        out = out.at[jnp.array(odd_idx)].set(samp[1][-1].astype(jnp.int32))
    return out


def sample_chain_thrml_single(unary, pairwise, key, n_chains=512, n_warmup=100, temperature=1.0):
    """Sample `n_chains` independent chains from ONE chain-CRF model.

    Args:
        unary: (L, V) unary potentials.
        pairwise: (L-1, V, V) pairwise potentials.
        key, n_chains, n_warmup, temperature: sampling controls.

    Returns:
        (n_chains, L) int32 samples.
    """
    L, V = unary.shape
    program, free_blocks, even_idx, odd_idx = _build_chain_program(
        unary, pairwise, V, temperature
    )
    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=1, steps_per_sample=1)

    def one(key_i):
        ks = jax.random.split(key_i, len(free_blocks) + 1)
        init = [
            jax.random.randint(ks[bi], (len(b.nodes),), 0, V, dtype=jnp.uint8)
            for bi, b in enumerate(free_blocks)
        ]
        samp = sample_states(ks[-1], program, schedule, init, [], free_blocks)
        return _reassemble(samp, even_idx, odd_idx, L)

    return jax.vmap(one)(jax.random.split(key, n_chains))


def sample_chain_thrml(unary, pairwise, key, n_warmup=100, temperature=1.0):
    """Sample one chain per batch element, each with its OWN potentials (for generation).

    Args:
        unary: (B, L, V) per-example unary potentials.
        pairwise: (B, L-1, V, V) per-example pairwise potentials.

    Returns:
        (B, L) int32 samples.
    """
    B, L, V = unary.shape

    def one(u_i, w_i, key_i):
        program, free_blocks, even_idx, odd_idx = _build_chain_program(
            u_i, w_i, V, temperature
        )
        schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=1, steps_per_sample=1)
        ks = jax.random.split(key_i, len(free_blocks) + 1)
        init = [
            jax.random.randint(ks[bi], (len(b.nodes),), 0, V, dtype=jnp.uint8)
            for bi, b in enumerate(free_blocks)
        ]
        samp = sample_states(ks[-1], program, schedule, init, [], free_blocks)
        return _reassemble(samp, even_idx, odd_idx, L)

    return jax.vmap(one)(unary, pairwise, jax.random.split(key, B))
