"""
Linear-chain CRF (categorical Markov random field) — exact inference in JAX.

For a sequence x = (x_0, ..., x_{L-1}) with x_i in {0, ..., V-1}, the model is

    score(x) = Σ_i u_i[x_i] + Σ_i W_i[x_i, x_{i+1}]
    p(x)     = exp(score(x)) / Z

with unary log-potentials `unary` of shape (L, V) and nearest-neighbour pairwise
log-potentials `pairwise` of shape (L-1, V, V). (Energy E = -score, matching
`factor_weight_network.compute_energy_from_weights` and THRML's
`-Σ weights[gathered]` factor convention, so the same tensors can be handed to
THRML's categorical factors unchanged.)

Because the graph is a chain, everything below is exact and differentiable:
- `chain_log_partition`  — forward algorithm, O(L·V²)
- `chain_log_likelihood` — score - log Z  (the M1 training objective)
- `chain_marginals`      — forward–backward node marginals
- `chain_sample`         — exact forward-filter backward-sample (FFBS)

All functions operate on a single sequence; use `jax.vmap` for batches.
References: Lafferty, McCallum & Pereira (2001); Rabiner (1989).
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _forward(unary: jnp.ndarray, pairwise: jnp.ndarray) -> jnp.ndarray:
    """Forward messages α_i[v] = log Σ_{x_0..x_i, x_i=v} exp(partial score). Shape (L, V)."""
    L = unary.shape[0]

    def step(a_prev, i):
        a_i = unary[i] + logsumexp(a_prev[:, None] + pairwise[i - 1], axis=0)
        return a_i, a_i

    a0 = unary[0]
    _, rest = jax.lax.scan(step, a0, jnp.arange(1, L))
    return jnp.concatenate([a0[None], rest], axis=0)


def _backward(unary: jnp.ndarray, pairwise: jnp.ndarray) -> jnp.ndarray:
    """Backward messages β_i[v] = log Σ_{x_{i+1}..} exp(rest | x_i=v). Shape (L, V)."""
    L, V = unary.shape
    bL = jnp.zeros(V)

    def step(b_next, i):
        b_i = logsumexp(pairwise[i] + (unary[i + 1] + b_next)[None, :], axis=1)
        return b_i, b_i

    idx = jnp.arange(L - 2, -1, -1)
    _, rest = jax.lax.scan(step, bL, idx)        # order i = L-2, ..., 0
    betas_fwd = jnp.flip(rest, axis=0)           # order i = 0, ..., L-2
    return jnp.concatenate([betas_fwd, bL[None]], axis=0)


def chain_log_partition(unary: jnp.ndarray, pairwise: jnp.ndarray) -> jnp.ndarray:
    """Exact log partition function log Z (scalar)."""
    return logsumexp(_forward(unary, pairwise)[-1])


def chain_score(x: jnp.ndarray, unary: jnp.ndarray, pairwise: jnp.ndarray) -> jnp.ndarray:
    """Unnormalised log-score of a configuration x (shape (L,), int)."""
    L = unary.shape[0]
    u = unary[jnp.arange(L), x].sum()
    pw = pairwise[jnp.arange(L - 1), x[:-1], x[1:]].sum()
    return u + pw


def chain_log_likelihood(x: jnp.ndarray, unary: jnp.ndarray, pairwise: jnp.ndarray) -> jnp.ndarray:
    """Exact log p(x) = score(x) - log Z (scalar). Differentiable in the potentials."""
    return chain_score(x, unary, pairwise) - chain_log_partition(unary, pairwise)


def chain_marginals(unary: jnp.ndarray, pairwise: jnp.ndarray) -> jnp.ndarray:
    """Exact node log-marginals log p(x_i = v), shape (L, V)."""
    a = _forward(unary, pairwise)
    b = _backward(unary, pairwise)
    logZ = logsumexp(a[-1])
    return a + b - logZ


def chain_sample(key: jax.Array, unary: jnp.ndarray, pairwise: jnp.ndarray) -> jnp.ndarray:
    """Exact joint sample x ~ p(x) via forward-filter backward-sample. Shape (L,), int32."""
    L = unary.shape[0]
    a = _forward(unary, pairwise)

    k_last, k_body = jax.random.split(key)
    x_last = jax.random.categorical(k_last, a[-1]).astype(jnp.int32)

    def step(x_next, inp):
        i, k = inp
        logits = a[i] + pairwise[i][:, x_next]
        x_i = jax.random.categorical(k, logits).astype(jnp.int32)
        return x_i, x_i

    idx = jnp.arange(L - 2, -1, -1)
    keys = jax.random.split(k_body, max(L - 1, 1))[: L - 1]
    _, xs_rev = jax.lax.scan(step, x_last, (idx, keys))
    xs_fwd = jnp.flip(xs_rev, axis=0)
    return jnp.concatenate([xs_fwd, x_last[None]], axis=0)
