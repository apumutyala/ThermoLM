"""
Correctness tests for the linear-chain CRF and its THRML sampler.

Everything is checked against brute-force enumeration on tiny (L, V), so these
are exact-correctness tests, not shape checks.
"""

import itertools

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from thermolm_jax.models.chain_crf import (
    chain_log_partition,
    chain_log_likelihood,
    chain_marginals,
    chain_sample,
)
from thermolm_jax.sampling.chain_mrf_thrml import sample_chain_thrml_single
from thermolm_jax.sampling.chain_gibbs_jax import sample_chain_jax_gibbs

pytestmark = pytest.mark.unit

L, V = 4, 3


def _random_potentials(seed=0, scale=1.0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    unary = jax.random.normal(k1, (L, V)) * scale
    pairwise = jax.random.normal(k2, (L - 1, V, V)) * scale
    return unary, pairwise


def _enumerate(unary, pairwise):
    u = np.asarray(unary)
    w = np.asarray(pairwise)
    configs = np.array(list(itertools.product(range(V), repeat=L)))  # (V^L, L)
    scores = u[np.arange(L), configs].sum(1)
    for i in range(L - 1):
        scores += w[i, configs[:, i], configs[:, i + 1]]
    logZ = np.log(np.exp(scores - scores.max()).sum()) + scores.max()
    probs = np.exp(scores - logZ)
    # node marginals (L, V)
    marg = np.zeros((L, V))
    for i in range(L):
        for v in range(V):
            marg[i, v] = probs[configs[:, i] == v].sum()
    return logZ, probs, configs, marg


def test_log_partition_matches_enumeration():
    unary, pairwise = _random_potentials(seed=1, scale=1.2)
    logZ_exact, _, _, _ = _enumerate(unary, pairwise)
    logZ = float(chain_log_partition(unary, pairwise))
    assert abs(logZ - logZ_exact) < 1e-4


def test_log_likelihood_matches_enumeration():
    unary, pairwise = _random_potentials(seed=2, scale=1.0)
    _, probs, configs, _ = _enumerate(unary, pairwise)
    for idx in [0, 5, 20, 50]:
        x = jnp.asarray(configs[idx])
        ll = float(chain_log_likelihood(x, unary, pairwise))
        assert abs(ll - float(np.log(probs[idx]))) < 1e-4


def test_marginals_match_enumeration():
    unary, pairwise = _random_potentials(seed=3, scale=1.0)
    _, _, _, marg_exact = _enumerate(unary, pairwise)
    marg = np.asarray(jnp.exp(chain_marginals(unary, pairwise)))
    assert np.abs(marg - marg_exact).max() < 1e-4


def test_ffbs_samples_match_marginals():
    unary, pairwise = _random_potentials(seed=4, scale=1.0)
    _, _, _, marg_exact = _enumerate(unary, pairwise)
    keys = jax.random.split(jax.random.PRNGKey(0), 20000)
    samples = jax.vmap(lambda k: chain_sample(k, unary, pairwise))(keys)
    samples = np.asarray(samples)
    emp = np.stack([(samples == v).mean(0) for v in range(V)], axis=1)  # (L, V)
    assert np.abs(emp - marg_exact).max() < 0.03


def test_thrml_samples_match_exact_marginals():
    unary, pairwise = _random_potentials(seed=5, scale=1.0)
    _, _, _, marg_exact = _enumerate(unary, pairwise)
    samples = np.asarray(
        sample_chain_thrml_single(
            unary, pairwise, jax.random.PRNGKey(0), n_chains=6000, n_warmup=200
        )
    )
    emp = np.stack([(samples == v).mean(0) for v in range(V)], axis=1)
    assert np.abs(emp - marg_exact).max() < 0.05


def test_jax_chromatic_gibbs_matches_exact_marginals():
    """JAX chromatic block Gibbs is the fair GPU baseline for THRML."""
    unary, pairwise = _random_potentials(seed=6, scale=1.0)
    _, _, _, marg_exact = _enumerate(unary, pairwise)
    samples = np.asarray(
        sample_chain_jax_gibbs(
            unary, pairwise, jax.random.PRNGKey(0), n_chains=6000, n_warmup=200
        )
    )
    emp = np.stack([(samples == v).mean(0) for v in range(V)], axis=1)
    assert np.abs(emp - marg_exact).max() < 0.05
