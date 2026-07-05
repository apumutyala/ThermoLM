"""
Correctness tests for THRML-native fully-visible ML training.

The gradient test is THRML's own documented validation recipe: for a model
small enough to enumerate, the Monte-Carlo two-term KL gradient from
``estimate_kl_grad`` must match the exact gradient

    grad_b = -beta ( <s_i>_data     - <s_i>_model )
    grad_w = -beta ( <s_i s_j>_data - <s_i s_j>_model )

where the model expectations are computed by exact enumeration. For the
fully-visible path the positive term is exact by construction (v0.1.3+
computes it directly from the data batch), so the only MC error is the
negative phase.
"""

import itertools

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from thrml import Block, SamplingSchedule
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init

from thermolm_jax.models.quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from thermolm_jax.models.thrml_quadratic import THRMLIsingSampler
from thermolm_jax.training.thrml_ml import (
    fit_ising_ml,
    make_kl_grad_step,
    params_from_ebm,
    params_to_ebm,
)

pytestmark = pytest.mark.unit


def _chain_ebm(n=6, init_scale=0.6, seed=0):
    key = jax.random.PRNGKey(seed)
    ebm = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=init_scale), key)
    mask = np.zeros((n, n), bool)
    for i in range(n - 1):
        mask[i, i + 1] = mask[i + 1, i] = True
    return ebm.set_connectivity(jnp.asarray(mask))


def _exact_model_moments(ebm, sampler, n):
    """Exact <s_i> and per-edge <s_i s_j> under the model by enumeration."""
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = np.asarray(ebm(jnp.asarray(states)))
    w = np.exp(-(E - E.min()))
    w /= w.sum()
    mag = (w[:, None] * states).sum(0)                                  # (n,)
    ei, ej = sampler._ei, sampler._ej
    pair = (w[:, None] * (states[:, ei] * states[:, ej])).sum(0)        # (n_edges,)
    return mag, pair


def test_kl_grad_matches_exact_two_term_gradient():
    """MC estimate_kl_grad ~= exact -beta(data_moments - model_moments)."""
    n = 6
    model = _chain_ebm(n=n, init_scale=0.6, seed=1)
    sampler = THRMLIsingSampler(np.asarray(model.connectivity_mask))
    params = params_from_ebm(sampler, model)

    # Fixed data batch (any distribution works: the positive term is exact).
    key = jax.random.PRNGKey(2)
    data_pm1 = (jax.random.randint(key, (64, n), 0, 2) * 2 - 1).astype(jnp.float32)
    data_mag = np.asarray(data_pm1).mean(0)
    data_pair = np.asarray(
        np.asarray(data_pm1)[:, sampler._ei] * np.asarray(data_pm1)[:, sampler._ej]
    ).mean(0)

    mag_model, pair_model = _exact_model_moments(model, sampler, n)
    beta = 1.0
    expected_b = -beta * (data_mag - mag_model)
    expected_w = -beta * (data_pair - pair_model)

    # Monte-Carlo gradient with a generous negative-phase schedule.
    ebm = IsingEBM(sampler.nodes, sampler.edges, params["biases"],
                   params["edge_weights"], jnp.asarray(beta))
    spec = IsingTrainingSpec(
        ebm, [Block(sampler.nodes)], [], [], sampler.free_blocks,
        SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=0),
        SamplingSchedule(n_warmup=200, n_samples=50, steps_per_sample=2),
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    init_neg = hinton_init(k1, ebm, sampler.free_blocks, (512,))
    grad_w, grad_b, _, _ = estimate_kl_grad(
        k2, spec, sampler.nodes, sampler.edges, [data_pm1 > 0], [], [], init_neg
    )

    assert np.abs(np.asarray(grad_b) - expected_b).max() < 0.05, (
        np.asarray(grad_b), expected_b)
    assert np.abs(np.asarray(grad_w) - expected_w).max() < 0.05, (
        np.asarray(grad_w), expected_w)


def test_params_roundtrip():
    """params_from_ebm / params_to_ebm preserve the effective couplings."""
    model = _chain_ebm(n=6, init_scale=0.7, seed=4)
    sampler = THRMLIsingSampler(np.asarray(model.connectivity_mask))
    params = params_from_ebm(sampler, model)
    model2 = params_to_ebm(sampler, params, model)
    # Same energies on random states => same distribution.
    key = jax.random.PRNGKey(5)
    x = (jax.random.randint(key, (32, 6), 0, 2) * 2 - 1).astype(jnp.float32)
    assert np.allclose(np.asarray(model(x)), np.asarray(model2(x)), atol=1e-5)


def test_thrml_ml_training_recovers_teacher_moments():
    """Fully-visible THRML-native ML drives model moments to the data's."""
    n = 6
    key = jax.random.PRNGKey(20)
    teacher = _chain_ebm(n=n, init_scale=0.8, seed=99)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = np.asarray(teacher(jnp.asarray(states)))
    w = np.exp(-(E - E.min()))
    w /= w.sum()
    key, kd = jax.random.split(key)
    idx = np.asarray(jax.random.choice(kd, len(states), shape=(4000,), p=jnp.asarray(w)))
    data = jnp.asarray(states[idx])
    data_mag = np.asarray(data).mean(0)

    student = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=0.01),
                           jax.random.PRNGKey(0))
    student = student.set_connectivity(teacher.connectivity_mask)

    student, sampler, history = fit_ising_ml(
        student, data, key, n_iters=120, batch_size=256, lr=0.05,
        n_chains_neg=256,
        schedule_negative=SamplingSchedule(n_warmup=30, n_samples=15, steps_per_sample=2),
    )

    # The exact-vs-sampled moment gap is the training signal; it should shrink.
    assert np.mean(history[-10:]) < np.mean(history[:10])

    # And the trained model's exact moments should match the data moments.
    mag_model, _ = _exact_model_moments(student, sampler, n)
    assert np.abs(mag_model - data_mag).max() < 0.15, (mag_model, data_mag)
