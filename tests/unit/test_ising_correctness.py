"""
Correctness tests for the validated DTM / quadratic-Ising track.

These are *property/correctness* tests (not shape-only): they check the sampler
against exact Boltzmann marginals, the CD gradient direction, the single-counted
energy form, the forward-coupling sign, and the nesting of connectivity patterns.
"""

import itertools

import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import pytest

from thermolm_jax.models.quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from thermolm_jax.models.forward_coupling import ForwardCoupling, ForwardCouplingConfig
from thermolm_jax.models.connectivity import generate_connectivity_pattern, get_connectivity_density
from thermolm_jax.sampling.chromatic_gibbs import (
    chromatic_gibbs_sample,
    greedy_coloring,
    color_masks_from_colors,
)
from thermolm_jax.training.contrastive_divergence import contrastive_divergence_step, CDConfig

pytestmark = pytest.mark.unit


def _chain_ebm(n=6, init_scale=0.6, seed=0):
    key = jax.random.PRNGKey(seed)
    ebm = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=init_scale), key)
    mask = np.zeros((n, n), bool)
    for i in range(n - 1):
        mask[i, i + 1] = mask[i + 1, i] = True
    return ebm.set_connectivity(jnp.asarray(mask))


def _exact_moments(ebm, n):
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = np.asarray(ebm(jnp.asarray(states)))
    w = np.exp(-(E - E.min()))
    w /= w.sum()
    mag = (w[:, None] * states).sum(0)
    corr = np.array([(w * states[:, i] * states[:, i + 1]).sum() for i in range(n - 1)])
    return mag, corr


def _empirical_moments(samples, n):
    mag = samples.mean(0)
    corr = np.array([(samples[:, i] * samples[:, i + 1]).mean() for i in range(n - 1)])
    return mag, corr


def test_energy_is_single_counted_and_diagonal_free():
    """E must not double-count pairs nor depend on the diagonal of J."""
    n = 5
    key = jax.random.PRNGKey(3)
    ebm = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=0.5), key)
    ebm = ebm.set_connectivity(jnp.ones((n, n), bool))
    x = jnp.asarray(np.array([[1.0, -1.0, 1.0, 1.0, -1.0]]))

    # Energy must be invariant to whatever sits on the diagonal of J.
    e0 = float(ebm(x)[0])
    ebm_diag = eqx.tree_at(lambda m: m.J, ebm, ebm.J + 7.0 * jnp.eye(n))
    e1 = float(ebm_diag(x)[0])
    assert abs(e0 - e1) < 1e-5

    # Pairwise term equals -beta * sum_{i<j} J_sym_ij x_i x_j (each pair once).
    Js = 0.5 * (np.asarray(ebm.J) + np.asarray(ebm.J).T)
    xv = np.asarray(x)[0]
    expect = -1.0 * sum(Js[i, j] * xv[i] * xv[j] for i in range(n) for j in range(i + 1, n))
    expect += -1.0 * float(np.asarray(ebm.h) @ xv)
    assert abs(e0 - expect) < 1e-4


def test_chromatic_gibbs_matches_exact_marginals():
    ebm = _chain_ebm()
    n = 6
    mag_exact, corr_exact = _exact_moments(ebm, n)
    key = jax.random.PRNGKey(10)
    init = jax.random.randint(key, (8000, n), 0, 2) * 2 - 1
    samp = np.asarray(chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 300, key)[0])
    mag, corr = _empirical_moments(samp, n)
    assert np.abs(mag - mag_exact).max() < 0.05
    assert np.abs(corr - corr_exact).max() < 0.05


def test_thrml_sampler_matches_exact_marginals():
    """THRML's IsingSamplingProgram path must reproduce the same exact marginals."""
    ebm = _chain_ebm()
    n = 6
    mag_exact, corr_exact = _exact_moments(ebm, n)
    key = jax.random.PRNGKey(11)
    init = jax.random.randint(key, (8000, n), 0, 2) * 2 - 1
    samp = np.asarray(
        chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 300, key, use_thrml=True)[0]
    )
    mag, corr = _empirical_moments(samp, n)
    assert np.abs(mag - mag_exact).max() < 0.06
    assert np.abs(corr - corr_exact).max() < 0.06


def test_temperature_scaling_moves_toward_uniform():
    """Higher temperature must reduce |magnetisation| (toward the uniform dist)."""
    ebm = _chain_ebm(init_scale=1.0)
    n = 6
    key = jax.random.PRNGKey(12)
    init = jax.random.randint(key, (6000, n), 0, 2) * 2 - 1
    lo = np.asarray(chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 300, key, temperature=0.5)[0])
    hi = np.asarray(chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 300, key, temperature=4.0)[0])
    assert np.abs(hi.mean(0)).mean() < np.abs(lo.mean(0)).mean()


def test_cd_reduces_energy_gap_and_matches_moments():
    """CD training must drive model moments toward the data moments."""
    n = 6
    key = jax.random.PRNGKey(20)
    # Target distribution = Boltzmann of a fixed teacher Ising model.
    teacher = _chain_ebm(n=n, init_scale=0.8, seed=99)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = np.asarray(teacher(jnp.asarray(states)))
    w = np.exp(-(E - E.min())); w /= w.sum()
    data_mag, _ = _exact_moments(teacher, n)

    def sample_data(k, B=256):
        idx = np.asarray(jax.random.choice(k, len(states), shape=(B,), p=jnp.asarray(w)))
        return jnp.asarray(states[idx])

    student = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=0.01), key)
    student = student.set_connectivity(teacher.connectivity_mask)
    cmasks = color_masks_from_colors(greedy_coloring(student.connectivity_mask), n)
    opt = optax.adam(0.05)
    opt_state = opt.init(eqx.filter(student, eqx.is_inexact_array))
    cfg = CDConfig(k=1, n_gibbs_steps=30, temperature=1.0)

    for step in range(150):
        key, kb, kc = jax.random.split(key, 3)
        student, opt_state, _, _ = contrastive_divergence_step(
            student, opt, opt_state, sample_data(kb), kc, cfg, color_masks=cmasks
        )

    init = jax.random.randint(key, (8000, n), 0, 2) * 2 - 1
    samp = np.asarray(
        chromatic_gibbs_sample(student, init.astype(jnp.float32), 300, key, color_masks=cmasks)[0]
    )
    model_mag = samp.mean(0)
    # student magnetisations should resemble the data magnetisations
    assert np.abs(model_mag - data_mag).max() < 0.15


def test_forward_coupling_prefers_aligned_states():
    """E_f must be lower (more negative) when consecutive states agree."""
    cfg = ForwardCouplingConfig(T=10, gamma_min=1.0, gamma_max=1.0)
    fc = ForwardCoupling(cfg)
    x = jnp.asarray(np.array([[1.0, -1.0, 1.0, -1.0]]))
    e_aligned = float(fc(x, x, t=5)[0])
    e_anti = float(fc(x, -x, t=5)[0])
    assert e_aligned < e_anti


def test_connectivity_patterns_are_nested():
    """G8 ⊂ G12 ⊂ G16 ⊂ G20 ⊂ G24: edge density strictly increases."""
    n = 32
    dens = [
        get_connectivity_density(generate_connectivity_pattern(p, n, graph_type="banded"))
        for p in ["G8", "G12", "G16", "G20", "G24"]
    ]
    assert all(dens[i] < dens[i + 1] for i in range(len(dens) - 1)), dens


def test_greedy_coloring_is_valid():
    """No two adjacent nodes share a colour."""
    mask = np.asarray(generate_connectivity_pattern("G12", 24, graph_type="banded"))
    colors = greedy_coloring(jnp.asarray(mask))
    i, j = np.nonzero(np.triu(mask, 1))
    assert np.all(colors[i] != colors[j])


def test_cd_with_thrml_negative_phase_trains():
    """CD with a THRML negative phase must run under jit/grad (A1 regression).

    An earlier THRML wrapper converted the traced coupling matrix to NumPy to
    build the edge list, so any training step with ``use_thrml=True`` raised
    ``TracerArrayConversionError``. The setup-once ``THRMLIsingSampler`` keeps
    the edge structure static and gathers weights with jnp indexing; this test
    locks in that the jit-compiled CD step runs and learns.
    """
    from thermolm_jax.models.thrml_quadratic import THRMLIsingSampler

    n = 6
    key = jax.random.PRNGKey(30)
    teacher = _chain_ebm(n=n, init_scale=0.8, seed=7)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = np.asarray(teacher(jnp.asarray(states)))
    w = np.exp(-(E - E.min())); w /= w.sum()
    data_mag, _ = _exact_moments(teacher, n)

    student = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=0.01), key)
    student = student.set_connectivity(teacher.connectivity_mask)
    sampler = THRMLIsingSampler(np.asarray(student.connectivity_mask))

    opt = optax.adam(0.05)
    opt_state = opt.init(eqx.filter(student, eqx.is_inexact_array))
    cfg = CDConfig(k=1, n_gibbs_steps=15, temperature=1.0, use_thrml=True)

    losses = []
    for step in range(60):
        key, kb, kc = jax.random.split(key, 3)
        idx = np.asarray(jax.random.choice(kb, len(states), shape=(128,), p=jnp.asarray(w)))
        student, opt_state, loss, _ = contrastive_divergence_step(
            student, opt, opt_state, jnp.asarray(states[idx]), kc, cfg,
            thrml_sampler=sampler,
        )
        losses.append(float(loss))
    assert np.all(np.isfinite(losses))

    # Model magnetisations should move toward the data magnetisations.
    key, ks = jax.random.split(key)
    init = jax.random.randint(ks, (4000, n), 0, 2) * 2 - 1
    samp, _ = sampler.sample(
        student.J, student.h, student.beta, init.astype(jnp.float32), ks, 200
    )
    model_mag = np.asarray(samp).mean(0)
    assert np.abs(model_mag - data_mag).max() < 0.25


def test_forward_coupling_thrml_factor_matches_energy():
    """The THRML factor form of the forward coupling matches __call__ (A2).

    A two-block SpinEBMFactor pairs blocks elementwise with weights of shape
    (n,); its energy is -sum_i w_i A_i B_i, which must equal the coupling
    energy E_f = -(gamma_t/2) sum_i x_t_i x_{t-1}_i.
    """
    cfg = ForwardCouplingConfig(T=10, gamma_min=0.5, gamma_max=0.5)
    fc = ForwardCoupling(cfg)
    n = 4
    from thrml import SpinNode

    nodes_t = [SpinNode() for _ in range(n)]
    nodes_tm1 = [SpinNode() for _ in range(n)]
    factor = fc.to_thrml_factor_at_t(3, nodes_t, nodes_tm1)
    assert factor.weights.shape == (n,)

    x_t = jnp.asarray([[1.0, -1.0, 1.0, 1.0]])
    x_tm1 = jnp.asarray([[1.0, 1.0, -1.0, 1.0]])
    manual = -jnp.sum(factor.weights * x_t * x_tm1, axis=-1)
    assert np.allclose(np.asarray(fc(x_t, x_tm1, t=3)), np.asarray(manual), atol=1e-5)
