"""
THRML-native maximum-likelihood training for fully-visible quadratic Ising EBMs.

Our quadratic EBM has no latent variables, which makes it exactly the
"fully visible" case THRML v0.1.3+ supports natively: ``estimate_kl_grad``
with ``init_state_positive=[]`` computes the positive-phase moments
**exactly from the data batch** (zero variance, no MCMC), while the negative
phase is sampled by THRML's block-Gibbs ``IsingSamplingProgram`` — the
TSU-compatible path. The resulting update is the true two-term ML/KL gradient

    grad_b = -beta ( <s_i>_data      - <s_i>_model )
    grad_W = -beta ( <s_i s_j>_data  - <s_i s_j>_model )

with only the model expectation estimated by sampling.

Contrast with ``contrastive_divergence``: CD-k initialises the negative chain
AT the data and runs a few sweeps (biased, low-variance, fast); this trainer
initialises negative chains from the model (``hinton_init``) and estimates
the model expectation directly (unbiased in the long-chain limit). Both share
the same energy convention and the same ``THRMLIsingSampler`` static
structure.

JAX pattern (see the THRML docs / skill): the training step is a plain
``jax.jit`` **closing over static structure** (nodes, edges, blocks,
schedules) and threading only arrays (params, data, keys). ``IsingEBM`` and
``IsingTrainingSpec`` are rebuilt inside the traced function — construction
is free at trace time — because their pytree leaves include ``SpinNode``
objects, which cannot be traced arguments.
"""

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from thrml import Block, SamplingSchedule
from thrml.models import IsingEBM, IsingTrainingSpec, estimate_kl_grad, hinton_init

from thermolm_jax.models.thrml_quadratic import THRMLIsingSampler

# The positive phase never runs for a fully-visible model (its moments are
# computed exactly from the data), but IsingTrainingSpec still wants a
# schedule object. This mirrors THRML's own fully-visible test.
_DUMMY_POSITIVE_SCHEDULE = SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=0)


def params_from_ebm(sampler: THRMLIsingSampler, model) -> dict:
    """Extract THRML-shaped parameters {biases (n,), edge_weights (n_edges,)}.

    Edge weights follow the shared convention: w_edge = 0.5*(J_ij + J_ji),
    exactly what ``QuadraticEBM._effective_couplings`` uses.
    """
    return {
        "biases": model.h,
        "edge_weights": sampler.edge_weights(model.J),
    }


def params_to_ebm(sampler: THRMLIsingSampler, params: dict, model):
    """Write {biases, edge_weights} back into a QuadraticEBM (new module).

    Both triangles of J are set to the edge weight so that
    ``0.5*(J + J.T)`` round-trips to the same per-edge couplings.
    """
    n = sampler.n_vars
    J = jnp.zeros((n, n), dtype=params["edge_weights"].dtype)
    J = J.at[sampler._ei, sampler._ej].set(params["edge_weights"])
    J = J.at[sampler._ej, sampler._ei].set(params["edge_weights"])
    model = eqx.tree_at(lambda m: m.J, model, J)
    model = eqx.tree_at(lambda m: m.h, model, params["biases"])
    return model


def make_kl_grad_step(
    sampler: THRMLIsingSampler,
    optimizer: optax.GradientTransformation,
    beta: float = 1.0,
    n_chains_neg: int = 128,
    schedule_negative: Optional[SamplingSchedule] = None,
):
    """Build the jit-compiled fully-visible ML step.

    Args:
        sampler: setup-once static structure (nodes/edges/colour blocks).
        optimizer: any optax optimizer over the {biases, edge_weights} pytree.
        beta: inverse temperature (fixed during training, as in CD).
        n_chains_neg: independent negative-phase chains per step.
        schedule_negative: THRML schedule for the negative phase; default
            SamplingSchedule(n_warmup=30, n_samples=15, steps_per_sample=2).

    Returns:
        step(params, opt_state, data_pm1, key) -> (params, opt_state, aux)
        where data_pm1 is (batch, n_vars) in {-1,+1} and aux carries the
        positive/negative moment gap (the training signal; -> 0 at optimum).
    """
    if schedule_negative is None:
        schedule_negative = SamplingSchedule(n_warmup=30, n_samples=15, steps_per_sample=2)

    nodes, edges, free_blocks = sampler.nodes, sampler.edges, sampler.free_blocks
    beta_arr = jnp.asarray(float(beta))

    @jax.jit
    def step(params, opt_state, data_pm1, key):
        ebm = IsingEBM(nodes, edges, params["biases"], params["edge_weights"], beta_arr)
        spec = IsingTrainingSpec(
            ebm,
            data_blocks=[Block(nodes)],
            conditioning_blocks=[],
            positive_sampling_blocks=[],       # fully visible: nothing to sample
            negative_sampling_blocks=free_blocks,
            schedule_positive=_DUMMY_POSITIVE_SCHEDULE,
            schedule_negative=schedule_negative,
        )

        k_init, k_grad = jax.random.split(key)
        init_neg = hinton_init(k_init, ebm, free_blocks, (n_chains_neg,))
        data_bool = data_pm1 > 0

        grad_w, grad_b, (mb_pos, mw_pos), (mb_neg, mw_neg) = estimate_kl_grad(
            k_grad, spec, nodes, edges,
            [data_bool], [],
            [],          # exact positive phase (fully visible, v0.1.3+)
            init_neg,
        )

        grads = {"biases": grad_b, "edge_weights": grad_w}
        with jax.numpy_dtype_promotion("standard"):
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        aux = {
            "moment_gap_b": jnp.mean(jnp.abs(
                jnp.mean(mb_pos, axis=(0, 1)) - jnp.mean(mb_neg, axis=0))),
            "moment_gap_w": jnp.mean(jnp.abs(
                jnp.mean(mw_pos, axis=(0, 1)) - jnp.mean(mw_neg, axis=0))),
        }
        return params, opt_state, aux

    return step


def fit_ising_ml(
    model,
    data_pm1: jnp.ndarray,
    key: jax.Array,
    n_iters: int = 150,
    batch_size: int = 128,
    lr: float = 0.05,
    n_chains_neg: int = 128,
    schedule_negative: Optional[SamplingSchedule] = None,
    sampler: Optional[THRMLIsingSampler] = None,
) -> Tuple[object, THRMLIsingSampler, list]:
    """Train a fully-visible QuadraticEBM by THRML-native ML.

    Args:
        model: a ``QuadraticEBM`` (its connectivity mask defines the graph).
        data_pm1: (N, n_vars) dataset of spins in {-1,+1}.
        key, n_iters, batch_size, lr, n_chains_neg, schedule_negative: controls.
        sampler: optional prebuilt ``THRMLIsingSampler`` (reused; else built).

    Returns:
        (trained_model, sampler, history) — history is the per-step mean
        |positive - negative| bias-moment gap (the convergence signal).
    """
    if sampler is None:
        sampler = THRMLIsingSampler(np.asarray(model.connectivity_mask))

    params = params_from_ebm(sampler, model)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    step = make_kl_grad_step(
        sampler, optimizer, beta=float(model.beta),
        n_chains_neg=n_chains_neg, schedule_negative=schedule_negative,
    )

    n = data_pm1.shape[0]
    history = []
    for _ in range(n_iters):
        key, kb, ks = jax.random.split(key, 3)
        batch = data_pm1[jax.random.randint(kb, (batch_size,), 0, n)]
        params, opt_state, aux = step(params, opt_state, batch, ks)
        history.append(float(aux["moment_gap_b"]))

    return params_to_ebm(sampler, params, model), sampler, history
