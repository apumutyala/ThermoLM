"""
Contrastive divergence training for quadratic Ising EBMs.

Maximum-likelihood training of an EBM p_θ(x) ∝ exp(-E_θ(x)) has the gradient

    ∇_θ L = E_data[∇_θ E_θ(x)] - E_model[∇_θ E_θ(x)]

(Hinton, 2002, "Training Products of Experts by Minimizing Contrastive
Divergence"). The first ("positive") expectation is over the data; the second
("negative") over model samples obtained by MCMC. In CD-k the negative chain is
initialised at the data and run for a few Gibbs sweeps.

This is implemented as the surrogate loss

    L(θ) = mean(E_θ(x_data)) - mean(E_θ(x_neg))           [+ optional L2 reg]

with ``x_neg`` treated as a constant (``stop_gradient``) so that autodiff
reproduces the two-term estimator above. Differentiating *through* the sampler
would be incorrect and expensive — and matters here because the sampler reads
the model parameters (local fields), so the stop-gradient is essential.

The optional L2 term ``λ (mean E_data² + mean E_neg²)`` keeps energy magnitudes
bounded; this is the regulariser used by Du & Mordatch (2019/2021), not the
"variance of negative energies" proxy used by an earlier version here.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

from ..sampling.chromatic_gibbs import chromatic_gibbs_sample


@dataclass(frozen=True)
class CDConfig:
    """Configuration for contrastive divergence.

    Frozen (hashable) so it can be passed as a static argument to the
    jit-compiled CD update.
    """
    k: int = 1                      # number of CD chains restarts (CD-k)
    n_gibbs_steps: int = 50         # Gibbs sweeps per chain
    temperature: float = 1.0        # sampling temperature
    l2_weight: float = 0.0          # weight of the L2 energy-magnitude regulariser
    use_thrml: bool = False         # use THRML's Ising sampler for the negative phase


def contrastive_divergence_loss(
    model,
    x_data: jnp.ndarray,
    key: jax.random.PRNGKey,
    config: CDConfig,
    energy_fn: Optional[Callable] = None,
    color_masks: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, dict]:
    """Contrastive-divergence surrogate loss and metrics.

    Args:
        model: Energy model; ``model(x)`` (or ``energy_fn(x)``) returns energy.
        x_data: (batch, n_vars) data spins in {-1, +1}.
        key: PRNG key.
        config: CD configuration.
        energy_fn: Optional energy callable (defaults to ``model``). When using
            the quadratic sampler this should be the model itself so the sampler
            can read ``J``/``h``.
        color_masks: Optional precomputed chromatic colour masks (keeps the
            negative-phase sampler ``jit``-able); see
            ``thermolm_jax.sampling.chromatic_gibbs.color_masks_from_colors``.

    Returns:
        (loss, info)
    """
    if energy_fn is None:
        energy_fn = model

    # Negative phase: short MCMC chain initialised at the data (CD-k).
    x_neg = x_data
    for _ in range(config.k):
        key, sub = jax.random.split(key)
        x_neg, _ = chromatic_gibbs_sample(
            energy_fn,
            x_neg,
            n_steps=config.n_gibbs_steps,
            key=sub,
            temperature=config.temperature,
            use_thrml=config.use_thrml,
            color_masks=color_masks,
        )
    # Treat negative samples as fixed: gives the correct two-term CD gradient.
    x_neg = jax.lax.stop_gradient(x_neg)

    E_data = energy_fn(x_data)
    E_neg = energy_fn(x_neg)

    cd_loss = jnp.mean(E_data) - jnp.mean(E_neg)

    l2_term = config.l2_weight * (jnp.mean(E_data ** 2) + jnp.mean(E_neg ** 2))
    loss = cd_loss + l2_term

    info = {
        "cd_loss": cd_loss,
        "l2_term": l2_term,
        "E_data": jnp.mean(E_data),
        "E_neg": jnp.mean(E_neg),
        "n_gibbs_steps": config.n_gibbs_steps,
        "k": config.k,
    }
    return loss, info


import equinox as eqx


@eqx.filter_jit
def _cd_update(model, opt_state, x_data, key, optimizer, config, color_masks):
    """jit-compiled CD value/grad + optimiser update.

    ``optimizer`` and ``config`` are static (non-array) and so are baked into the
    compiled function; ``color_masks`` is a fixed-shape array threaded in so the
    negative-phase sampler compiles (no in-trace graph colouring).
    """

    def loss_fn(m):
        return contrastive_divergence_loss(
            m, x_data, key, config, energy_fn=m, color_masks=color_masks
        )

    (loss, info), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, info


def contrastive_divergence_step(
    model,
    optimizer: optax.GradientTransformation,
    opt_state,
    x_data: jnp.ndarray,
    key: jax.random.PRNGKey,
    config: CDConfig,
    energy_fn: Optional[Callable] = None,
    color_masks: Optional[jnp.ndarray] = None,
) -> Tuple:
    """One CD optimisation step (Equinox model + Optax).

    For the quadratic Ising model, pass ``color_masks`` (from
    ``color_masks_from_colors``) to use the fast jit-compiled path. The
    ``energy_fn`` argument is accepted for API symmetry but the jit path always
    uses ``model`` as its own energy (so gradients flow to its parameters).

    Returns (model, opt_state, loss, info).
    """
    if color_masks is not None and (energy_fn is None or energy_fn is model):
        return _cd_update(model, opt_state, x_data, key, optimizer, config, color_masks)

    # Eager fallback (generic energy_fn, or no colour masks supplied).
    def loss_fn(m):
        return contrastive_divergence_loss(
            m, x_data, key, config,
            energy_fn=m if energy_fn is None else energy_fn,
            color_masks=color_masks,
        )

    (loss, info), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, info
