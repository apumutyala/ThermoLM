"""
End-to-end demo for the validated DTM / quadratic-Ising track.

What it shows, on CPU in well under a minute:
  1. A quadratic Ising EBM, sampled with chromatic block Gibbs, reproduces the
     EXACT Boltzmann marginals of a small random model (sanity check that the
     sampler is correct, not just plausible).
  2. The same EBM is trained by contrastive divergence to model a toy bimodal
     binary distribution, and generated samples then concentrate on the data
     modes.

Run:
    python scripts/dtm_ising_demo.py

This is the reproducible anchor for the repository; see the README and
STATUS.md for what is and is not validated.
"""

import itertools
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from thermolm_jax.models.quadratic_ebm import QuadraticEBM, QuadraticEBMConfig
from thermolm_jax.sampling.chromatic_gibbs import (
    chromatic_gibbs_sample,
    greedy_coloring,
    color_masks_from_colors,
)
from thermolm_jax.training.contrastive_divergence import contrastive_divergence_step, CDConfig


def exact_marginals(ebm, n):
    """First and (nearest-neighbour) second moments by exact enumeration."""
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    E = np.asarray(ebm(jnp.asarray(states)))
    w = np.exp(-(E - E.min()))
    w /= w.sum()
    mag = (w[:, None] * states).sum(0)
    return states, w, mag


def part1_sampler_correctness(key):
    print("\n[1] Chromatic Gibbs vs EXACT Boltzmann marginals (random Ising, n=6)")
    n = 6
    ebm = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=0.6), key)
    # sparse chain so the colouring is non-trivial (2 colours)
    mask = np.zeros((n, n), bool)
    for i in range(n - 1):
        mask[i, i + 1] = mask[i + 1, i] = True
    ebm = ebm.set_connectivity(jnp.asarray(mask))

    _, _, mag_exact = exact_marginals(ebm, n)

    init = jax.random.randint(key, (5000, n), 0, 2) * 2 - 1
    samp = np.asarray(chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 300, key)[0])
    mag_jax = samp.mean(0)

    samp_t = np.asarray(
        chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 300, key, use_thrml=True)[0]
    )
    mag_thrml = samp_t.mean(0)

    print("    exact <s_i>:", np.round(mag_exact, 3))
    print("    JAX   <s_i>:", np.round(mag_jax, 3), " max|err| =", round(float(np.abs(mag_jax - mag_exact).max()), 3))
    print("    THRML <s_i>:", np.round(mag_thrml, 3), " max|err| =", round(float(np.abs(mag_thrml - mag_exact).max()), 3))


def part2_cd_training(key):
    print("\n[2] Contrastive-divergence training on a toy bimodal distribution (n=8)")
    n = 8
    proto = np.array(
        [[1, 1, 1, 1, -1, -1, -1, -1], [-1, -1, 1, 1, -1, -1, 1, 1]], dtype=float
    )

    def make_batch(k, B=128, flip=0.05):
        k_idx, k_noise = jax.random.split(k)
        idx = np.asarray(jax.random.randint(k_idx, (B,), 0, 2))
        x = proto[idx]
        noise = np.asarray(jax.random.uniform(k_noise, (B, n))) < flip
        return jnp.asarray(np.where(noise, -x, x))

    ebm = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=0.01), key)
    ebm = ebm.set_connectivity(jnp.ones((n, n), bool))  # fully connected
    # Precompute the chromatic colour masks once so the CD step compiles (jit).
    cmasks = color_masks_from_colors(greedy_coloring(ebm.connectivity_mask), n)
    opt = optax.adam(0.05)
    opt_state = opt.init(eqx.filter(ebm, eqx.is_inexact_array))
    cfg = CDConfig(k=1, n_gibbs_steps=20, temperature=1.0)

    for step in range(120):
        key, kb, kc = jax.random.split(key, 3)
        ebm, opt_state, loss, info = contrastive_divergence_step(
            ebm, opt, opt_state, make_batch(kb), kc, cfg, color_masks=cmasks
        )
        if step % 30 == 0 or step == 119:
            print(
                f"    step {step:3d}  E_data {float(info['E_data']):7.3f}"
                f"  E_neg {float(info['E_neg']):7.3f}  gap {float(info['E_data'] - info['E_neg']):6.3f}"
            )

    init = jax.random.randint(key, (1000, n), 0, 2) * 2 - 1
    samp = np.asarray(
        chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 400, key, color_masks=cmasks)[0]
    )
    dist = np.minimum((samp != proto[0]).sum(1), (samp != proto[1]).sum(1))
    print(f"    fraction of samples exactly on a data mode: {float((dist == 0).mean()):.2f}")
    print(f"    mean Hamming distance to nearest mode:       {float(dist.mean()):.2f}  (of {n} bits)")


def part3_thrml_native_ml(key):
    print("\n[3] THRML-native ML training: exact positive phase + THRML negative phase (n=8)")
    from thermolm_jax.models.thrml_quadratic import THRMLIsingSampler
    from thermolm_jax.training.thrml_ml import fit_ising_ml

    n = 8
    proto = np.array(
        [[1, 1, 1, 1, -1, -1, -1, -1], [-1, -1, 1, 1, -1, -1, 1, 1]], dtype=float
    )
    key, k_idx, k_flip, kf = jax.random.split(key, 4)
    idx = np.asarray(jax.random.randint(k_idx, (2000,), 0, 2))
    flips = np.asarray(jax.random.uniform(k_flip, (2000, n))) < 0.05
    data = jnp.asarray(np.where(flips, -proto[idx], proto[idx]))

    ebm = QuadraticEBM(QuadraticEBMConfig(n_vars=n, beta=1.0, init_scale=0.01), key)
    ebm = ebm.set_connectivity(jnp.ones((n, n), bool))

    ebm, sampler, history = fit_ising_ml(
        ebm, data, kf, n_iters=100, batch_size=256, lr=0.05, n_chains_neg=256
    )
    print(f"    positive/negative moment gap: start {history[0]:.3f} -> end {history[-1]:.3f}")
    print("    (positive moments are EXACT from data - v0.1.3 fully-visible path;")
    print("     negative phase sampled by THRML's IsingSamplingProgram)")

    key, ks = jax.random.split(key)
    init = jax.random.randint(ks, (1000, n), 0, 2) * 2 - 1
    samp, _ = sampler.sample(ebm.J, ebm.h, ebm.beta, init.astype(jnp.float32), ks, 400)
    samp = np.asarray(samp)
    dist = np.minimum((samp != proto[0]).sum(1), (samp != proto[1]).sum(1))
    print(f"    fraction of samples exactly on a data mode: {float((dist == 0).mean()):.2f}")
    print(f"    mean Hamming distance to nearest mode:       {float(dist.mean()):.2f}  (of {n} bits)")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    key = jax.random.PRNGKey(0)
    part1_sampler_correctness(key)
    part2_cd_training(jax.random.PRNGKey(1))
    part3_thrml_native_ml(jax.random.PRNGKey(2))
    print("\nDone. Sampler correctness, CD training, and THRML-native ML training all ran on CPU.")


if __name__ == "__main__":
    main()
