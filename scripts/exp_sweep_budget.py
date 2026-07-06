"""
Fair head-to-head: exact FFBS vs. GPU chromatic Gibbs vs. THRML block Gibbs.

The chain-CRF reverse step admits EXACT inference (``chain_marginals``), while
the same distribution can be sampled by two block-Gibbs algorithms:
- a JAX chromatic Gibbs sampler on GPU (the fair GPU baseline), and
- THRML's ``IsingSamplingProgram`` (the hardware-shaped / TSU path).

This script measures the quality/efficiency tradeoff across all three samplers
on the *same* model, varying the compute budget. Quality is measured by total-
variation distance to the exact per-position marginals and by mean log-
likelihood of the samples under the exact CRF. Compute is measured by wall-
clock time and by number of full-graph updates (sweeps for Gibbs, independent
samples for FFBS).

This is the experiment Zach's critique calls for: a fair GPU-vs-thermodynamic
comparison on sparse-graph language models, with an exact oracle that makes the
approximation gap measurable.

Modes:
  --random-potentials   grid over (L, V, scale) with random potentials
  --ckpt runs/x.pkl     potentials from a trained diffusion LM's denoiser at
                        chosen noise levels t (the LM's actual reverse step)

Outputs results/sweep_budget*.csv and .png.

Examples:
    python scripts/exp_sweep_budget.py --random-potentials
    python scripts/exp_sweep_budget.py --ckpt runs/charlm_tinyshakespeare.pkl \
        --t-levels 1.0 0.5 --n-chains 4000
"""

import argparse
import csv
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import time

from thermolm_jax.models.chain_crf import chain_marginals, chain_sample, chain_log_likelihood
from thermolm_jax.sampling.chain_mrf_thrml import sample_chain_thrml_single
from thermolm_jax.sampling.chain_gibbs_jax import sample_chain_jax_gibbs

DEFAULT_SWEEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def tv_to_exact(samples: np.ndarray, log_marg_exact: jnp.ndarray, V: int) -> float:
    """Mean-over-positions total-variation distance of empirical vs exact marginals."""
    emp = np.stack([(samples == v).mean(0) for v in range(V)], axis=1)  # (L, V)
    exact = np.asarray(jnp.exp(log_marg_exact))                          # (L, V)
    return float(0.5 * np.abs(emp - exact).sum(axis=1).mean())


def mean_nll(samples: np.ndarray, unary, pairwise) -> float:
    """Mean negative log-likelihood of samples under the exact CRF."""
    samples = jnp.asarray(samples)
    nlls = -jax.vmap(lambda x: chain_log_likelihood(x, unary, pairwise))(samples)
    return float(np.asarray(nlls.mean()))


def _block_until_ready(x):
    """Force JAX arrays to be materialised before timing."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    return x


def _time_sampler(fn, *args, n_reps=3):
    """Wall-clock time (seconds) of fn(*args), averaged over n_reps."""
    # Warmup
    out = fn(*args)
    _block_until_ready(out)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        out = fn(*args)
        _block_until_ready(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def run_curve(
    key, unary, pairwise, sweeps, n_chains, label, temperature=1.0, samplers=None
):
    """Compare samplers for one set of potentials. Returns list of result dicts.

    samplers: subset of {"ffbs", "jax_gibbs", "thrml_gibbs"}. Default all.
    """
    samplers = samplers or {"ffbs", "jax_gibbs", "thrml_gibbs"}
    L, V = unary.shape
    u = unary / temperature
    w = pairwise / temperature
    log_marg = chain_marginals(u, w)

    # Pre-compile / time one unit of work for each sampler.
    timings = {}
    if "ffbs" in samplers:
        key, k_t = jax.random.split(key)
        def ffbs_fn():
            return jax.vmap(lambda k: chain_sample(k, u, w))(
                jax.random.split(k_t, n_chains)
            )
        timings["ffbs"] = _time_sampler(ffbs_fn)
        print(f"  [{label}] L={L} V={V}  FFBS time (n={n_chains}): {timings['ffbs']:.4f}s")

    if "jax_gibbs" in samplers:
        key, k_t = jax.random.split(key)
        def jax_fn():
            return sample_chain_jax_gibbs(u, w, k_t, n_chains=n_chains, n_warmup=1)
        timings["jax_gibbs"] = _time_sampler(jax_fn)
        print(f"  [{label}] L={L} V={V}  JAX-Gibbs 1 sweep (n={n_chains}): {timings['jax_gibbs']:.4f}s")

    if "thrml_gibbs" in samplers:
        key, k_t = jax.random.split(key)
        def thrml_fn():
            return sample_chain_thrml_single(u, w, k_t, n_chains=n_chains, n_warmup=1)
        timings["thrml_gibbs"] = _time_sampler(thrml_fn)
        print(f"  [{label}] L={L} V={V}  THRML-Gibbs 1 sweep (n={n_chains}): {timings['thrml_gibbs']:.4f}s")

    # Baseline: exact FFBS noise floor at a fixed, generous n_chains.
    key, k_floor = jax.random.split(key)
    ffbs_samples = np.asarray(jax.vmap(lambda k: chain_sample(k, u, w))(
        jax.random.split(k_floor, n_chains)
    ))
    floor = tv_to_exact(ffbs_samples, log_marg, V)
    floor_nll = mean_nll(ffbs_samples, u, w)
    print(f"  [{label}] L={L} V={V}  FFBS noise floor (n={n_chains}): TV={floor:.4f}  NLL={floor_nll:.3f}")

    rows = []
    for k in sweeps:
        key, k_s = jax.random.split(key)

        if "thrml_gibbs" in samplers:
            samples = np.asarray(sample_chain_thrml_single(
                u, w, k_s, n_chains=n_chains, n_warmup=k,
            ))
            tv = tv_to_exact(samples, log_marg, V)
            nll = mean_nll(samples, u, w)
            rows.append({
                "label": label, "sampler": "thrml_gibbs", "L": L, "V": V,
                "sweeps": k, "wall_seconds": timings["thrml_gibbs"] * k,
                "tv": tv, "nll": nll, "ffbs_floor_tv": floor, "ffbs_floor_nll": floor_nll,
                "n_chains": n_chains,
            })
            print(f"  [{label}] THRML sweeps={k:4d}  TV={tv:.4f}  NLL={nll:.3f}  time={timings['thrml_gibbs']*k:.3f}s")

        if "jax_gibbs" in samplers:
            key, k_s2 = jax.random.split(key)
            samples = np.asarray(sample_chain_jax_gibbs(
                u, w, k_s2, n_chains=n_chains, n_warmup=k,
            ))
            tv = tv_to_exact(samples, log_marg, V)
            nll = mean_nll(samples, u, w)
            rows.append({
                "label": label, "sampler": "jax_gibbs", "L": L, "V": V,
                "sweeps": k, "wall_seconds": timings["jax_gibbs"] * k,
                "tv": tv, "nll": nll, "ffbs_floor_tv": floor, "ffbs_floor_nll": floor_nll,
                "n_chains": n_chains,
            })
            print(f"  [{label}] JAX   sweeps={k:4d}  TV={tv:.4f}  NLL={nll:.3f}  time={timings['jax_gibbs']*k:.3f}s")

        if "ffbs" in samplers:
            # At the same wall time as k Gibbs sweeps, how many independent exact samples?
            for sampler_name in ["jax_gibbs", "thrml_gibbs"]:
                if sampler_name not in timings:
                    continue
                t_budget = timings[sampler_name] * k
                n_exact = max(1, int(round(t_budget / timings["ffbs"])))
                key, k_e = jax.random.split(key)
                exact_samples = np.asarray(jax.vmap(lambda k: chain_sample(k, u, w))(
                    jax.random.split(k_e, n_exact)
                ))
                tv = tv_to_exact(exact_samples, log_marg, V)
                nll = mean_nll(exact_samples, u, w)
                rows.append({
                    "label": label, "sampler": f"ffbs_at_{sampler_name}_time", "L": L, "V": V,
                    "sweeps": k, "wall_seconds": t_budget,
                    "tv": tv, "nll": nll, "ffbs_floor_tv": floor, "ffbs_floor_nll": floor_nll,
                    "n_chains": n_exact,
                })
                print(f"  [{label}] FFBS({n_exact:4d} samples, {sampler_name} time)  TV={tv:.4f}  NLL={nll:.3f}")

    return rows


def curves_random(key, sweeps, n_chains, scale, samplers=None):
    """Grid of random-potential chains of increasing size/vocab."""
    rows = []
    grid = [(16, 8), (32, 16), (64, 27), (128, 65)]
    for L, V in grid:
        key, k1, k2, k_run = jax.random.split(key, 4)
        unary = jax.random.normal(k1, (L, V)) * scale
        pairwise = jax.random.normal(k2, (L - 1, V, V)) * scale
        rows += run_curve(k_run, unary, pairwise, sweeps, n_chains,
                          label=f"random L={L} V={V} s={scale}", samplers=samplers)
    return rows


def curves_ckpt(key, ckpt_path, t_levels, sweeps, n_chains, samplers=None):
    """Potentials from a trained diffusion LM denoiser at noise levels t.

    At t=1.0 the context is all-MASK (the first reverse step). For t<1 we mask
    a real window at rate t, so the potentials are conditioned on genuine text.
    """
    from thermolm_jax.models.diffusion_lm import build_net, q_xt
    from thermolm_jax.data.char_tokenizer import CharTokenizer

    with open(ckpt_path, "rb") as f:
        blob = pickle.load(f)
    cfg, params = blob["cfg"], blob["params"]
    tok = CharTokenizer(itos=blob["itos"], stoi={c: i for i, c in enumerate(blob["itos"])})
    net = build_net(cfg)
    L, V, mask_id = cfg.seq_len, cfg.vocab_size, cfg.vocab_size

    # A real context window for partially-masked levels: prefer training data
    # if present, else a neutral repeated fallback.
    text_path = "data/tinyshakespeare.txt"
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            ids = tok.encode(f.read()[: 4 * L + 1000])
    else:
        ids = tok.encode(("the quick brown fox jumps over the lazy dog " * 40))
    x0 = jnp.asarray(np.asarray(ids[:L]))[None, :]  # (1, L)

    rows = []
    for t in t_levels:
        key, k_q, k_run = jax.random.split(key, 3)
        if t >= 1.0:
            x_t = jnp.full((1, L), mask_id, dtype=jnp.int32)
        else:
            x_t = q_xt(x0, jnp.asarray([[t]]), mask_index=mask_id, key=k_q)
        t_arr = jnp.asarray([t], dtype=jnp.float32)
        unary, pairwise = net.apply(params, x_t, t_arr)
        rows += run_curve(k_run, unary[0], pairwise[0], sweeps, n_chains,
                          label=f"ckpt t={t}", samplers=samplers)
    return rows


def save_outputs(rows, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = out_prefix + ".csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping plot (pip install matplotlib)")
        return

    labels = sorted({r["label"] for r in rows})

    # Plot 1: TV vs sweeps
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in labels:
        for sampler in ["jax_gibbs", "thrml_gibbs"]:
            sub = [r for r in rows if r["label"] == label and r["sampler"] == sampler]
            if not sub:
                continue
            marker = "o" if sampler == "jax_gibbs" else "s"
            ax.plot([r["sweeps"] for r in sub], [r["tv"] for r in sub],
                    marker=marker, label=f"{label} ({sampler})")
        floor = next((r["ffbs_floor_tv"] for r in rows if r["label"] == label), None)
        if floor is not None:
            ax.axhline(floor, ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("block-Gibbs sweeps")
    ax.set_ylabel("TV distance to exact marginals")
    ax.set_title("Sampler quality vs. sweep budget (THRML vs. JAX-GPU)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    png_path = out_prefix + "_sweeps.png"
    fig.savefig(png_path, dpi=150)
    print(f"[save] {png_path}")

    # Plot 2: TV vs wall-clock time (the fair comparison)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in labels:
        for sampler in ["jax_gibbs", "thrml_gibbs", "ffbs_at_thrml_gibbs_time"]:
            sub = [r for r in rows if r["label"] == label and r["sampler"] == sampler]
            if not sub:
                continue
            style = {"jax_gibbs": ("o", "-"), "thrml_gibbs": ("s", "-"),
                     "ffbs_at_thrml_gibbs_time": ("^", "--")}[sampler]
            ax.plot([r["wall_seconds"] for r in sub], [r["tv"] for r in sub],
                    marker=style[0], ls=style[1], label=f"{label} ({sampler})")
        floor = next((r["ffbs_floor_tv"] for r in rows if r["label"] == label), None)
        if floor is not None:
            ax.axhline(floor, ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock time (seconds)")
    ax.set_ylabel("TV distance to exact marginals")
    ax.set_title("Fair head-to-head: same wall-clock budget, same model")
    ax.legend(fontsize=7)
    fig.tight_layout()
    png_path = out_prefix + "_time.png"
    fig.savefig(png_path, dpi=150)
    print(f"[save] {png_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--random-potentials", action="store_true",
                    help="run the random-potential grid (default if no --ckpt)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="trained diffusion-LM checkpoint (.pkl)")
    ap.add_argument("--t-levels", type=float, nargs="+", default=[1.0, 0.5],
                    help="noise levels for --ckpt potentials")
    ap.add_argument("--sweeps", type=int, nargs="+", default=DEFAULT_SWEEPS)
    ap.add_argument("--n-chains", type=int, default=2000)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="potential scale for --random-potentials")
    ap.add_argument("--out", type=str, default=None,
                    help="output prefix (default results/sweep_budget[_ckpt])")
    ap.add_argument("--samplers", type=str, nargs="+",
                    default=["ffbs", "jax_gibbs", "thrml_gibbs"],
                    help="which samplers to benchmark")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    samplers = set(args.samplers)
    if args.ckpt:
        rows = curves_ckpt(key, args.ckpt, args.t_levels, args.sweeps, args.n_chains, samplers)
        out = args.out or "results/sweep_budget_ckpt"
    else:
        rows = curves_random(key, args.sweeps, args.n_chains, args.scale, samplers)
        out = args.out or "results/sweep_budget"
    save_outputs(rows, out)


if __name__ == "__main__":
    main()
