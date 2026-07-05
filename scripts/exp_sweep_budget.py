"""
Exactness-anchored sweep-budget experiment (the oracle plot).

The chain-CRF reverse step admits EXACT inference (``chain_marginals``), while
the same distribution runs on THRML as a 2-coloured block-Gibbs program — the
TSU-compatible path. That pairing lets us measure, exactly, how sample
fidelity depends on the Gibbs sweep budget the hardware spends: for each
budget k we draw THRML samples with n_warmup=k, form empirical per-position
marginals, and report the total-variation distance to the exact marginals.
The same estimate from exact FFBS samples gives the finite-sample noise floor.

Modes:
  --random-potentials   grid over (L, V, scale) with random potentials (CPU-fast)
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

from thermolm_jax.models.chain_crf import chain_marginals, chain_sample
from thermolm_jax.sampling.chain_mrf_thrml import sample_chain_thrml_single

DEFAULT_SWEEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def tv_to_exact(samples: np.ndarray, log_marg_exact: jnp.ndarray, V: int) -> float:
    """Mean-over-positions total-variation distance of empirical vs exact marginals."""
    emp = np.stack([(samples == v).mean(0) for v in range(V)], axis=1)  # (L, V)
    exact = np.asarray(jnp.exp(log_marg_exact))                          # (L, V)
    return float(0.5 * np.abs(emp - exact).sum(axis=1).mean())


def ffbs_floor(key, unary, pairwise, n_chains: int, V: int, log_marg) -> float:
    """Finite-sample TV of EXACT joint samples — the noise floor for n_chains."""
    keys = jax.random.split(key, n_chains)
    samples = np.asarray(jax.vmap(lambda k: chain_sample(k, unary, pairwise))(keys))
    return tv_to_exact(samples, log_marg, V)


def run_curve(key, unary, pairwise, sweeps, n_chains, label, temperature=1.0):
    """TV-vs-sweeps for one set of potentials. Returns list of result dicts."""
    L, V = unary.shape
    log_marg = chain_marginals(unary / temperature, pairwise / temperature)

    key, k_floor = jax.random.split(key)
    floor = ffbs_floor(k_floor, unary / temperature, pairwise / temperature,
                       n_chains, V, log_marg)
    print(f"  [{label}] L={L} V={V}  FFBS noise floor (n={n_chains}): TV={floor:.4f}")

    rows = []
    for k in sweeps:
        key, k_s = jax.random.split(key)
        samples = np.asarray(sample_chain_thrml_single(
            unary, pairwise, k_s, n_chains=n_chains, n_warmup=k,
            temperature=temperature,
        ))
        tv = tv_to_exact(samples, log_marg, V)
        rows.append({"label": label, "L": L, "V": V, "sweeps": k,
                     "tv": tv, "ffbs_floor": floor})
        print(f"  [{label}] sweeps={k:4d}  TV={tv:.4f}")
    return rows


def curves_random(key, sweeps, n_chains, scale):
    """Grid of random-potential chains of increasing size/vocab."""
    rows = []
    for L, V in [(16, 8), (32, 16), (64, 27)]:
        key, k1, k2, k_run = jax.random.split(key, 4)
        unary = jax.random.normal(k1, (L, V)) * scale
        pairwise = jax.random.normal(k2, (L - 1, V, V)) * scale
        rows += run_curve(k_run, unary, pairwise, sweeps, n_chains,
                          label=f"random L={L} V={V} s={scale}")
    return rows


def curves_ckpt(key, ckpt_path, t_levels, sweeps, n_chains):
    """Potentials from a trained diffusion LM's denoiser at noise levels t.

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
                          label=f"ckpt t={t}")
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = sorted({r["label"] for r in rows})
    for label in labels:
        sub = [r for r in rows if r["label"] == label]
        ax.plot([r["sweeps"] for r in sub], [r["tv"] for r in sub],
                marker="o", label=label)
        ax.axhline(sub[0]["ffbs_floor"], ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("block-Gibbs sweeps (THRML n_warmup)")
    ax.set_ylabel("TV distance to exact marginals")
    ax.set_title("Cost of thermodynamic sampling vs the exact-inference oracle\n"
                 "(dotted lines: exact-FFBS finite-sample noise floor)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = out_prefix + ".png"
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    if args.ckpt:
        rows = curves_ckpt(key, args.ckpt, args.t_levels, args.sweeps, args.n_chains)
        out = args.out or "results/sweep_budget_ckpt"
    else:
        rows = curves_random(key, args.sweeps, args.n_chains, args.scale)
        out = args.out or "results/sweep_budget"
    save_outputs(rows, out)


if __name__ == "__main__":
    main()
