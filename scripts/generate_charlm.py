"""
Generate text from a trained Tier-1 char-level chain-CRF diffusion LM.

Each reverse-diffusion step samples the chain CRF jointly — either exactly
(forward-filter backward-sample) or on THRML (the TSU-compatible path).

Example:
    python scripts/generate_charlm.py --ckpt runs/charlm.pkl --n 8 --thrml
"""

import argparse
import pickle

import numpy as np
import jax

from thermolm_jax.data.char_tokenizer import CharTokenizer
from thermolm_jax.models.diffusion_lm import build_net, generate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default="runs/charlm.pkl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--thrml", action="store_true",
                    help="sample each reverse step on THRML (TSU path) instead of exact FFBS")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.ckpt, "rb") as f:
        blob = pickle.load(f)
    cfg, params = blob["cfg"], blob["params"]
    tok = CharTokenizer(itos=blob["itos"], stoi={c: i for i, c in enumerate(blob["itos"])})
    net = build_net(cfg)

    samples = generate(
        params, net, cfg, jax.random.PRNGKey(args.seed),
        n_samples=args.n, n_steps=args.steps,
        temperature=args.temperature, use_thrml=args.thrml,
    )
    path = "THRML/TSU" if args.thrml else "exact FFBS"
    print(f"--- {args.n} samples (reverse-step sampler: {path}) ---")
    for s in np.asarray(samples):
        print(repr(tok.decode(s)))


if __name__ == "__main__":
    main()
