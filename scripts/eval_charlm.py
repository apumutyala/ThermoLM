"""
Evaluate a trained Tier-1 char-level chain-CRF diffusion LM.

Loads a checkpoint, computes held-out bits/char, and generates samples with
both the exact FFBS sampler and the THRML (TSU-compatible) path.

Example:
    python scripts/eval_charlm.py --ckpt runs/charlm_tinyshakespeare.pkl
"""

import argparse
import json
import os
import pickle

import numpy as np
import jax

from thermolm_jax.data.char_tokenizer import CharTokenizer
from thermolm_jax.models.diffusion_lm import DiffusionLMConfig, build_net, generate, denoising_loss
from thermolm_jax.models.diffusion_lm import q_xt


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    cfg = blob["cfg"]
    params = blob["params"]
    itos = blob["itos"]
    tok = CharTokenizer(itos=itos, stoi={c: i for i, c in enumerate(itos)})
    return cfg, params, tok


def evaluate_bits_per_char(params, net, cfg, tok, text_ids, key,
                           n_batches: int = 10, eval_batch: int = 64):
    """Evaluate mean bits/char on a held-out text stream.

    Cuts the stream into non-overlapping seq_len windows and scores up to
    ``n_batches`` batches of ``eval_batch`` windows each. (An earlier version
    conflated seq_len with the batch size when slicing.)
    """
    ids = np.asarray(text_ids)
    L = cfg.seq_len
    # Non-overlapping windows for clean held-out evaluation
    n_windows = len(ids) // L
    windows = []
    for i in range(n_windows):
        w = ids[i * L : i * L + L]
        if len(w) == L:
            windows.append(w)
    if not windows:
        return None
    windows = np.stack(windows)
    import jax.numpy as jnp
    windows_j = jnp.asarray(windows)

    n_full_batches = (len(windows) + eval_batch - 1) // eval_batch
    bpc_vals = []
    for b in range(min(n_batches, n_full_batches)):
        batch = windows_j[b * eval_batch : (b + 1) * eval_batch]
        if len(batch) == 0:
            break
        key, k = jax.random.split(key)
        loss, bpc = denoising_loss(params, net, batch, k, cfg)
        bpc_vals.append(float(bpc))
    if not bpc_vals:
        return None
    return float(np.mean(bpc_vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, required=True, help="path to .pkl checkpoint")
    ap.add_argument("--val_text", type=str, default=None, help="optional held-out text file")
    ap.add_argument("--eval_batch", type=int, default=64, help="windows per eval batch")
    ap.add_argument("--n_samples", type=int, default=8, help="number of generated samples")
    ap.add_argument("--n_steps", type=int, default=16, help="generation steps")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", type=str, default=None, help="write metrics to JSON")
    args = ap.parse_args()

    cfg, params, tok = load_checkpoint(args.ckpt)
    net = build_net(cfg)
    key = jax.random.PRNGKey(args.seed)

    print(f"[eval] loaded checkpoint: {args.ckpt}")
    print(f"[eval] vocab={tok.vocab_size}  seq_len={cfg.seq_len}  hidden={cfg.hidden_size}  layers={cfg.n_layers}")

    # Held-out bits/char
    val_bpc = None
    if args.val_text and os.path.exists(args.val_text):
        with open(args.val_text, "r", encoding="utf-8") as f:
            val_text = f.read()
        val_ids = tok.encode(val_text)
        val_bpc = evaluate_bits_per_char(params, net, cfg, tok, val_ids, key,
                                         eval_batch=args.eval_batch)
        if val_bpc is not None:
            print(f"[eval] held-out bits/char ~ {val_bpc:.3f}")
    else:
        print("[eval] no --val_text provided; skipping held-out bits/char")

    # Generation: exact FFBS
    key, k_gen = jax.random.split(key)
    samples_exact = np.asarray(
        generate(params, net, cfg, k_gen, n_samples=args.n_samples, n_steps=args.n_steps,
                 temperature=args.temperature, use_thrml=False)
    )
    print("\n--- Exact FFBS samples ---")
    for s in samples_exact:
        print(repr(tok.decode(s)))

    # Generation: THRML path
    key, k_gen = jax.random.split(key)
    try:
        samples_thrml = np.asarray(
            generate(params, net, cfg, k_gen, n_samples=args.n_samples, n_steps=args.n_steps,
                     temperature=args.temperature, use_thrml=True)
        )
        print("\n--- THRML/TSU samples ---")
        for s in samples_thrml:
            print(repr(tok.decode(s)))
    except Exception as e:
        print(f"\n[warn] THRML generation failed (expected if THRML not installed): {e}")
        samples_thrml = None

    metrics = {
        "checkpoint": args.ckpt,
        "vocab_size": tok.vocab_size,
        "seq_len": cfg.seq_len,
        "hidden_size": cfg.hidden_size,
        "n_layers": cfg.n_layers,
        "val_bpc": val_bpc,
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "temperature": args.temperature,
    }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[save] metrics -> {args.out_json}")


if __name__ == "__main__":
    main()
