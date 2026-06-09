"""
Train the Tier-1 char-level chain-CRF discrete-diffusion language model.

Trains by exact CRF denoising conditional ML and reports bits/char vs the
unigram baseline. Saves params + tokenizer for generation.

Examples:
    # full run on TinyShakespeare (download to data/tinyshakespeare.txt first):
    #   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python scripts/train_charlm.py --data data/tinyshakespeare.txt --iters 3000 \
        --seq_len 128 --hidden 256 --out runs/charlm.pkl

    # offline sanity run (tiny, CPU, embedded text):
    python scripts/train_charlm.py --sanity
"""

import argparse
import os
import pickle

import jax

from thermolm_jax.data.char_tokenizer import CharTokenizer, make_windows
from thermolm_jax.models.diffusion_lm import DiffusionLMConfig, fit, unigram_bits_per_char

# Small public-domain-style fallback corpus so the pipeline runs with no download.
_FALLBACK = (
    "to be or not to be that is the question whether tis nobler in the mind "
    "to suffer the slings and arrows of outrageous fortune or to take arms "
    "against a sea of troubles and by opposing end them to die to sleep no more "
) * 40


def load_text(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    print("[data] no --data file found; using the embedded fallback corpus.")
    return _FALLBACK


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--out", type=str, default="runs/charlm.pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true",
                    help="tiny CPU config on the embedded corpus")
    args = ap.parse_args()

    if args.sanity:
        args.seq_len, args.hidden, args.layers, args.iters, args.batch = 24, 128, 2, 300, 64

    text = load_text(args.data)
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    stride = args.stride or max(args.seq_len // 4, 1)
    windows = make_windows(ids, args.seq_len, stride)
    print(f"[data] vocab={tok.vocab_size}  chars={len(ids)}  windows={windows.shape}")

    cfg = DiffusionLMConfig(
        vocab_size=tok.vocab_size, seq_len=args.seq_len,
        hidden_size=args.hidden, n_layers=args.layers,
    )
    net, params, history = fit(
        windows, cfg, jax.random.PRNGKey(args.seed),
        n_iters=args.iters, batch_size=args.batch, lr=args.lr,
    )

    baseline = unigram_bits_per_char(ids, tok.vocab_size)
    final = sum(history[-20:]) / min(len(history), 20)
    print(f"[train] final bits/char ~ {final:.3f}   (unigram baseline {baseline:.3f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"params": params, "cfg": cfg, "itos": tok.itos}, f)
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
