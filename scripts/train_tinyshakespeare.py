"""
Train the Tier-1 char-level chain-CRF diffusion LM on TinyShakespeare.

Auto-downloads the dataset if not present, splits into train/val, trains with
GPU-optimised JAX settings, and saves checkpoints.

Examples:
    # CPU sanity (tiny, fast)
    python scripts/train_tinyshakespeare.py --sanity

    # Full GPU run on TinyShakespeare (A100/H100, ~10 min)
    python scripts/train_tinyshakespeare.py --iters 3000 --seq_len 128 \
        --hidden 256 --batch 64 --lr 3e-3 --out runs/charlm_tinyshakespeare.pkl

    # With GPU XLA flags (recommended on A100+):
    XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true" \
        python scripts/train_tinyshakespeare.py --iters 3000 --seq_len 128 --hidden 256
"""

import argparse
import os
import pickle
import time
import urllib.request

import numpy as np
import jax
import jax.numpy as jnp

from thermolm_jax.data.char_tokenizer import CharTokenizer, make_windows
from thermolm_jax.models.diffusion_lm import DiffusionLMConfig, fit, unigram_bits_per_char

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tinyshakespeare(path: str = "data/tinyshakespeare.txt") -> str:
    """Download TinyShakespeare if it doesn't exist."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[data] downloading TinyShakespeare -> {path}")
    urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, path)
    return path


def split_train_val(ids: np.ndarray, val_frac: float = 0.1) -> tuple:
    """Split token ids into train/val (contiguous, last val_frac is validation)."""
    n = len(ids)
    n_val = max(1, int(n * val_frac))
    return ids[:-n_val], ids[-n_val:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/tinyshakespeare.txt")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--out", type=str, default="runs/charlm_tinyshakespeare.pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true",
                    help="tiny CPU config on a small embedded corpus")
    ap.add_argument("--gpu", action="store_true",
                    help="enable GPU-specific optimisations (bfloat16, XLA hints)")
    args = ap.parse_args()

    # GPU optimisation: set default matmul precision for A100+ speed
    if args.gpu:
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        print("[gpu] bfloat16 matmul precision enabled")

    if args.sanity:
        text = (
            "to be or not to be that is the question "
            "whether tis nobler in the mind to suffer "
        ) * 40
        args.seq_len, args.hidden, args.layers, args.iters, args.batch = 24, 128, 2, 300, 64
    else:
        data_path = download_tinyshakespeare(args.data)
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    train_ids, val_ids = split_train_val(ids, args.val_frac)
    print(f"[data] vocab={tok.vocab_size}  total_chars={len(ids)}  "
          f"train={len(train_ids)}  val={len(val_ids)}")

    stride = args.stride or max(args.seq_len // 4, 1)
    train_windows = make_windows(train_ids, args.seq_len, stride)
    val_windows = make_windows(val_ids, args.seq_len, stride)
    print(f"[data] train_windows={train_windows.shape}  val_windows={val_windows.shape}")

    cfg = DiffusionLMConfig(
        vocab_size=tok.vocab_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        n_layers=args.layers,
    )

    key = jax.random.PRNGKey(args.seed)
    print(f"[train] config={cfg}")
    print(f"[train] starting {args.iters} iterations (batch={args.batch}, lr={args.lr})")

    t0 = time.time()
    net, params, history = fit(
        train_windows, cfg, key,
        n_iters=args.iters, batch_size=args.batch, lr=args.lr,
    )
    elapsed = time.time() - t0

    # Final validation bpc (average over last 20 iters for smoothing)
    final_bpc = sum(history[-20:]) / min(len(history), 20)
    baseline = unigram_bits_per_char(train_ids, tok.vocab_size)
    print(f"[train] finished in {elapsed:.1f}s")
    print(f"[train] final train bits/char ~ {final_bpc:.3f}  (unigram baseline {baseline:.3f})")

    # Optional: quick validation bits/char (cheap, one forward pass)
    if len(val_windows) > 0:
        from thermolm_jax.models.diffusion_lm import denoising_loss
        key, k_val = jax.random.split(key)
        val_loss, val_bpc = denoising_loss(params, net, val_windows[:args.batch], k_val, cfg)
        print(f"[val]   val bits/char   ~ {float(val_bpc):.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"params": params, "cfg": cfg, "itos": tok.itos, "history": history}, f)
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
