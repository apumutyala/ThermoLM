"""
Multi-GPU data-parallel training for the Tier-1 chain-CRF diffusion LM.

Uses JAX pmap to replicate parameters across devices, shard the batch, and
all-reduce gradients. Designed for 2×A100s (or more) with bfloat16 Tensor Core
precision. Falls back to single-device training on CPU or 1 GPU.

Examples:
    # 2×A100 full run (~2–3 hours for 20k iters)
    python scripts/train_distributed.py --dataset wikitext2 --iters 20000 --seq_len 256 \
        --hidden 512 --layers 6 --batch 256 --lr 3e-3 \
        --out runs/charlm_2xa100.pkl
    # TinyShakespeare distributed/single-GPU run
    python scripts/train_distributed.py --dataset tinyshakespeare --iters 5000

    # Tiny CPU test (single device, pmap with 1 device)
    python scripts/train_distributed.py --sanity
"""

import argparse
import os
import pickle
import time
import urllib.request
import zipfile

import numpy as np
import jax
import jax.numpy as jnp
import optax

try:
    from tqdm import tqdm
    progress_write = tqdm.write
except ImportError:
    # fallback: simple progress counter
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', None) or len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:
                print(f"[progress] {i}/{total}")
            yield item
        print(f"[progress] done ({i+1} items)")
    def progress_write(s):
        print(s)

from thermolm_jax.data.char_tokenizer import CharTokenizer, make_windows
from thermolm_jax.models.diffusion_lm import (
    DiffusionLMConfig, build_net, init_params, denoising_loss,
    unigram_bits_per_char,
)

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# Mirrors tried in order; the original research.metamind.io S3 bucket is dead.
_WIKITEXT2_RAW_URLS = [
    "https://wikitext.smerity.com/wikitext-2-raw-v1.zip",
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
]


def download_tinyshakespeare(path: str = "data/tinyshakespeare.txt") -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[data] downloading TinyShakespeare -> {path}")
    urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, path)
    return path

def ensure_wikitext2_raw(data_dir: str = "data/wikitext-2-raw") -> tuple[str, str, str]:
    """Download/extract WikiText-2 raw and return train/valid/test file paths."""
    direct_train = os.path.join(data_dir, "wiki.train.raw")
    direct_valid = os.path.join(data_dir, "wiki.valid.raw")
    direct_test = os.path.join(data_dir, "wiki.test.raw")
    if all(os.path.exists(p) for p in (direct_train, direct_valid, direct_test)):
        return direct_train, direct_valid, direct_test

    nested_dir = os.path.join(data_dir, "wikitext-2-raw")
    nested_train = os.path.join(nested_dir, "wiki.train.raw")
    nested_valid = os.path.join(nested_dir, "wiki.valid.raw")
    nested_test = os.path.join(nested_dir, "wiki.test.raw")
    if all(os.path.exists(p) for p in (nested_train, nested_valid, nested_test)):
        return nested_train, nested_valid, nested_test

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "wikitext-2-raw-v1.zip")
    if not os.path.exists(zip_path):
        for url in _WIKITEXT2_RAW_URLS:
            try:
                print(f"[data] downloading WikiText-2 raw from {url}")
                urllib.request.urlretrieve(url, zip_path)
                break
            except Exception as e:  # noqa: BLE001 - try the next mirror
                print(f"[data] download failed ({e}); trying next source")

    if os.path.exists(zip_path):
        print(f"[data] extracting WikiText-2 raw -> {data_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

        if all(os.path.exists(p) for p in (nested_train, nested_valid, nested_test)):
            return nested_train, nested_valid, nested_test
        if all(os.path.exists(p) for p in (direct_train, direct_valid, direct_test)):
            return direct_train, direct_valid, direct_test

    # Last resort: rebuild the raw files via the HuggingFace `datasets` hub.
    print("[data] zip mirrors unavailable; falling back to the `datasets` package")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise FileNotFoundError(
            f"Could not download WikiText-2 raw under {data_dir}. Install the "
            "fallback loader with `pip install datasets` and retry."
        ) from e
    for split, fname in [("train", direct_train), ("validation", direct_valid), ("test", direct_test)]:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        with open(fname, "w", encoding="utf-8") as f:
            f.write("".join(ds["text"]))
        print(f"[data] wrote {fname}")
    return direct_train, direct_valid, direct_test


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_train_val(ids: np.ndarray, val_frac: float = 0.1) -> tuple:
    n = len(ids)
    n_val = max(1, int(n * val_frac))
    return ids[:-n_val], ids[-n_val:]


def cosine_lr_schedule(lr: float, warmup_steps: int, total_steps: int):
    """Cosine decay with linear warmup."""
    def schedule(step):
        step = jnp.minimum(step, total_steps)
        warmup_lr = lr * step / jnp.maximum(warmup_steps, 1)
        decay_steps = jnp.maximum(total_steps - warmup_steps, 1)
        progress = (step - warmup_steps) / decay_steps
        progress = jnp.clip(progress, 0.0, 1.0)
        decay_lr = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        return jnp.where(step < warmup_steps, warmup_lr, decay_lr)
    return schedule


def create_train_step(net, cfg, opt):
    """Create a pmapped train step."""
    def loss_fn(params, batch, key):
        return denoising_loss(params, net, batch, key, cfg)

    def train_step(params, opt_state, batch, key):
        (loss, bpc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, key)
        # all-reduce gradients and metrics across devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        loss = jax.lax.pmean(loss, axis_name='batch')
        bpc = jax.lax.pmean(bpc, axis_name='batch')
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, bpc

    return jax.pmap(train_step, axis_name='batch', donate_argnums=(0, 1))


def create_eval_step(net, cfg):
    """Create a pmapped eval step (no gradients, no updates)."""
    def eval_fn(params, batch, key):
        loss, bpc = denoising_loss(params, net, batch, key, cfg)
        loss = jax.lax.pmean(loss, axis_name='batch')
        bpc = jax.lax.pmean(bpc, axis_name='batch')
        return loss, bpc

    return jax.pmap(eval_fn, axis_name='batch')


def shard_batch(batch: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """Reshape batch from (B, ...) to (n_devices, B//n_devices, ...) for pmap."""
    B = batch.shape[0]
    per_device = B // n_devices
    return batch[:per_device * n_devices].reshape((n_devices, per_device, *batch.shape[1:]))


def replicate_to_devices(x, devices):
    """Replicate a PyTree across all devices."""
    n_devices = len(devices)
    return jax.tree_util.tree_map(lambda y: jnp.broadcast_to(y, (n_devices,) + y.shape), x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["wikitext2", "tinyshakespeare", "text"], default="wikitext2",
                    help="dataset source; use --data for a custom path")
    ap.add_argument("--data", type=str, default=None,
                    help="WikiText-2 directory, TinyShakespeare file, or custom text file")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256, help="global batch size across all devices")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--out", type=str, default="runs/charlm_distributed.pkl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity", action="store_true", help="tiny CPU test")
    ap.add_argument("--gpu", action="store_true", help="enable GPU-specific optimisations")
    args = ap.parse_args()

    # GPU setup
    if args.gpu:
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        print("[gpu] bfloat16 matmul precision enabled")

    devices = jax.devices()
    n_devices = len(devices)
    print(f"[device] {n_devices} device(s): {devices}")

    # Data
    if args.sanity:
        text = (
            "to be or not to be that is the question "
            "whether tis nobler in the mind to suffer "
        ) * 40
        args.dataset = "sanity"
        args.seq_len, args.hidden, args.layers, args.iters, args.batch = 24, 128, 2, 200, 64
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        train_ids, val_ids = split_train_val(ids, args.val_frac)
        total_chars = len(ids)
    else:
        if args.dataset == "wikitext2":
            data_dir = args.data or "data/wikitext-2-raw"
            train_path, valid_path, test_path = ensure_wikitext2_raw(data_dir)
            train_text = read_text(train_path)
            val_text = read_text(valid_path)
            test_text = read_text(test_path)
            # Character-level vocab construction over all splits avoids OOV chars at eval/generation time.
            tok = CharTokenizer.from_text(train_text + val_text + test_text)
            train_ids = tok.encode(train_text)
            val_ids = tok.encode(val_text)
            total_chars = len(train_ids) + len(val_ids) + len(tok.encode(test_text))
            print(f"[data] dataset=wikitext2  train_file={train_path}  valid_file={valid_path}")
        elif args.dataset == "tinyshakespeare":
            data_path = download_tinyshakespeare(args.data or "data/tinyshakespeare.txt")
            text = read_text(data_path)
            tok = CharTokenizer.from_text(text)
            ids = tok.encode(text)
            train_ids, val_ids = split_train_val(ids, args.val_frac)
            total_chars = len(ids)
            print(f"[data] dataset=tinyshakespeare  file={data_path}")
        elif args.dataset == "text":
            if not args.data:
                raise ValueError("--dataset text requires --data /path/to/text.txt")
            text = read_text(args.data)
            tok = CharTokenizer.from_text(text)
            ids = tok.encode(text)
            train_ids, val_ids = split_train_val(ids, args.val_frac)
            total_chars = len(ids)
            print(f"[data] dataset=text  file={args.data}")
        else:
            raise ValueError(f"unknown dataset: {args.dataset}")
    print(f"[data] vocab={tok.vocab_size}  total_chars={total_chars}  "
          f"train={len(train_ids)}  val={len(val_ids)}")

    stride = args.stride or max(args.seq_len // 4, 1)
    train_windows = make_windows(train_ids, args.seq_len, stride)
    val_windows = make_windows(val_ids, args.seq_len, stride)
    print(f"[data] train_windows={train_windows.shape}  val_windows={val_windows.shape}")

    # Config
    cfg = DiffusionLMConfig(
        vocab_size=tok.vocab_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        n_layers=args.layers,
    )
    print(f"[train] config={cfg}")
    print(f"[train] {args.iters} iters, global batch={args.batch}, lr={args.lr}, "
          f"warmup={args.warmup}, devices={n_devices}")

    # Build net & init params
    net = build_net(cfg)
    key = jax.random.PRNGKey(args.seed)
    params = init_params(net, cfg, key)

    # LR schedule + optimizer
    lr_schedule = cosine_lr_schedule(args.lr, args.warmup, args.iters)
    opt = optax.adam(lr_schedule)
    opt_state = opt.init(params)

    # Replicate params and opt_state across devices
    params = replicate_to_devices(params, devices)
    opt_state = replicate_to_devices(opt_state, devices)

    # Create pmapped train step and eval step
    train_step_pmap = create_train_step(net, cfg, opt)
    eval_step_pmap = create_eval_step(net, cfg)

    n_train = train_windows.shape[0]
    per_device_batch = args.batch // n_devices
    if per_device_batch < 1:
        raise ValueError(f"global batch {args.batch} is smaller than device count {n_devices}")
    if args.batch % n_devices != 0:
        effective_batch = per_device_batch * n_devices
        print(f"[warn] global batch {args.batch} is not divisible by {n_devices}; using {effective_batch}")
        args.batch = effective_batch
    history = []
    val_history = []
    t0 = time.time()

    for step in tqdm(range(args.iters), desc="training"):
        # Sample global batch and shard across devices
        key, kb = jax.random.split(key)
        idx = jax.random.randint(kb, (args.batch,), 0, n_train)
        batch = train_windows[idx]  # (global_batch, seq_len)
        batch_sharded = shard_batch(batch, n_devices)  # (n_devices, batch_per_device, seq_len)

        # Split key per device
        key, k_step = jax.random.split(key)
        keys_per_device = jax.random.split(k_step, n_devices)

        # Pmap step: inputs already sharded, outputs replicated (pmean)
        params, opt_state, loss, bpc = train_step_pmap(
            params, opt_state, batch_sharded, keys_per_device,
        )
        history.append(float(bpc[0]))  # bpc is same on all devices (pmean)

        # Validation
        if (step + 1) % args.val_every == 0 or step == 0:
            if len(val_windows) > 0:
                val_batch = val_windows[:args.batch]
                if val_batch.shape[0] < args.batch:
                    # Pad with repeats
                    val_batch = jnp.tile(val_batch, (args.batch // val_batch.shape[0] + 1, 1))[:args.batch]
                val_sharded = shard_batch(val_batch, n_devices)
                key, k_val = jax.random.split(key)
                val_keys = jax.random.split(k_val, n_devices)
                _, val_bpc = eval_step_pmap(params, val_sharded, val_keys)
                val_bpc_val = float(val_bpc[0])
                val_history.append((step + 1, val_bpc_val))
                progress_write(f"[step {step + 1}] train bpc={history[-1]:.4f}  val bpc={val_bpc_val:.4f}")
            else:
                progress_write(f"[step {step + 1}] train bpc={history[-1]:.4f}")

        # Checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = args.out.replace(".pkl", f"_step{step + 1}.pkl")
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            # Unreplicate params (take first device copy)
            params_unrep = jax.tree_util.tree_map(lambda x: x[0], params)
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "params": params_unrep,
                    "cfg": cfg,
                    "itos": tok.itos,
                    "history": history,
                    "val_history": val_history,
                    "step": step + 1,
                    "dataset": args.dataset,
                }, f)
            progress_write(f"[save] checkpoint -> {ckpt_path}")

    elapsed = time.time() - t0
    final_bpc = sum(history[-100:]) / min(len(history), 100)
    baseline = unigram_bits_per_char(train_ids, tok.vocab_size)
    print(f"\n[train] finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"[train] final train bpc ~ {final_bpc:.3f}  (unigram baseline {baseline:.3f})")
    if val_history:
        print(f"[val]   best val bpc = {min(v for _, v in val_history):.3f}")

    # Final save
    params_unrep = jax.tree_util.tree_map(lambda x: x[0], params)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({
            "params": params_unrep,
            "cfg": cfg,
            "itos": tok.itos,
            "history": history,
            "val_history": val_history,
            "step": args.iters,
            "dataset": args.dataset,
        }, f)
    print(f"[save] final checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
