# ThermoLM RunPod Deployment Guide

This folder contains everything needed to train and evaluate ThermoLM on a RunPod GPU instance.

## Quick start (one-time per pod)

```bash
cd /workspace/ThermoLM
bash runpod/setup.sh
```

This installs JAX with the correct CUDA version, vendored THRML, and the package.

## Training

```bash
bash runpod/train.sh
```

This sets GPU XLA flags (`--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true`) and runs the full TinyShakespeare pipeline:
- `scripts/train_tinyshakespeare.py` with bfloat16, 5k iters, 128 seq, 256 hidden
- `scripts/eval_charlm.py` with held-out bits/char and generation samples

Results are saved to `runs/charlm_tinyshakespeare.pkl` and `runs/metrics_tinyshakespeare.json`.

## 2×A100 WikiText-2 distributed training

```bash
bash runpod/train_distributed.sh
```

This runs `scripts/train_distributed.py` with JAX `pmap` data parallelism:
- dataset: WikiText-2 raw, character-level
- model: 6 layers, 512 hidden, seq_len 256
- batch: 256 global (128 examples per A100 on a 2-GPU pod)
- optimizer: Adam with linear warmup + cosine decay
- checkpoints: `runs/charlm_wikitext2_2xa100_step*.pkl`
- final checkpoint: `runs/charlm_wikitext2_2xa100.pkl`

The first run downloads and extracts WikiText-2 raw under `data/wikitext-2-raw/`.

## Recommended GPU configs

| GPU | `--iters` | `--batch` | `--hidden` | `--seq_len` | Notes |
|-----|-----------|-----------|------------|-------------|-------|
| H100 80GB | 5000 | 64 | 256 | 128 | Full speed, ~8–12 min |
| A100 80GB | 5000 | 64 | 256 | 128 | Full speed, ~10–15 min |
| RTX 4090 24GB | 5000 | 64 | 256 | 128 | Reduce batch to 32 if OOM, ~20–30 min |
| RTX 3060 12GB | 3000 | 16 | 128 | 64 | Small config, ~2–3 hours |

If you hit OOM, lower `--batch` or `--hidden` first.

## WikiText-2 scaling (char-level)

For WikiText-2 char-level (~2.1M tokens, V~100, seq_len=256, hidden=512, layers=4, batch=128, ~20k–50k iters):
- **H100**: ~1.5–3 hours
- **A100**: ~2–4 hours
- **RTX 4090**: ~6–10 hours (reduce batch to 64 if needed)

Note: the Tier-1 chain-CRF exact forward algorithm is O(L·V²). It is efficient for small character vocabularies (V~65) but becomes prohibitive at large subword vocabularies (V~10k). Scaling to BPE requires a different architecture.

## Manual commands

```bash
source .venv/bin/activate

export XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16

python scripts/train_tinyshakespeare.py --gpu --iters 5000 --seq_len 128 --hidden 256 --batch 64 --out runs/charlm.pkl
python scripts/eval_charlm.py --ckpt runs/charlm.pkl --val_text data/tinyshakespeare.txt --out_json runs/metrics.json

# 2×A100 WikiText-2 distributed run
python scripts/train_distributed.py --dataset wikitext2 --gpu --iters 20000 --seq_len 256 --hidden 512 --layers 6 --batch 256 --out runs/charlm_wikitext2_2xa100.pkl
```

## Troubleshooting

- **JAX not finding GPU**: ensure `jax[cuda12]` (or `cuda11`) is installed, not CPU-only `jax`.
- **OOM during training**: lower `--batch` or `--hidden`.
- **THRML not found**: run `pip install -e external/thrml` inside the venv.
- **Slow compilation on first run**: JAX compiles XLA kernels on first execution; subsequent runs are fast.
