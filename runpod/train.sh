#!/bin/bash
# RunPod training script for ThermoLM Tier-1 char-level chain-CRF LM on TinyShakespeare.
# Run after setup.sh has completed.

set -e

source .venv/bin/activate

# GPU XLA performance flags (NVIDIA A100 / H100 / 4090)
export XLA_FLAGS="
    --xla_gpu_triton_gemm_any=True
    --xla_gpu_enable_latency_hiding_scheduler=true
"

# Optional: enable bfloat16 matmul precision (A100+ only; safe on 4090 too)
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16

echo "[train] XLA flags: ${XLA_FLAGS}"
echo "[train] JAX matmul precision: ${JAX_DEFAULT_MATMUL_PRECISION}"
echo "[train] starting TinyShakespeare training..."

# Recommended configs by GPU type (edit as needed)
# A100 80GB / H100 80GB: full speed, bfloat16
# RTX 4090 24GB: reduce batch if OOM, bfloat16 works

python scripts/train_tinyshakespeare.py \
    --gpu \
    --iters 5000 \
    --seq_len 128 \
    --hidden 256 \
    --layers 2 \
    --batch 64 \
    --lr 3e-3 \
    --val_frac 0.1 \
    --val_every 500 \
    --out runs/charlm_tinyshakespeare.pkl \
    --seed 42

echo "[train] training complete. evaluating..."

python scripts/eval_charlm.py \
    --ckpt runs/charlm_tinyshakespeare.pkl \
    --val_text data/tinyshakespeare.txt \
    --n_samples 8 \
    --n_steps 16 \
    --temperature 1.0 \
    --out_json runs/metrics_tinyshakespeare.json \
    --seed 42

echo "[train] all done. results in runs/"
