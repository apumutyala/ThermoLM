#!/bin/bash
# RunPod 2×A100 distributed training script for ThermoLM Tier-1 char-level chain-CRF LM.
# Run after setup.sh has completed.

set -e

source .venv/bin/activate

# JAX/XLA GPU performance flags for NVIDIA A100/H100.
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"
export JAX_DEFAULT_MATMUL_PRECISION=bfloat16

echo "[train-distributed] visible accelerators:"
python - <<'PY'
import jax
print(jax.devices())
print("local_device_count =", jax.local_device_count())
PY

echo "[train-distributed] starting WikiText-2 char-level pmap training..."

python scripts/train_distributed.py \
    --dataset wikitext2 \
    --gpu \
    --iters 20000 \
    --seq_len 256 \
    --hidden 512 \
    --layers 6 \
    --batch 256 \
    --lr 3e-3 \
    --warmup 1000 \
    --val_every 2000 \
    --save_every 5000 \
    --out runs/charlm_wikitext2_2xa100.pkl \
    --seed 42

echo "[train-distributed] training complete. Sampling/evaluating checkpoint..."

python scripts/eval_charlm.py \
    --ckpt runs/charlm_wikitext2_2xa100.pkl \
    --val_text data/wikitext-2-raw/wikitext-2-raw/wiki.valid.raw \
    --n_samples 8 \
    --n_steps 24 \
    --temperature 1.0 \
    --out_json runs/metrics_wikitext2_2xa100.json \
    --seed 42

echo "[train-distributed] all done. Results in runs/"
