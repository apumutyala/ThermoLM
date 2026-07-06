#!/usr/bin/env bash
# RunPod GPU run script for ThermoLM.
# Paste this into a RunPod A100/H100 pod terminal after creating the pod.
# Designed for fresh pods (Ubuntu/CUDA/JAX preinstalled or installable).

set -euo pipefail

REPO_URL="https://github.com/apumutyala/ThermoLM.git"
REPO_DIR="ThermoLM"
RUNPOD_OUTPUT_DIR="$HOME/thermolm_runpod_outputs_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# 1. Base environment
# ---------------------------------------------------------------------------
apt-get update -qq
apt-get install -y -qq git unzip rsync htop nvtop

python -m venv "$HOME/thermolm_venv"
source "$HOME/thermolm_venv/bin/activate"

pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2. Clone repo
# ---------------------------------------------------------------------------
rm -rf "$REPO_DIR"
git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
# Vendored THRML first, then ThermoLM with LM + viz + dev extras.
pip install -e external/thrml
pip install -e ".[llm,viz,dev]"

# Install JAX with CUDA if not already present (adjust CUDA version as needed).
# For CUDA 12.x; use jax[cuda11] for CUDA 11.x.
# pip install --upgrade "jax[cuda12]" >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 4. Sanity checks
# ---------------------------------------------------------------------------
echo "=== JAX devices ==="
python -c "import jax; print('devices:', jax.devices()); print('local:', jax.local_devices())"

echo "=== THRML import ==="
python -c "import thrml; from thrml.models import estimate_kl_grad; print('thrml OK')"

echo "=== CPU test suite (quick) ==="
pytest -q

# ---------------------------------------------------------------------------
# 5. Disk layout for outputs
# ---------------------------------------------------------------------------
mkdir -p "$RUNPOD_OUTPUT_DIR"/{runs,results,logs}

# ---------------------------------------------------------------------------
# 6. Run A: TinyShakespeare full training (single GPU)
# ---------------------------------------------------------------------------
echo "=== [A] TinyShakespeare training ==="
python scripts/train_tinyshakespeare.py \
    --iters 3000 --seq_len 128 --hidden 256 --batch 64 --gpu \
    --out runs/charlm_tinyshakespeare.pkl \
    | tee "$RUNPOD_OUTPUT_DIR/logs/train_tinyshakespeare.log"

cp runs/charlm_tinyshakespeare.pkl "$RUNPOD_OUTPUT_DIR/runs/"

echo "=== [A] TinyShakespeare eval ==="
python scripts/eval_charlm.py --ckpt runs/charlm_tinyshakespeare.pkl \
    | tee "$RUNPOD_OUTPUT_DIR/logs/eval_tinyshakespeare.log"

echo "=== [A] Generation (exact + THRML) ==="
python scripts/generate_charlm.py --ckpt runs/charlm_tinyshakespeare.pkl --n 16 \
    | tee "$RUNPOD_OUTPUT_DIR/logs/generate_exact.log"
python scripts/generate_charlm.py --ckpt runs/charlm_tinyshakespeare.pkl --n 8 --thrml \
    | tee "$RUNPOD_OUTPUT_DIR/logs/generate_thrml.log"

# ---------------------------------------------------------------------------
# 7. Run B: Sweep-budget oracle on trained checkpoint (the signature plot)
#     Three-way comparison: exact FFBS vs. JAX-GPU chromatic Gibbs vs. THRML
#     block Gibbs, at matched wall-clock and matched sweep budget. Outputs
#     both TV-vs-sweeps and TV-vs-time curves.
# ---------------------------------------------------------------------------
echo "=== [B] Sweep-budget oracle on trained checkpoint ==="
python scripts/exp_sweep_budget.py \
    --ckpt runs/charlm_tinyshakespeare.pkl \
    --out results/sweep_budget_trained \
    | tee "$RUNPOD_OUTPUT_DIR/logs/exp_sweep_budget_trained.log"

cp results/sweep_budget_trained.csv "$RUNPOD_OUTPUT_DIR/results/"
cp results/sweep_budget_trained_sweeps.png "$RUNPOD_OUTPUT_DIR/results/"
cp results/sweep_budget_trained_time.png "$RUNPOD_OUTPUT_DIR/results/"

# ---------------------------------------------------------------------------
# 8. Run C: WikiText-2 distributed training (2x A100)
#     Skipped automatically if fewer than 2 GPUs are visible.
# ---------------------------------------------------------------------------
N_GPU=$(python -c "import jax; print(len(jax.devices()))")
if [ "$N_GPU" -ge 2 ]; then
    echo "=== [C] WikiText-2 distributed training ($N_GPU GPUs) ==="
    python scripts/train_distributed.py \
        --dataset wikitext2 --gpu \
        --iters 20000 --seq_len 256 --hidden 512 --layers 6 --batch 256 \
        --out runs/charlm_wikitext2.pkl \
        | tee "$RUNPOD_OUTPUT_DIR/logs/train_wikitext2.log"
    cp runs/charlm_wikitext2.pkl "$RUNPOD_OUTPUT_DIR/runs/"
else
    echo "=== [C] SKIPPED: only $N_GPU GPU(s) visible; WikiText-2 distributed needs 2 ==="
fi

# ---------------------------------------------------------------------------
# 9. Package outputs for download
# ---------------------------------------------------------------------------
echo "=== Output bundle ==="
ls -lhR "$RUNPOD_OUTPUT_DIR" > "$RUNPOD_OUTPUT_DIR/MANIFEST.txt"
tar -czf "$RUNPOD_OUTPUT_DIR.tar.gz" -C "$HOME" "$(basename "$RUNPOD_OUTPUT_DIR")"

echo "Done. Download with:"
echo "  scp root@<pod-ip>:$RUNPOD_OUTPUT_DIR.tar.gz ."
echo "Or use RunPod Cloud Sync / Volume Download."
