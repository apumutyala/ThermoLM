#!/bin/bash
# RunPod setup for ThermoLM JAX training
# Run this once after provisioning the pod.

set -e

echo "[setup] updating packages and installing python3-venv..."
apt-get update -qq && apt-get install -y -qq python3-venv git wget

# Detect CUDA version and install matching JAX
CUDA_MAJOR=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\).*/\1/p')
echo "[setup] detected CUDA ${CUDA_MAJOR}"

if [ -z "$CUDA_MAJOR" ]; then
    echo "[warn] nvcc not found; assuming CUDA 12 and installing jax[cuda12]"
    CUDA_MAJOR=12
fi

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install JAX with the correct CUDA version
echo "[setup] installing JAX for CUDA ${CUDA_MAJOR}..."
if [ "$CUDA_MAJOR" = "12" ]; then
    pip install -q "jax[cuda12]>=0.4.20"
elif [ "$CUDA_MAJOR" = "11" ]; then
    pip install -q "jax[cuda11]>=0.4.20"
else
    echo "[error] unsupported CUDA version: ${CUDA_MAJOR}"
    exit 1
fi

# Install core dependencies
echo "[setup] installing core dependencies..."
pip install -q flax>=0.7.0 optax>=0.1.0 equinox>=0.11.0 numpy>=1.24.0

# Install vendored THRML
echo "[setup] installing vendored THRML..."
pip install -q -e external/thrml

# Install this package
echo "[setup] installing ThermoLM..."
pip install -q -e .

echo "[setup] done. Activate the environment with: source .venv/bin/activate"
