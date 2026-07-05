# ThermoLM JAX

Energy-based models for thermodynamic (TSU-style) sampling, built on Extropic's
[THRML](https://github.com/extropic-ai/thrml) library in JAX.

> **Scope note.** This repository is intentionally small: only the validated
> core is imported by the package by default. An older experimental track lives
> under `experimental/` and is not imported. See [`STATUS.md`](STATUS.md) for a
> plain, specific accounting.

## Validated: Quadratic-Ising EBM + chromatic Gibbs

A quadratic Ising energy-based model

```
E(x) = -β( Σ_{i<j} J_ij x_i x_j + Σ_i h_i x_i ),   x_i ∈ {-1, +1}
```

that is:

- **Sampled correctly** by chromatic block Gibbs. Each colour class is a true
  independent set (colouring derived from the actual interaction graph), so a
  whole class is resampled in parallel — the structure that maps onto
  thermodynamic hardware. The site conditional is `P(x_i=+1)=σ(2β f_i/T)` with
  local field `f_i = Σ_j J_ij x_j + h_i`.
- **Sampled via THRML** through the library's own `IsingSamplingProgram` (the
  TSU-compatible path), validated to agree with the JAX sampler and with exact
  marginals.
- **Trained by contrastive divergence** with the correct two-term gradient
  (`∇L = E_data[∇E] − E_model[∇E]`), real data in the positive phase, and
  `stop_gradient` on negative samples.

**This is demonstrated, not asserted** — see [Validation](#validation).

### Quickstart (CPU, ~1 minute)

THRML is vendored under `external/thrml` (Apache-2.0), so no network or
submodule init is required.

```bash
git clone https://github.com/apumutyala/ThermoLM.git
cd ThermoLM

python -m venv .venv && source .venv/bin/activate   # or your env of choice
pip install -e external/thrml        # vendored THRML
pip install -e ".[dev]"              # this package + test extras

# 1) the end-to-end demo: sampler-vs-exact + CD training on toy data
python scripts/dtm_ising_demo.py

# 2) the validated test suite
pytest
```

Expected demo output (abridged):

```
[1] Chromatic Gibbs vs EXACT Boltzmann marginals (random Ising, n=6)
    exact <s_i>: [-0.89 -0.80 -0.30 -0.52 -0.66 -0.71]
    JAX   <s_i>: ...  max|err| = 0.014
    THRML <s_i>: ...  max|err| = 0.033
[2] Contrastive-divergence training on a toy bimodal distribution (n=8)
    ...
    fraction of samples exactly on a data mode: 0.68
```

### Minimal API

```python
import jax, jax.numpy as jnp, optax, equinox as eqx
from thermolm_jax import QuadraticEBM, QuadraticEBMConfig, CDConfig
from thermolm_jax import chromatic_gibbs_sample, greedy_coloring, contrastive_divergence_step
from thermolm_jax.sampling.chromatic_gibbs import color_masks_from_colors

key = jax.random.PRNGKey(0)
ebm = QuadraticEBM(QuadraticEBMConfig(n_vars=8, beta=1.0, init_scale=0.01), key)
ebm = ebm.set_connectivity(jnp.ones((8, 8), bool))

# precompute colour masks once -> the CD step is jit-compiled
cmasks = color_masks_from_colors(greedy_coloring(ebm.connectivity_mask), 8)

opt = optax.adam(0.05)
opt_state = opt.init(eqx.filter(ebm, eqx.is_inexact_array))
cfg = CDConfig(k=1, n_gibbs_steps=20, temperature=1.0)

# training step (x_data: (batch, 8) spins in {-1, +1})
ebm, opt_state, loss, info = contrastive_divergence_step(
    ebm, opt, opt_state, x_data, key, cfg, color_masks=cmasks
)

# sampling
init = jax.random.randint(key, (256, 8), 0, 2) * 2 - 1
samples, _ = chromatic_gibbs_sample(ebm, init.astype(jnp.float32), 300, key, color_masks=cmasks)
# THRML-backed path: chromatic_gibbs_sample(..., use_thrml=True)
```

### Validation

`tests/unit/test_ising_correctness.py` (run with `pytest`) checks, against ground
truth rather than just shapes:

- chromatic Gibbs reproduces the **exact** enumerated Boltzmann marginals
  (first and second moments), for both the JAX and THRML sampling paths;
- CD training drives model moments toward the data moments;
- the energy is single-counted and diagonal-free; temperature scaling behaves;
- the forward-coupling sign favours aligned states; connectivity patterns are
  nested; the graph colouring is valid.

### Key modules (validated track)

| File | Role |
|------|------|
| [`models/quadratic_ebm.py`](thermolm_jax/models/quadratic_ebm.py) | Quadratic Ising energy `E(x)` |
| [`models/connectivity.py`](thermolm_jax/models/connectivity.py) | Sparse connectivity patterns |
| [`sampling/chromatic_gibbs.py`](thermolm_jax/sampling/chromatic_gibbs.py) | Chromatic block Gibbs (+ colouring) |
| [`models/thrml_quadratic.py`](thermolm_jax/models/thrml_quadratic.py) | THRML `IsingSamplingProgram` path |
| [`training/contrastive_divergence.py`](thermolm_jax/training/contrastive_divergence.py) | CD loss/step |
| [`scripts/dtm_ising_demo.py`](scripts/dtm_ising_demo.py) | End-to-end demo |

## Language model (Tier 1: chain-CRF discrete diffusion)

A small language model built on the validated core. Text is modelled by **masked
discrete diffusion whose reverse step is a linear-chain CRF** sampled on THRML:

- A denoising network (`models/factor_weight_network.py`) conditioned on `(x_t, t)`
  emits the reverse-step energy as unary + nearest-neighbour pairwise categorical
  potentials — a linear-chain CRF over the clean vocabulary.
- The pairwise term captures joint token structure at each reverse step. Unlike
  standard diffusion LMs that predict tokens independently (the "factorization
  mismatch"), the chain-CRF uses **exact forward–backward inference** for both
  training and generation. This is the key property that EDLM-style energy
  corrections add to standard diffusion models — here it is exact, not
  approximate.
- **Training** is exact conditional ML of the chain CRF (`models/chain_crf.py`,
  forward algorithm for `log Z` — no MCMC).
- **Generation** samples the chain CRF jointly at each reverse step, either exactly
  (forward-filter backward-sample) or on **THRML** (`sampling/chain_mrf_thrml.py`,
  the TSU path), validated against the exact forward–backward marginals.

```bash
# offline sanity run (tiny, CPU, embedded corpus)
python scripts/train_charlm.py --sanity --out runs/charlm.pkl
python scripts/generate_charlm.py --ckpt runs/charlm.pkl --n 8          # exact FFBS
python scripts/generate_charlm.py --ckpt runs/charlm.pkl --n 4 --thrml  # TSU path

# full TinyShakespeare training (auto-downloads dataset)
python scripts/train_tinyshakespeare.py --iters 3000 --seq_len 128 --hidden 256 \
    --out runs/charlm_tinyshakespeare.pkl
python scripts/eval_charlm.py --ckpt runs/charlm_tinyshakespeare.pkl

# distributed WikiText-2 training on 2×A100 (see RunPod section below)
python scripts/train_distributed.py --dataset wikitext2 --gpu \
    --iters 20000 --seq_len 256 --hidden 512 --layers 6 --batch 256 \
    --out runs/charlm_wikitext2.pkl
```

The CPU sanity run trains on a small embedded corpus and generates corpus-like text.
Full TinyShakespeare training is a short single-GPU job. WikiText-2 distributed
training targets semi-coherent generation on a real corpus. See `STATUS.md` for
expected GPU hours and scaling limits.

## Architecture notes: why chain-CRF for character-level LM

The chain-CRF architecture makes a specific trade-off: it uses **exact joint
inference** over the sequence (via the forward algorithm), which captures
pairwise dependencies between adjacent tokens, but limits the inference to a
chain structure to keep it tractable. This is well-suited for character-level
language modeling because:

- **Small vocabulary** (V~65–100 for char-level) makes the O(L·V²) forward
  algorithm cheap — ~500k ops per sequence for L=128, V=65.
- **Exact inference** means no MCMC approximation or sampling error in training.
- **Pairwise dependencies** are the minimal structure needed to capture n-gram
  and local morphological patterns in text.
- **The THRML path** maps the chain CRF onto thermodynamic hardware via
  chromatic 2-colouring, validated against exact JAX inference.

The limitation is explicit: for large subword vocabularies (V~10k), the exact
forward algorithm becomes prohibitive. Scaling to BPE-level models would require
approximate inference (e.g., factorised output, or an EDLM-style energy
correction on top of a tractable base distribution). This is a known and stated
architectural boundary, not a hidden gap.

### Relationship to EDLM and related work

Recent work (Xu et al., ICLR 2025; "Energy-Based Diffusion Language Models")
proposes adding an energy-based correction to standard diffusion LMs to fix the
"factorization mismatch" — the problem that standard diffusion models predict
tokens independently at each denoising step, ignoring sequence-level correlations.

**The chain-CRF already implements this correction exactly** for the small-vocab
regime: the pairwise CRF potentials capture the joint token structure that EDLM
adds via a residual energy term. The EDLM formulation would become relevant if we
scale to larger vocabularies where exact CRF inference is intractable. For
current character-level experiments, the chain-CRF is the principled way to do
what EDLM approximates.

## Training & evaluation pipeline

### Scripts

| Script | Purpose | Key args |
|--------|---------|----------|
| `scripts/train_tinyshakespeare.py` | Single-device training on TinyShakespeare | `--iters`, `--seq_len`, `--hidden`, `--batch`, `--gpu` |
| `scripts/train_distributed.py` | Multi-GPU (`pmap`) training on WikiText-2 or custom text | `--dataset`, `--iters`, `--seq_len`, `--hidden`, `--layers`, `--batch`, `--gpu` |
| `scripts/eval_charlm.py` | Evaluation: held-out bits/char + generation samples | `--ckpt`, `--val_text`, `--n_samples`, `--temperature` |
| `scripts/generate_charlm.py` | Standalone generation from checkpoint | `--ckpt`, `--n`, `--thrml` |
| `scripts/dtm_ising_demo.py` | Ising EBM demo (exact marginals + CD training) | — |

### RunPod deployment (GPU training)

The `runpod/` directory contains one-command deployment scripts:

```bash
# One-time setup
bash runpod/setup.sh

# TinyShakespeare single-GPU run
bash runpod/train.sh

# WikiText-2 distributed 2×A100 run
bash runpod/train_distributed.sh
```

Both scripts set XLA GPU performance flags (`--xla_gpu_triton_gemm_any=True`,
`--xla_gpu_enable_latency_hiding_scheduler=true`) and bfloat16 matmul
precision for A100/H100 Tensor Cores.

## Limitations of the validated track (honest)

- The DTM here uses a **single, time-independent** shared EBM; it does not learn
  a per-timestep sequence of EBMs, so the full noise→data chain over real data
  is not demonstrated. The validated demo is the single-EBM case.
- The forward-corruption schedule `q_xt` is a simple independent masking process,
  not a derived corruption kernel with a learned schedule.
- The chain-CRF exact forward algorithm is **O(L·V²)**. It is cheap for the small
  character vocabularies used here (V~65–100), but scaling to large subword
  vocabularies (V~10k) would require approximate inference or a different
  architecture. This is a known and stated architectural boundary, not a hidden
  gap.
- The language model is a research demonstration, not a production system. No
  checkpointing resume, no mixed-precision training beyond JAX's bfloat16
  matmul, no gradient clipping or advanced optimisation.

## Exploratory track (not validated)

An older experimental track (FSQ, discrete/hybrid energy-diffusion LM, continuous
autoencoder, etc.) lives under `experimental/` and is **not imported by the package
by default**. It has known correctness issues documented in `STATUS.md`. Import
by path if you want to experiment.

## Research & references

- Extropic DTM: *An efficient probabilistic hardware architecture for
  diffusion-like models*, Jelinčič et al. (arXiv:2510.23972).
- THRML: https://github.com/extropic-ai/thrml (vendored, Apache-2.0).
- Contrastive divergence: Hinton (2002); Du & Mordatch (2019/2021).
- Masked diffusion: Sahoo et al. (2024), Lou et al. (2023).
- EDLM: Xu et al., *Energy-Based Diffusion Language Models for Text Generation*,
  ICLR 2025. The chain-CRF implements exact joint inference in the small-vocab
  regime where EDLM adds approximate energy corrections.
- Character-level transformers: Al-Rfou et al., *Character-Level Language Modeling
  with Deeper Self-Attention*, AAAI 2019.

## License

MIT (this repository). Vendored THRML retains its Apache-2.0 license under
`external/thrml/`.

## Author

Apuroop Mutyala — Georgia Institute of Technology, MS Computer Engineering.
