# ThermoLM JAX

Energy-based models for thermodynamic (TSU-style) sampling, built on Extropic's
[THRML](https://github.com/extropic-ai/thrml) library in JAX.

> **Scope note.** This repository has a small, **validated** core (quadratic
> Ising EBMs sampled and trained correctly, with a THRML-backed path) and a
> larger **exploratory** energy-diffusion language-model track that is a research
> sketch and **not** validated. This README leads with what works; see
> [`STATUS.md`](STATUS.md) for an honest, specific accounting of both.

## Validated: DTM / quadratic-Ising

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
| [`models/dtm.py`](thermolm_jax/models/dtm.py) | DTM scaffold (single shared EBM — see limitations) |
| [`scripts/dtm_ising_demo.py`](scripts/dtm_ising_demo.py) | End-to-end demo |

## Language model (Tier 1: chain-CRF discrete diffusion)

A small but genuine language model built on the validated core. Text is modelled
by **masked discrete diffusion whose reverse step is a linear-chain CRF** sampled
on THRML:

- A denoising network (`models/factor_weight_network.py`) conditioned on `(x_t, t)`
  emits the reverse-step energy as unary + nearest-neighbour pairwise categorical
  potentials — a linear-chain CRF over the clean vocabulary.
- The pairwise term is the EDLM-style "energy correction" that lets a reverse step
  model joint token structure a factorised denoiser can't; a chain is the simplest
  structure that is both **exactly trainable** and **TSU-samplable**.
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

# full char-level run (download TinyShakespeare to data/tinyshakespeare.txt)
python scripts/train_charlm.py --data data/tinyshakespeare.txt --iters 3000 \
    --seq_len 128 --hidden 256 --out runs/charlm.pkl
```

The CPU sanity run reaches ~1.0 bits/char vs a 3.9 unigram baseline and generates
text from the corpus fragments. Full TinyShakespeare training is a short
single-GPU/RunPod job (see `STATUS.md`); larger vocab/corpus and full
contrastive-divergence energy training are the planned next tiers.

## Exploratory (not validated)

The discrete / hybrid **energy-diffusion language-model** components
(`models/{discrete_edlm, discrete_energy, d3pm, fsq, thrml_discrete, hybrid_*}`,
`models/sampler.py`, the EDLM trainers, and the scripts under `experimental/`)
are research sketches with known correctness problems documented in
[`STATUS.md`](STATUS.md). They are not imported by the package by default and
their legacy tests live under `tests/experimental/` (excluded from `pytest`).
Import them by path if you want to experiment.

## Background

**Energy-based models** define `p(x) = exp(-E(x)) / Z`; sampling means finding
low-energy configurations. **Thermodynamic sampling** (Extropic's TSU) performs
block Gibbs updates physically, which requires energies that factor into local
terms over a sparse graph — exactly the quadratic Ising form above. **Chromatic
Gibbs** colours that graph so conditionally-independent variables update in
parallel.

## Limitations of the validated track (honest)

- The DTM here uses a **single, time-independent** shared EBM; it does not learn
  a per-timestep sequence of EBMs, so the full noise→data chain over real data
  is not demonstrated. The validated demo is the single-EBM case.
- The forward-coupling schedule `γ(t)` is a heuristic ramp, not derived from a
  specific corruption kernel.

## Research & references

- Extropic DTM: *An efficient probabilistic hardware architecture for
  diffusion-like models*, Jelinčič et al. (arXiv:2510.23972).
- THRML: https://github.com/extropic-ai/thrml (vendored, Apache-2.0).
- Contrastive divergence: Hinton (2002); Du & Mordatch (2019/2021).
- (Exploratory track) FSQ: Mentzer et al. (2023); masked diffusion: Sahoo et al.
  (2024), Lou et al. (2023); NVIDIA EDLM: Xu et al. (2024).

External reference repos (NVIDIA Energy-Diffusion-LLM, dtm-replication) are not
bundled to keep the clone small; see the links above.

## License

MIT (this repository). Vendored THRML retains its Apache-2.0 license under
`external/thrml/`.

## Author

Apuroop Mutyala — Georgia Institute of Technology, MS Computer Engineering.
