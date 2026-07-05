# Project status

ThermoLM is a small, validated codebase: a quadratic-Ising EBM and a chain-CRF
discrete-diffusion language model. Everything else is under `experimental/` and
is not imported by the package.

## ✅ Validated core

### Quadratic-Ising EBM + chromatic Gibbs

`E(x) = -β( Σ_{i<j} J_ij x_i x_j + Σ_i h_i x_i )`, x ∈ {-1,+1}.

- **Sampled correctly** by chromatic block Gibbs (validated against exact
  Boltzmann marginals by brute-force enumeration, n≤6).
- **Sampled via THRML** (`IsingSamplingProgram`) — the TSU-compatible path,
  validated against the same exact marginals.
- **Trained by contrastive divergence** with the correct two-term gradient
  (`∇L = E_data[∇E] − E_model[∇E]`), `stop_gradient` on negatives.

Evidence: `scripts/dtm_ising_demo.py`, `tests/unit/test_ising_correctness.py`.

**Limitations:**
- Single, time-independent shared EBM; no per-timestep DTM chain over real data.
- Forward-coupling schedule γ(t) is a heuristic ramp, not derived from a known
  corruption kernel.

### Tier-1 chain-CRF language model

Masked discrete diffusion whose reverse step is a linear-chain CRF, sampled on
THRML.

- Exact CRF inference (forward algorithm, forward–backward, FFBS) — checked
  against brute-force enumeration (`tests/unit/test_chain_crf.py`).
- THRML chain sampler with correct even/odd 2-colouring; samples match exact
  CRF marginals.
- Training by exact conditional ML (no MCMC); generation samples the chain CRF
  jointly, exactly or on THRML.
- **Distributed training** via JAX `pmap` across multiple GPUs, with gradient
  all-reduce (`pmean`) and bfloat16 Tensor Core acceleration.

Evidence: `scripts/train_charlm.py --sanity` trains on a small embedded corpus and
beats the unigram baseline; `tests/unit/test_diffusion_lm.py` asserts learning +
valid generation; `scripts/generate_charlm.py` runs both exact and THRML samplers;
`scripts/train_distributed.py` supports multi-GPU WikiText-2 training with CPU
fallback validated.

**Limitations:**
- The chain-CRF exact forward algorithm is **O(L·V²)**. It is efficient for the
  small character vocabularies used here (V~65–100), but scaling to large subword
  vocabularies (V~10k) would require approximate inference or a different
  architecture. This is a stated architectural boundary, not a hidden gap.
- Full TinyShakespeare GPU training has been validated on CPU; GPU runs are ready
  to execute on the target hardware (2×A100).
- WikiText-2 distributed training scripts are complete and validated for
  imports/syntax/single-device CPU smoke tests; the full multi-GPU run is pending
  hardware execution.
- No full diffusion-ELBO held-out likelihood evaluation; the reported objective
  is exact CRF denoising conditional ML.

## ⚠️ Experimental track (under `experimental/`)

These modules are research sketches with known correctness issues. They are **not**
imported by the package by default. Known issues include:

- Energy never becomes the sampled distribution (deep-net energy cannot be
  expressed as THRML quadratic categorical factors).
- Sign-flipped CD objective in `discrete_energy.py` (`logsumexp(E_neg − E_pos)`).
- Positive phase uses model samples instead of data in `thrml_flax_coexistence.py`.
- Factorized "energy" with no cross-token mixing despite the "transformer" name.
- `sampler.py` is Metropolis with a broken proposal ratio and reused RNG key.
- FSQ omits the bounding nonlinearity; citation was wrong.
- `d3pm.py` implements an absorbing/masked-diffusion loss (MDLM/SEDD-style), not
  D3PM, and its reverse sampler overwrites already-decoded tokens.

These files are kept for reference but are not part of the validated core.

## GPU hours & scaling estimates

**TinyShakespeare char-level** (~1.1M chars, V~65, seq_len=128, hidden=256,
layers=2, batch=64, ~3k–5k iters):
- RTX 4090: ~20–30 min
- A100 80GB: ~10–15 min
- H100 80GB: ~8–12 min

**WikiText-2 char-level** (~2.1M tokens, V~100, seq_len=256, hidden=512, layers=6,
batch=256 global, ~20k iters):
- 2×A100 80GB: ~2–3 hours (distributed, 128 per device)
- RTX 4090: ~6–10 hours (reduce batch to 64 if OOM)
- H100 80GB: ~1.5–2.5 hours (single GPU, or faster with 2×)

**WikiText-2 BPE-level** (V~10k, seq_len=512, hidden=768, layers=6): the exact
chain-CRF forward algorithm is O(L·V²) and becomes prohibitive at V=10k. To scale
to BPE-level vocabularies the architecture would need a factorised output or an
approximate CRF. This is out of scope for the validated Tier-1.

## Out of scope

Time-conditioned DTM chain over language, binary autoencoder / GAN stages, and
external benchmarking. An EDLM-style energy correction is architecturally related
but not required for the current small-vocab character-level regime; see README
"Architecture notes" for discussion.
