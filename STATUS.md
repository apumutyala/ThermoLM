# Project status

ThermoLM is a small, validated codebase: a quadratic-Ising EBM and a chain-CRF
discrete-diffusion language model. Everything else is under `experimental/` and
is not imported by the package.

**July 2026 update:** vendored THRML refreshed to v0.1.3+ (upstream `bbbba9d`);
two confirmed defects fixed with regression tests (see "Fixed defects" below);
THRML-native maximum-likelihood training added and validated against exact
gradients; codebase audited line-by-line against Extropic's published
[thrml-skill](https://github.com/extropic-ai/thrml-skill) conventions. Suite:
**19 tests, all against ground truth.**

## ✅ Validated core

### Quadratic-Ising EBM + chromatic Gibbs

`E(x) = -β( Σ_{i<j} J_ij x_i x_j + Σ_i h_i x_i )`, x ∈ {-1,+1}.

- **Sampled correctly** by chromatic block Gibbs (validated against exact
  Boltzmann marginals by brute-force enumeration, n≤6).
- **Sampled via THRML** (`IsingSamplingProgram`) — the TSU-compatible path,
  validated against the same exact marginals. `THRMLIsingSampler` separates
  static structure (edges, nodes, colour blocks; built once) from traced
  arrays (J/h/β gathered with jnp indexing), so the THRML path is
  **jit/grad-safe** and usable inside compiled training steps.
- **Trained by contrastive divergence** with the correct two-term gradient
  (`∇L = E_data[∇E] − E_model[∇E]`), `stop_gradient` on negatives — pure-JAX
  or THRML negative phase.
- **Trained natively on THRML** by fully-visible maximum likelihood
  (`training/thrml_ml.py`): exact positive-phase moments from the data
  (v0.1.3 fully-visible `estimate_kl_grad`), THRML-sampled negative phase.
  The Monte-Carlo gradient matches the exact enumerated two-term KL gradient
  to <0.05, and training recovers a teacher model's exact moments
  (`tests/unit/test_thrml_ml.py`).

Evidence: `scripts/dtm_ising_demo.py` (3 parts),
`tests/unit/test_ising_correctness.py`, `tests/unit/test_thrml_ml.py`.

**Limitations:**
- Single, time-independent shared EBM; no per-timestep DTM chain over real data.
- Forward-coupling schedule γ(t) is a heuristic ramp, not derived from a known
  corruption kernel.
- CD/ML negative phases are short block-Gibbs chains: strongly-coupled
  (low-temperature) models mix slowly — the standard Gibbs caveat, now also
  stated in THRML's own docs. The sweep-budget experiment quantifies this.

### Fixed defects (July 2026, each with a regression test)

- **A1** — the old THRML wrapper converted the traced coupling matrix to
  NumPy inside the autodiff trace (`TracerArrayConversionError`), so
  THRML-backed *training* had never actually been possible. Fixed by the
  static-structure/traced-array split in `THRMLIsingSampler`.
- **A2** — `ForwardCoupling.to_thrml_factor_at_t` passed an (n,n) eye-matrix
  where a two-block `SpinEBMFactor` requires elementwise (n,) weights
  (THRML rejects the shape). Fixed; the factor now provably matches the
  coupling energy.
- Smaller: eval batching conflated seq_len with batch size; `set_seed`
  globally enabled float64; stale docstrings; dead WikiText-2 mirror;
  invalid even/odd colouring helper removed ("bipartite" patterns renamed
  "banded" — they are not bipartite for G12+).

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

**Research assets (July 2026):**
- `examples/lm_on_thrml.ipynb` — self-contained CPU notebook (THRML docs
  style, upstream-contributable): chain CRF as a THRML factor graph, fidelity
  vs sweep budget against the exact oracle, LM training, dual-path generation.
- `scripts/exp_sweep_budget.py` — the exactness anchor: TV distance between
  THRML block-Gibbs marginals and exact forward–backward marginals vs sweep
  budget, with the exact-FFBS noise floor; random potentials or a trained
  checkpoint's reverse step. `generate(..., thrml_warmup=k)` exposes the
  budget at generation time.
- `docs/RESEARCH_ROADMAP.md` — the three-aim scaling programme (budget curves,
  banded graphs beyond exact inference, latent-spin corrections).

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
