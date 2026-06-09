# Project status — what is validated and what is not

ThermoLM explores two ideas for thermodynamic (TSU) sampling on Extropic's
**THRML** library: Extropic's **Denoising Thermodynamic Model (DTM)** and an
NVIDIA-style **Energy Diffusion Language Model (EDLM)**. The codebase is split
into a validated, supported track and an earlier exploratory track that is not
yet correct. This file states plainly which is which.

## ✅ Validated (the supported track): DTM / quadratic-Ising

A quadratic Ising energy-based model,
`E(x) = -β( Σ_{i<j} J_ij x_i x_j + Σ_i h_i x_i )`, x ∈ {-1,+1}, that is:

- **Sampled correctly** by chromatic block Gibbs. The conditional is
  `P(x_i=+1) = σ(2β f_i / T)` with local field `f_i = Σ_j J_ij x_j + h_i`; the
  graph colouring is derived from the actual interaction graph
  (`greedy_coloring`) so each colour class is a genuine independent set.
- **Sampled via THRML** through the library's own `IsingSamplingProgram`
  (`THRMLQuadraticEBM.sample`) — the TSU-compatible path.
- **Trained by contrastive divergence** with the correct two-term gradient
  (`∇L = E_data[∇E] − E_model[∇E]`), real data in the positive phase, and
  `stop_gradient` on negative samples.

**Evidence (reproducible):**
- `scripts/dtm_ising_demo.py` — sampler reproduces exact Boltzmann marginals of a
  small random model, and CD training concentrates samples on toy data modes.
- `tests/unit/test_ising_correctness.py` — exact-marginal agreement (JAX and
  THRML paths), CD moment-matching, single-counted energy, forward-coupling
  sign, connectivity nesting, valid colouring.

**Known limitations of this track (honest, not bugs):**
- The DTM uses a single, **time-independent** shared EBM; it does not yet learn a
  per-timestep sequence of EBMs, so the full noise→data chain over real data is
  not demonstrated. The validated demo is the single-EBM case.
- The forward-coupling schedule `γ(t)` is a heuristic ramp, not derived from a
  specific forward corruption kernel.

## ✅ Validated (Tier 1): chain-CRF discrete-diffusion language model

A small language model built on the validated core: masked discrete diffusion
whose reverse step is a linear-chain CRF (unary + nearest-neighbour pairwise),
sampled on THRML.

- Exact CRF inference (`models/chain_crf.py`): `log Z` (forward algorithm),
  log-likelihood, marginals (forward–backward), and exact FFBS sampling — all
  checked against brute-force enumeration (`tests/unit/test_chain_crf.py`).
- THRML chain sampler (`sampling/chain_mrf_thrml.py`) with a correct even/odd
  2-colouring; its samples match the exact CRF marginals.
- Training by exact conditional ML (no MCMC); generation samples the chain CRF
  jointly, exactly or on THRML.

**Evidence:** `scripts/train_charlm.py --sanity` reaches ~1.0 bits/char vs a ~3.9
unigram baseline on CPU and generates corpus-like text; `scripts/generate_charlm.py`
runs both the exact and THRML reverse-step samplers;
`tests/unit/test_diffusion_lm.py` asserts learning + valid generation.

**Deferred (needs compute / future work):**
- Full TinyShakespeare char-level training (short single-GPU/RunPod run) — code
  and configs are ready; only the run is pending compute.
- Tier 2 (small-BPE WikiText-2 scaling + ablation), Tier 3 (FSQ latent codes),
  and M2 (full contrastive-divergence energy training on a denser graph). See the
  roadmap.
- The training objective is exact CRF *denoising* conditional ML; a full
  diffusion-ELBO held-out likelihood eval is not yet implemented.

## ⚠️ Exploratory / NOT validated: EDLM language-model track

These modules are research sketches kept for reference. They are **not** imported
by the package by default; import them by path if you want to experiment. Known
issues to resolve before they can be trusted:

- **Energy is never the sampled distribution.** `discrete_edlm.py` /
  `thrml_discrete.py`: the trained neural energy is not converted into the THRML
  factor graph that actually produces samples (a deep-net energy cannot be
  expressed as THRML's quadratic categorical factors), and the call signatures
  are mismatched. Generation does not depend on the trained energy.
- **Sign-flipped CD objective** in `discrete_energy.py` (`logsumexp(E_neg − E_pos)`)
  trains the model to raise data energy.
- **Positive phase uses model samples, not data** in
  `thrml_flax_coexistence.py` (`train_step_flax_only`).
- **Factorized "energy"**: `DiscreteEnergyFunction` is position-wise MLPs with no
  cross-token mixing, so it models no interactions despite the "transformer" name.
- **`sampler.py`** is labelled Gibbs but is a Metropolis sampler with a fixed
  `N(0,I)` independence proposal, missing the proposal ratio and reusing the RNG
  key; it does not target `exp(-E)`. It also imports but never uses THRML.
- **FSQ** (`fsq.py`) omits the bounding nonlinearity, so codes collapse to the
  grid boundary; the citation was also wrong (FSQ is Mentzer et al. 2023).
- **`d3pm.py`** implements an absorbing/masked-diffusion loss (MDLM/SEDD-style),
  not D3PM, and its reverse sampler overwrites already-decoded tokens.

Affected files (non-exhaustive): `models/{discrete_edlm, discrete_energy, d3pm,
fsq, thrml_discrete, sampler, hybrid_energy, hybrid_edlm, continuous_encoder,
quantization, factor_weight_network, binary_autoencoder}`, `training/{train_*_edlm,
thrml_training, thrml_flax_coexistence, acp, total_correlation, hybrid_training}`,
and the scripts under `experimental/`.

## Out of scope (future work)

Making the EDLM language model train and generate on real text (a deep
energy↔sampler rewrite), a time-conditioned DTM chain over language, the binary
autoencoder / GAN stages, distributed training, and external benchmarking.
