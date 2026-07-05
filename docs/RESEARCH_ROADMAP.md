# Research roadmap: exactness-anchored scaling of discrete-diffusion LMs on thermodynamic hardware

Thermodynamic sampling units (TSUs) accelerate one primitive: block Gibbs
sampling on sparse probabilistic graphical models. The central open question
for ML on this hardware is quantitative — **how much model quality does
hardware-shaped sampling cost, and how does that cost scale** with model
size, vocabulary, connectivity, and sampling budget? Most energy-based and
discrete-diffusion work cannot answer this cleanly because neither training
nor generation has accessible ground truth.

This repo occupies an unusual position: the diffusion LM's reverse step is a
**linear-chain CRF**, so exact inference (log-partition, marginals,
forward-filter backward-sampling) is available as an oracle, while the same
distribution also runs as a 2-coloured block-Gibbs program on THRML — the
TSU-compatible path. Every hardware approximation is therefore measurable
exactly. The foundation is validated (`pytest`): samplers match enumerated
Boltzmann marginals, chain-CRF inference matches brute force, and the
THRML-native trainer's Monte-Carlo gradient matches the exact enumerated
two-term KL gradient.

## Aim 1 — Quality-versus-budget scaling curves

Bits-per-character and marginal fidelity (TV to the exact oracle) as
functions of Gibbs sweeps, sequence length, vocabulary size, and temperature.

- **Instrument:** [`scripts/exp_sweep_budget.py`](../scripts/exp_sweep_budget.py)
  — TV-vs-sweeps for random and *trained* reverse-step potentials, with the
  exact-FFBS finite-sample noise floor; CSV + plots.
- **Model:** [`models/chain_crf.py`](../thermolm_jax/models/chain_crf.py)
  (oracle), [`sampling/chain_mrf_thrml.py`](../thermolm_jax/sampling/chain_mrf_thrml.py)
  (hardware path), [`models/diffusion_lm.py`](../thermolm_jax/models/diffusion_lm.py)
  (`generate(..., thrml_warmup=k)` for budget-controlled generation).
- **Expected shape:** random unit-scale potentials mix in ~5 sweeps; trained
  (low-entropy) potentials at low temperature mix slower — the measured gap is
  the empirical "price of thermodynamic sampling" for language, and its growth
  with L and sharpness is the scaling law of interest.

## Aim 2 — Beyond chains: banded reverse-step graphs

Widen the reverse-step interaction graph from the chain (skip-1) to banded
skip-{1,2,4,...} families
([`models/connectivity.py`](../thermolm_jax/models/connectivity.py)). Exact
inference is O(L·V^(treewidth+1)) and dies quickly; block Gibbs does not.

- Small widths keep a junction-tree oracle feasible → extend the exactness
  anchor one or two rungs.
- Beyond that, the *only* viable inference is sampling — precisely the regime
  TSUs exist for. Locating the quality/width/budget crossover is the
  headline experiment.
- The chromatic machinery already generalizes:
  [`sampling/chromatic_gibbs.py`](../thermolm_jax/sampling/chromatic_gibbs.py)
  derives valid colourings for any banded graph, and THRML factors accept
  arbitrary pairwise structure.

## Aim 3 — Latent-spin energy corrections

Couple latent spin layers to the categorical token chain via THRML's mixed
spin/categorical `DiscreteEBMFactor`s and SuperBlocks, trained with the
library's two-phase KL-gradient estimator — an EDLM-style "energy correction"
kept inside the hardware's tractable factor family.

- The fully-visible trainer
  ([`training/thrml_ml.py`](../thermolm_jax/training/thrml_ml.py)) is the
  degenerate (no-latent) case and is already validated against exact
  gradients; adding latents reuses the same `IsingTrainingSpec` machinery
  with a clamped-positive phase.
- Success metric: bits/char improvement over the chain-CRF baseline at equal
  hardware budget (Aim 1 curves make this comparison meaningful).

## Deliverables

Open-source code extending THRML; a worked language-model example suitable
for THRML's documentation ([`examples/lm_on_thrml.ipynb`](../examples/lm_on_thrml.ipynb));
scaling datasets/curves (`results/`); a technical report.

## Relation to existing work

- **DTM (Extropic):** image-domain denoising thermodynamic models; no exact
  oracle in the reverse step. We add the exactness anchor and the language
  domain.
- **GPU-native discrete diffusion (D3PM, MDLM, SEDD):** factorized reverse
  steps sampled exactly on GPUs; no hardware sampling constraint, and no
  joint reverse-step structure.
- **Transformer-energy EBMs (EDLM):** expressive but inexpressible as TSU
  factors; our Aim 3 pursues the same correction idea inside the tractable
  family.

This work measures — rather than assumes — the cost of the thermodynamic
path for language modeling.
