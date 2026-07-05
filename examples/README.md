# Examples

## [`lm_on_thrml.ipynb`](lm_on_thrml.ipynb) — Language modeling on THRML

A self-contained, CPU-runnable (~3 min) notebook that:

1. expresses a linear-chain CRF as a THRML factor graph (categorical factors,
   even/odd 2-colouring) and samples it with block Gibbs;
2. measures sample fidelity against **exact** forward–backward marginals as a
   function of the Gibbs sweep budget (the "exactness anchor");
3. trains a small character-level discrete-diffusion LM whose reverse step is
   that chain CRF, by exact conditional maximum likelihood;
4. generates text with the exact FFBS sampler and the THRML block-Gibbs
   sampler side by side — same distribution, two substrates.

The notebook is written to the conventions of THRML's documentation notebooks
(00–03 at [docs.thrml.ai](https://docs.thrml.ai)) with the intent that it
could be contributed upstream as a language-modeling example. It depends on
`thermolm_jax` only for the exact-inference oracle, the tokenizer, and the
training loop; all THRML usage is spelled out inline.

Run it after the repo Quickstart (README) with the `viz` extra installed:

```bash
pip install -e ".[dev,viz]"
jupyter lab examples/lm_on_thrml.ipynb
```
