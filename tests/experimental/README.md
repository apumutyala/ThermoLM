# Legacy / experimental tests (not run by default)

These are the original, mostly **shape-only** tests for the unvalidated
energy-diffusion language-model components (FSQ, D3PM, discrete/hybrid EDLM,
AdaLN, rotary, EMA, etc.). They check tensor shapes and run-to-completion, not
correctness, and several rely on APIs that have since been corrected or
removed. They are excluded from the default `pytest` run (see `pytest.ini`'s
`--ignore=tests/experimental`).

The validated suite is `tests/unit/test_ising_correctness.py`. See the
repository `STATUS.md` for what is and is not validated.
