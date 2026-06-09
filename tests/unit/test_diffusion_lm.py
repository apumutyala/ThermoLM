"""
CPU sanity test for the Tier-1 chain-CRF diffusion LM.

Trains a tiny model on a small embedded corpus and asserts that the denoising
bits/char drops well below the unigram baseline and that generation runs and
produces in-vocabulary text. Fast enough for the default suite.
"""

import numpy as np
import jax
import pytest

from thermolm_jax.data.char_tokenizer import CharTokenizer, make_windows
from thermolm_jax.models.diffusion_lm import (
    DiffusionLMConfig,
    fit,
    generate,
    unigram_bits_per_char,
)

pytestmark = pytest.mark.unit


def test_charlm_learns_and_generates():
    text = (
        "to be or not to be that is the question "
        "whether tis nobler in the mind to suffer "
    ) * 40
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    L = 24
    windows = make_windows(ids, seq_len=L, stride=4)

    cfg = DiffusionLMConfig(vocab_size=tok.vocab_size, seq_len=L, hidden_size=96, n_layers=2)
    net, params, history = fit(
        windows, cfg, jax.random.PRNGKey(0), n_iters=200, batch_size=64, lr=3e-3
    )

    baseline = unigram_bits_per_char(ids, tok.vocab_size)
    final_bpc = sum(history[-20:]) / 20
    # the structured denoiser should beat the order-0 baseline by a wide margin
    assert final_bpc < 0.6 * baseline, (final_bpc, baseline)
    assert history[-1] < history[0]

    samples = np.asarray(
        generate(params, net, cfg, jax.random.PRNGKey(1), n_samples=4, n_steps=12, use_thrml=False)
    )
    assert samples.shape == (4, L)
    assert samples.min() >= 0 and samples.max() < tok.vocab_size  # in-vocabulary
