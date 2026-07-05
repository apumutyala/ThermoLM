"""
Chain-CRF discrete-diffusion language model (Tier 1).

Architecture (see plans/roadmap):
- Absorbing/masked forward corruption x_0 -> x_t (a fraction of positions become
  a MASK token, `q_xt` from d3pm).
- A denoising network (FactorWeightNetwork) conditioned on (x_t, t) emits the
  reverse-step energy as a linear-chain CRF over the CLEAN vocab:
  unary (B,L,V) + nearest-neighbour pairwise (B,L-1,V,V).
- Training (M1): exact conditional ML of the chain CRF — predict x_0 from x_t,
  scored by `chain_log_likelihood` (exact log Z via the forward algorithm). The
  pairwise term lets the reverse step model joint token structure that a
  factorised denoiser cannot (the EDLM-style "energy correction").
- Generation: iterative unmasking; each step samples the chain CRF jointly,
  either exactly (FFBS) or on THRML (the TSU path).

Input vocab to the net is V+1 (clean vocab + MASK); output categories are V.
"""

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .factor_weight_network import FactorWeightNetwork
from .chain_crf import chain_log_likelihood, chain_sample

_LN2 = jnp.log(2.0)


def q_xt(x: jnp.ndarray, move_chance: jnp.ndarray, mask_index: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Masked forward corruption: each position becomes mask_index with probability move_chance."""
    move_indices = jax.random.uniform(key, x.shape) < move_chance
    xt = jnp.where(move_indices, mask_index, x)
    return xt


@dataclass
class DiffusionLMConfig:
    vocab_size: int           # clean vocabulary size V (mask id = V)
    seq_len: int = 128
    hidden_size: int = 256
    n_layers: int = 2
    t_min: float = 1e-3       # avoid t=0 (no corruption) degeneracy


def build_net(cfg: DiffusionLMConfig) -> FactorWeightNetwork:
    """Denoising net: input vocab V+1 (incl. MASK), outputs V-way factors."""
    return FactorWeightNetwork(
        vocab_size=cfg.vocab_size + 1,
        hidden_size=cfg.hidden_size,
        n_levels=cfg.vocab_size,
        n_layers=cfg.n_layers,
    )


def init_params(net: FactorWeightNetwork, cfg: DiffusionLMConfig, key) -> dict:
    x = jnp.zeros((1, cfg.seq_len), dtype=jnp.int32)
    t = jnp.zeros((1,), dtype=jnp.float32)
    return net.init(key, x, t)


def denoising_loss(params, net, x0, key, cfg: DiffusionLMConfig):
    """Mean exact-CRF denoising NLL over a batch, plus bits/char.

    Args:
        x0: (B, L) clean token ids in [0, V).
    Returns:
        (loss, bits_per_char)
    """
    B, L = x0.shape
    k_t, k_q = jax.random.split(key)
    t = jax.random.uniform(k_t, (B,), minval=cfg.t_min, maxval=1.0)
    x_t = q_xt(x0, t[:, None], mask_index=cfg.vocab_size, key=k_q)  # (B, L), MASK = V

    unary, pairwise = net.apply(params, x_t, t)  # (B,L,V), (B,L-1,V,V)
    lls = jax.vmap(chain_log_likelihood)(x0, unary, pairwise)       # (B,)
    nll_per_token = -lls.mean() / L
    return nll_per_token, nll_per_token / _LN2


def unigram_bits_per_char(ids, vocab_size: int) -> float:
    """Order-0 (unigram) entropy in bits/char — the baseline to beat."""
    counts = np.bincount(np.asarray(ids), minlength=vocab_size) + 1
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def fit(windows, cfg: DiffusionLMConfig, key, n_iters=300, batch_size=64, lr=3e-3):
    """Train the chain-CRF diffusion LM by exact denoising conditional ML.

    Args:
        windows: (N, L) int array of clean token windows.
        cfg, key, n_iters, batch_size, lr: training controls.
    Returns:
        (net, params, history) where history is a list of bits/char per iter.
    """
    net = build_net(cfg)
    params = init_params(net, cfg, key)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch, k):
        (loss, bpc), g = jax.value_and_grad(denoising_loss, has_aux=True)(
            params, net, batch, k, cfg
        )
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, upd), opt_state, bpc

    n = windows.shape[0]
    history = []
    for _ in range(n_iters):
        key, kb, ks = jax.random.split(key, 3)
        batch = windows[jax.random.randint(kb, (batch_size,), 0, n)]
        params, opt_state, bpc = step(params, opt_state, batch, ks)
        history.append(float(bpc))
    return net, params, history


@jax.jit
def _ffbs_batch(unary, pairwise, key):
    keys = jax.random.split(key, unary.shape[0])
    return jax.vmap(chain_sample)(keys, unary, pairwise)


def generate(
    params,
    net,
    cfg: DiffusionLMConfig,
    key,
    n_samples: int = 16,
    n_steps: int = 16,
    temperature: float = 1.0,
    use_thrml: bool = False,
):
    """Generate sequences by iterative unmasking with joint chain-CRF sampling.

    Starts from all-MASK and commits a growing fraction of positions each step,
    sampling the full chain CRF jointly so co-committed positions are coherent.

    Returns (n_samples, L) int32 token ids in [0, V).
    """
    L, V, mask = cfg.seq_len, cfg.vocab_size, cfg.vocab_size
    x = jnp.full((n_samples, L), mask, dtype=jnp.int32)

    x0_hat = None
    for step in range(n_steps):
        key, k_net, k_samp, k_commit = jax.random.split(key, 4)
        t_val = max(1.0 - step / n_steps, cfg.t_min)
        t = jnp.full((n_samples,), t_val, dtype=jnp.float32)
        unary, pairwise = net.apply(params, x, t)
        unary = unary / temperature
        pairwise = pairwise / temperature

        if use_thrml:
            from ..sampling.chain_mrf_thrml import sample_chain_thrml
            x0_hat = sample_chain_thrml(unary, pairwise, k_samp, temperature=1.0)
        else:
            x0_hat = _ffbs_batch(unary, pairwise, k_samp)

        is_mask = x == mask
        p_commit = 1.0 / (n_steps - step)  # expected all committed by the end
        commit = jax.random.uniform(k_commit, x.shape) < p_commit
        x = jnp.where(is_mask & commit, x0_hat, x)

    # commit anything still masked
    x = jnp.where(x == mask, x0_hat, x)
    return x
