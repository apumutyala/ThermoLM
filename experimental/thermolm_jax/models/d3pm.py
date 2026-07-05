"""
Masked / absorbing-state discrete diffusion for ThermoLM JAX

STATUS: EXPERIMENTAL / NOT VALIDATED. See STATUS.md. This is actually the
absorbing-state (masked) diffusion loss in the style of MDLM (Sahoo et al.,
2024) / SEDD (Lou et al., 2023) — NOT D3PM (Austin et al., 2021), which uses
general transition matrices. The reverse sampler also overwrites already-decoded
tokens, violating the absorbing structure. Name kept for continuity.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple


def q_xt(
    x: jnp.ndarray,
    move_chance: jnp.ndarray,
    mask_index: int,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Compute noisy sample xt using d3pm forward process.

    Args:
        x: Input tokens, shape (batch, seq_len)
        move_chance: Probability of moving to mask, shape (batch, 1)
        mask_index: Index of mask token
        key: PRNG key for stochastic masking

    Returns:
        Noisy tokens xt, shape (batch, seq_len)
    """
    # Sample which tokens to mask
    move_indices = jax.random.uniform(key, x.shape) < move_chance
    xt = jnp.where(move_indices, mask_index, x)
    return xt


def d3pm_parameterization(
    logits: jnp.ndarray,
    mask_index: int
) -> jnp.ndarray:
    """
    Apply d3pm parameterization to logits.
    
    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size)
        mask_index: Index of mask token
    
    Returns:
        Parameterized logits, shape (batch, seq_len, vocab_size)
    """
    # Set mask token log prob to -infinity
    logits = logits.at[:, :, mask_index].set(-1e9)
    
    # Normalize to log probabilities
    logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    
    return logits


def d3pm_loss(
    model_output: jnp.ndarray,
    xt: jnp.ndarray,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    mask_index: int,
    T: int
) -> jnp.ndarray:
    """
    Compute d3pm loss.
    
    Args:
        model_output: Model output logits, shape (batch, seq_len, vocab_size)
        xt: Noisy tokens at timestep t, shape (batch, seq_len)
        x0: Clean tokens at timestep 0, shape (batch, seq_len)
        t: Timestep, shape (batch,)
        mask_index: Index of mask token
        T: Total number of timesteps
    
    Returns:
        Loss, shape (batch, seq_len)
    """
    dt = 1.0 / T
    
    # Clamp t to avoid division by zero
    t_clamped = jnp.clip(t, 1e-4, 1.0 - 1e-4)
    
    # Compute alpha_t and alpha_s
    alpha_t = 1.0 - t_clamped
    alpha_s = 1.0 - (t_clamped - dt)
    
    # Get log probabilities for x0 and mask
    log_x_theta_at_x0 = jnp.take_along_axis(model_output, x0[:, :, None], axis=-1).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, mask_index]
    x_theta_at_m = jnp.exp(log_x_theta_at_m)
    
    # Reshape time-dependent parameters for broadcasting with seq_len dimension
    # Handle both (batch_size,) and (batch_size, 1) input shapes
    if t_clamped.ndim == 1:
        alpha_t = alpha_t[:, None]  # (batch_size,) -> (batch_size, 1)
        alpha_s = alpha_s[:, None]
        t_clamped = t_clamped[:, None]
    # If already (batch_size, 1), no reshaping needed
    
    # Compute term 1
    term_1_coef = dt / t_clamped
    term_1_log_nr = jnp.log(alpha_t * x_theta_at_m / t_clamped + 1.0)
    term_1_log_dr = log_x_theta_at_x0
    
    # Compute term 2
    term_2_coef = 1.0 - dt / t_clamped
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = jnp.log(alpha_s * x_theta_at_m / (t_clamped - dt) + 1.0)
    
    # Compute loss
    L_vb_masked = (
        term_1_coef * (term_1_log_nr - term_1_log_dr)
        + term_2_coef * (term_2_log_nr - term_2_log_dr)
    )
    
    # Mask loss for non-masked tokens
    L_vb = L_vb_masked * (xt == mask_index)
    
    return T * L_vb


def sample_prior(
    batch_size: int,
    seq_len: int,
    mask_index: int
) -> jnp.ndarray:
    """
    Sample from prior (all masks).
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        mask_index: Index of mask token
    
    Returns:
        Prior sample, shape (batch, seq_len)
    """
    return jnp.ones((batch_size, seq_len), dtype=jnp.int32) * mask_index


def d3pm_sample(
    model_output: jnp.ndarray,
    xt: jnp.ndarray,
    t: jnp.ndarray,
    mask_index: int,
    T: int,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Sample from D3PM reverse process (single timestep).
    
    Args:
        model_output: Model output logits, shape (batch, seq_len, vocab_size)
        xt: Noisy tokens at timestep t, shape (batch, seq_len)
        t: Timestep, shape (batch,)
        mask_index: Index of mask token
        T: Total number of timesteps
        key: PRNG key
    
    Returns:
        x_{t-1}: Sample at timestep t-1, shape (batch, seq_len)
    """
    dt = 1.0 / T
    
    # Clamp t to avoid division by zero
    t_clamped = jnp.clip(t, 1e-4, 1.0)
    
    # Compute transition probabilities
    # For D3PM, we use a simple transition: with probability (1-t) stay same, with probability t move to model prediction
    alpha_t = 1.0 - t_clamped
    alpha_s = 1.0 - jnp.clip(t_clamped - dt, 1e-4, 1.0)
    
    # Get model predictions
    logits = d3pm_parameterization(model_output, mask_index)
    probs = jnp.exp(logits)
    
    # Sample from transition
    # With probability alpha_t/alpha_s, stay at xt
    # With probability 1 - alpha_t/alpha_s, sample from model
    stay_prob = alpha_t / (alpha_s + 1e-8)
    stay_prob = jnp.clip(stay_prob, 0.0, 1.0)
    
    # Sample whether to stay or move
    key, stay_key, sample_key = jax.random.split(key, 3)
    stay_mask = jax.random.uniform(stay_key, xt.shape) < stay_prob[:, None]
    
    # Sample from model distribution
    sample_from_model = jax.random.categorical(sample_key, logits, axis=-1)
    
    # Combine: stay where stay_mask is True, otherwise use model sample
    x_prev = jnp.where(stay_mask, xt, sample_from_model)
    
    return x_prev


def d3pm_reverse_process(
    model: callable,
    shape: Tuple[int, int],
    mask_index: int,
    T: int,
    key: jax.random.PRNGKey,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Full reverse diffusion process for generation.
    
    Args:
        model: Model that takes (x_t, t) and outputs logits
        shape: (batch_size, seq_len)
        mask_index: Index of mask token
        T: Total number of timesteps
        key: PRNG key
        mask: Optional attention mask
    
    Returns:
        x_0: Generated samples, shape (batch, seq_len)
    """
    batch_size, seq_len = shape
    
    # Start from prior (all masks)
    x = sample_prior(batch_size, seq_len, mask_index)
    
    # Use jax.lax.scan for JIT compatibility (Phase 2.6)
    def reverse_step(carry, t_step):
        x, key = carry
        key, model_key, sample_key = jax.random.split(key, 3)
        
        # Create timestep array
        t = jnp.ones((batch_size,), dtype=jnp.float32) * (t_step / T)
        
        # Get model predictions
        model_output = model(x, t)
        
        # Sample reverse step
        x_new = d3pm_sample(model_output, x, t, mask_index, T, sample_key)
        
        return (x_new, key), None  # (carry, output)
    
    # Create timesteps going backwards from T to 1
    timesteps = jnp.arange(T, 0, -1, dtype=jnp.float32)
    
    # Run scan
    (x_final, _), _ = jax.lax.scan(reverse_step, (x, key), timesteps)
    
    return x_final


# TODO: Test d3pm loss implementation with actual training
