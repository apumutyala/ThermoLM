"""
Diffusion Schedule Module for ThermoLM JAX

Implements noise schedules for energy-based diffusion models.
Supports cosine, linear, and sigmoid schedules.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
from typing import Literal, Optional, Tuple


def cosine_schedule(T: int = 1000, s: float = 0.008) -> jnp.ndarray:
    """
    Cosine noise schedule (improved over linear).
    
    This schedule provides better diffusion quality by smoothly
    transitioning from high to low noise levels.
    
    Args:
        T: Number of timesteps
        s: Offset parameter for smoothness
    
    Returns:
        betas: Noise schedule (T,)
    """
    steps = T + 1
    x = jnp.linspace(0, T, steps)
    alphas_cumprod = jnp.cos(((x / T) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)


def linear_schedule(
    T: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.9999
) -> jnp.ndarray:
    """
    Linear noise schedule.
    
    Args:
        T: Number of timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        betas: Noise schedule (T,)
    """
    return jnp.linspace(beta_start, beta_end, T)


def sigmoid_schedule(
    T: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.9999,
    sigmoid_scale: float = 6.0
) -> jnp.ndarray:
    """
    Sigmoid noise schedule.
    
    Args:
        T: Number of timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        sigmoid_scale: Scale parameter for sigmoid
    
    Returns:
        betas: Noise schedule (T,)
    """
    betas = jnp.linspace(-sigmoid_scale, sigmoid_scale, T)
    betas = jax.nn.sigmoid(betas)
    betas = betas * (beta_end - beta_start) + beta_start
    return betas


def compute_alpha_bar(betas: jnp.ndarray, t: int) -> float:
    """
    Compute cumulative product of alphas at timestep t.
    
    Args:
        betas: Noise schedule (T,)
        t: Timestep
    
    Returns:
        alpha_bar: Cumulative product of (1 - beta) up to t
    """
    alpha = 1 - betas
    alpha_bar = jnp.cumprod(alpha)
    return alpha_bar[t]


def q_sample(
    x_0: jnp.ndarray,
    t: int,
    betas: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from forward diffusion process.
    
    Args:
        x_0: Clean data
        t: Timestep
        betas: Noise schedule
        key: PRNG key
    
    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added
    """
    alpha_bar = compute_alpha_bar(betas, t)
    noise = jax.random.normal(key, shape=x_0.shape)
    x_t = jnp.sqrt(alpha_bar) * x_0 + jnp.sqrt(1 - alpha_bar) * noise
    return x_t, noise


def get_schedule(
    schedule_type: Literal['cosine', 'linear', 'sigmoid'] = 'cosine',
    T: int = 1000,
    **kwargs
) -> jnp.ndarray:
    """
    Get noise schedule by type.
    
    Args:
        schedule_type: Type of schedule ('cosine', 'linear', 'sigmoid')
        T: Number of timesteps
        **kwargs: Additional parameters for schedule
    
    Returns:
        betas: Noise schedule (T,)
    """
    if schedule_type == 'cosine':
        return cosine_schedule(T, **kwargs)
    elif schedule_type == 'linear':
        return linear_schedule(T, **kwargs)
    elif schedule_type == 'sigmoid':
        return sigmoid_schedule(T, **kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


class DiffusionSchedule:
    """
    Wrapper class for diffusion schedules.
    
    Provides convenient interface for managing diffusion schedules
    and related computations.
    """
    
    def __init__(
        self,
        schedule_type: Literal['cosine', 'linear', 'sigmoid'] = 'cosine',
        T: int = 1000,
        **kwargs
    ):
        """
        Initialize diffusion schedule.
        
        Args:
            schedule_type: Type of schedule
            T: Number of timesteps
            **kwargs: Additional parameters
        """
        self.schedule_type = schedule_type
        self.T = T
        self.betas = get_schedule(schedule_type, T, **kwargs)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
    
    def get_alpha_bar(self, t: int) -> float:
        """Get cumulative product at timestep t."""
        return self.alphas_cumprod[t]
    
    def sample_forward(
        self,
        x_0: jnp.ndarray,
        t: int,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample from forward process."""
        return q_sample(x_0, t, self.betas, key)


def p_sample(
    model: callable,
    x_t: jnp.ndarray,
    t: int,
    betas: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from reverse diffusion process (single timestep).
    
    Args:
        model: Model that predicts noise
        x_t: Noisy data at timestep t
        t: Timestep
        betas: Noise schedule
        key: PRNG key
    
    Returns:
        x_{t-1}: Sample at timestep t-1
        pred_noise: Predicted noise
    """
    # Predict noise
    pred_noise = model(x_t, t)
    
    # Compute reverse process parameters
    alpha = 1 - betas
    alpha_bar = jnp.cumprod(alpha)
    
    # Clamp indices
    t_clamped = jnp.clip(t, 0, len(betas) - 1)
    
    # Get alpha values
    alpha_t = alpha[t_clamped]
    alpha_bar_t = alpha_bar[t_clamped]
    alpha_bar_t_minus_1 = alpha_bar[max(0, t_clamped - 1)]
    
    # Compute posterior variance
    beta_t = betas[t_clamped]
    posterior_variance = beta_t * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)
    
    # Compute mean
    mean = (x_t - beta_t * pred_noise / jnp.sqrt(1 - alpha_bar_t)) / jnp.sqrt(alpha_t)
    
    # Add noise if t > 0
    if t > 0:
        noise = jax.random.normal(key, shape=x_t.shape)
        x_prev = mean + jnp.sqrt(posterior_variance) * noise
    else:
        x_prev = mean
    
    return x_prev, pred_noise


def p_sample_loop(
    model: callable,
    shape: Tuple[int, ...],
    betas: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Full reverse diffusion loop for generation.
    
    Args:
        model: Model that predicts noise
        shape: Shape of output
        betas: Noise schedule
        key: PRNG key
    
    Returns:
        x_0: Generated samples
    """
    # Start from noise
    key, init_key = jax.random.split(key)
    x = jax.random.normal(init_key, shape)
    
    # Iterate backwards
    for t in range(len(betas) - 1, -1, -1):
        key, sample_key = jax.random.split(key)
        x, _ = p_sample(model, x, t, betas, sample_key)
    
    return x


# TODO: Add schedule visualization utilities (for debugging)
# TODO: Implement adaptive schedule based on training progress
