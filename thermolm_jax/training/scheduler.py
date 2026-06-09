"""
Learning rate schedulers for ThermoLM JAX.

Provides various learning rate schedules using Optax.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import optax
import jax.numpy as jnp
from typing import Optional


def get_cosine_schedule(
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    alpha: float = 0.0,
) -> optax.Schedule:
    """
    Get cosine learning rate schedule.
    
    Args:
        learning_rate: Peak learning rate
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        alpha: Minimum learning rate fraction
    
    Returns:
        schedule: Optax schedule
    """
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=learning_rate * alpha,
        )
    else:
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps,
            alpha=alpha,
        )
    return schedule


def get_linear_schedule(
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
) -> optax.Schedule:
    """
    Get linear learning rate schedule.
    
    Args:
        learning_rate: Peak learning rate
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
    
    Returns:
        schedule: Optax schedule
    """
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=0.0,
        )
    else:
        schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=total_steps,
        )
    return schedule


def get_constant_schedule(
    learning_rate: float,
) -> optax.Schedule:
    """
    Get constant learning rate schedule.
    
    Args:
        learning_rate: Learning rate
    
    Returns:
        schedule: Optax schedule
    """
    return optax.constant_schedule(learning_rate)


def get_schedule(
    schedule_type: str,
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
    **kwargs
) -> optax.Schedule:
    """
    Get learning rate schedule by type.
    
    Args:
        schedule_type: Type of schedule ('cosine', 'linear', 'constant')
        learning_rate: Peak learning rate
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        **kwargs: Additional schedule-specific parameters
    
    Returns:
        schedule: Optax schedule
    """
    if schedule_type == 'cosine':
        return get_cosine_schedule(learning_rate, total_steps, warmup_steps, **kwargs)
    elif schedule_type == 'linear':
        return get_linear_schedule(learning_rate, total_steps, warmup_steps)
    elif schedule_type == 'constant':
        return get_constant_schedule(learning_rate)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
