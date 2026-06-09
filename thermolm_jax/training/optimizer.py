"""
Optimizer Module for ThermoLM JAX

Provides optimizer configurations and utilities using Optax.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax.numpy as jnp
import optax
from typing import Optional, Callable, Dict, Any


def create_optimizer(
    learning_rate: float = 0.0001,
    weight_decay: float = 0.01,
    schedule: Optional[Callable] = None,
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with optional learning rate schedule.
    
    Args:
        learning_rate: Learning rate
        weight_decay: Weight decay
        schedule: Optional learning rate schedule
    
    Returns:
        optimizer: Optax optimizer
    """
    if schedule is None:
        # Default cosine schedule
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=100000
        )
    
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=weight_decay
    )
    
    return optimizer


def create_sgd_optimizer(
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    schedule: Optional[Callable] = None,
) -> optax.GradientTransformation:
    """
    Create SGD optimizer with momentum.
    
    Args:
        learning_rate: Learning rate
        momentum: Momentum coefficient
        schedule: Optional learning rate schedule
    
    Returns:
        optimizer: Optax optimizer
    """
    if schedule is None:
        schedule = optax.constant_schedule(learning_rate)
    
    optimizer = optax.sgd(
        learning_rate=schedule,
        momentum=momentum
    )
    
    return optimizer


def create_adam_optimizer(
    learning_rate: float = 0.0001,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    schedule: Optional[Callable] = None,
) -> optax.GradientTransformation:
    """
    Create Adam optimizer.
    
    Args:
        learning_rate: Learning rate
        b1: Beta1 for momentum
        b2: Beta2 for RMSprop
        eps: Epsilon for numerical stability
        schedule: Optional learning rate schedule
    
    Returns:
        optimizer: Optax optimizer
    """
    if schedule is None:
        schedule = optax.constant_schedule(learning_rate)
    
    optimizer = optax.adam(
        learning_rate=schedule,
        b1=b1,
        b2=b2,
        eps=eps
    )
    
    return optimizer


# TODO: Add more optimizer options - Implemented below (cosine schedule, warmup)
# TODO: Implement adaptive learning rate schedules - Implemented below
# TODO: Add optimizer state inspection utilities - Implemented below


def cosine_schedule_with_warmup(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:
    """
    Create cosine learning rate schedule with warmup.
    
    Args:
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
    
    Returns:
        schedule: Learning rate schedule
    """
    # Build from Optax primitives: linear warmup + cosine decay
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    cosine = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps - warmup_steps,
    )
    return optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])


def inspect_optimizer_state(opt_state: optax.OptState) -> Dict[str, Any]:
    """
    Inspect optimizer state for debugging.
    
    Args:
        opt_state: Optimizer state
    
    Returns:
        state_info: Dictionary with optimizer state information
    """
    state_info = {}
    
    def traverse_state(state, prefix=""):
        if isinstance(state, dict):
            for key, value in state.items():
                traverse_state(value, f"{prefix}.{key}")
        elif hasattr(state, '__dict__'):
            for key, value in state.__dict__.items():
                traverse_state(value, f"{prefix}.{key}")
        elif isinstance(state, jnp.ndarray):
            state_info[prefix] = {
                'shape': state.shape,
                'dtype': state.dtype,
                'mean': float(state.mean()) if state.size > 0 else 0.0,
                'std': float(state.std()) if state.size > 0 else 0.0,
                'min': float(state.min()) if state.size > 0 else 0.0,
                'max': float(state.max()) if state.size > 0 else 0.0,
            }
    
    traverse_state(opt_state)
    
    return state_info
