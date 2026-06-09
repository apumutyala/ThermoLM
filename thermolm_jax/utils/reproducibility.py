"""
Reproducibility utilities for ThermoLM JAX.

Provides random seed management for exact reproducibility.
Implements DD-ARCH-014.

Author: Apuroop Mutyala
Date: April 14, 2026
"""

import random
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional


def set_seed(seed: int):
    """
    Set random seed for all libraries.
    
    Design Decision: Strict random seed control
    - Rationale: Exact reproducibility
    - Impact: Same seed = same results
    - Trade-off: Less randomness in exploration
    - Downstream: Debugging easier
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    # JAX handles seeds differently via PRNGKey
    # Set JAX to use deterministic operations where possible
    jax.config.update("jax_enable_x64", True)


def get_rng_key(seed: int) -> jax.random.PRNGKey:
    """
    Get JAX PRNG key from seed.
    
    Args:
        seed: Random seed
    
    Returns:
        key: JAX PRNG key
    """
    return jax.random.PRNGKey(seed)


def split_rng_key(key: jax.random.PRNGKey, num: int = 2) -> list:
    """
    Split JAX PRNG key.
    
    Args:
        key: JAX PRNG key
        num: Number of keys to split into
    
    Returns:
        keys: List of PRNG keys
    """
    return jax.random.split(key, num)


def fork_rng_key(key: jax.random.PRNGKey) -> tuple:
    """
    Fork JAX PRNG key (split into 2).
    
    Args:
        key: JAX PRNG key
    
    Returns:
        (key1, key2): Two PRNG keys
    """
    return jax.random.split(key, 2)
