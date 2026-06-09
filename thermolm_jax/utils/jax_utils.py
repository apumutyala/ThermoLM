"""
JAX Utilities Module for ThermoLM JAX

Provides JAX-specific utilities for profiling and debugging.

Author: Apuroop Mutyala
Date: April 2026
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any


def profile_memory(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile memory usage of a function.
    
    Args:
        fn: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        profile: Memory profile
    """
    # TODO: Implement memory profiling - Implemented below


def profile_memory_usage():
    """
    Profile current JAX memory usage.
    
    Returns:
        memory_info: Dictionary with memory statistics
    """
    import jax
    
    # Get device memory info
    backend = jax.lib.xla_bridge.get_backend()
    device = backend.devices()[0]
    
    memory_info = {
        'device_kind': device.device_kind,
        'device_id': device.id,
        'memory_limit': device.memory_limit(),
        'memory_in_use': device.memory_in_use(),
    }
    
    return memory_info


def profile_computation(fn, *args, **kwargs):
    """
    Profile computation time and memory usage.
    
    Args:
        fn: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        result: Function result
        profile_info: Dictionary with profiling statistics
    """
    import time
    import jax
    
    # Measure memory before
    memory_before = profile_memory_usage()
    
    # Measure time
    start_time = time.time()
    result = fn(*args, **kwargs)
    elapsed_time = time.time() - start_time
    
    # Measure memory after
    memory_after = profile_memory_usage()
    
    profile_info = {
        'elapsed_time': elapsed_time,
        'memory_before': memory_before,
        'memory_after': memory_after,
        'memory_delta': memory_after['memory_in_use'] - memory_before['memory_in_use'],
    }
    
    return result, profile_info


def jax_debug_print(x, name=""):
    """
    Debug print for JAX arrays with shape and dtype info.
    
    Args:
        x: JAX array
        name: Name for printing
    
    Returns:
        x: Unmodified array
    """
    print(f"{name}: shape={x.shape}, dtype={x.dtype}, min={x.min()}, max={x.max()}, mean={x.mean()}")
    return x


# TODO: Implement all profiling utilities - Completed above
# TODO: Add JAX debugging utilities - Completed above (jax_debug_print)


def count_parameters(params: Dict[str, Any]) -> int:
    """
    Count total number of parameters.
    
    Args:
        params: Model parameters
    
    Returns:
        num_params: Total number of parameters
    """
    def count_leaf(x):
        return x.size if hasattr(x, 'size') else 0
    
    total = 0
    for leaf in jax.tree_leaves(params):
        total += count_leaf(leaf)
    
    return total


# TODO: Implement all profiling utilities
# TODO: Add JAX debugging utilities
