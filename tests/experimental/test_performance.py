"""
Benchmark tests for ThermoLM JAX.

Performance profiling and memory benchmarking.
"""

import pytest
import jax
import jax.numpy as jnp
import time


@pytest.mark.benchmark
@pytest.mark.slow
def test_energy_function_performance():
    """Benchmark energy function performance."""
    # TODO: Implement when energy function is fully implemented
    pytest.skip("Energy function not yet fully implemented")


@pytest.mark.benchmark
@pytest.mark.slow
def test_sampler_performance():
    """Benchmark sampler performance."""
    # TODO: Implement when sampler is fully implemented
    pytest.skip("Sampler not yet fully implemented")


# TODO: Add more benchmark tests
# TODO: Memory profiling
# TODO: Computation time profiling
