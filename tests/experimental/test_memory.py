"""
Memory benchmark tests for ThermoLM JAX.

Memory usage profiling.
"""

import pytest
import jax
import jax.numpy as jnp


@pytest.mark.benchmark
@pytest.mark.slow
def test_model_memory_usage():
    """Benchmark model memory usage."""
    # TODO: Implement when model is fully implemented
    pytest.skip("Model not yet fully implemented")


@pytest.mark.benchmark
@pytest.mark.slow
def test_data_memory_usage():
    """Benchmark data loading memory usage."""
    # TODO: Implement when data pipeline is fully tested
    pytest.skip("Data pipeline not yet fully tested")


# TODO: Add more memory tests
# TODO: Test memory scaling with batch size
# TODO: Test memory scaling with sequence length
