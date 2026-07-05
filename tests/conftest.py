"""
Pytest configuration for ThermoLM JAX tests.

Kept deliberately minimal: the correctness tests build their own models and
data. (An earlier conftest carried EDLM-era fixtures — GPT-2 vocab batches,
transformer configs — and a session fixture that disabled JIT globally.)
"""

import pytest
import jax


@pytest.fixture
def rng_key():
    """A JAX PRNG key for tests that want one."""
    return jax.random.PRNGKey(42)


def pytest_configure(config):
    """Register the markers used by the suite."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
