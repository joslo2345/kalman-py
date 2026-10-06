"""Shared test fixtures and JAX setup."""

import pytest

from tests.scenarios import LinearScenario, make_cv_2d

try:
    import jax
except ImportError:
    pass
else:
    # JAX defaults to float32; equivalence tests compare against float64 libraries.
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


@pytest.fixture(scope="session")
def cv_scenario() -> LinearScenario:
    return make_cv_2d(seed=0, steps=200)


@pytest.fixture(scope="session")
def long_scenario() -> LinearScenario:
    return make_cv_2d(seed=1, steps=100_000)
