"""Shared test fixtures for TurboQuant tests."""

import numpy as np
import pytest

DIMS = [384, 512, 768, 1024, 1536, 2048, 3072]
BIT_WIDTHS = [1, 2, 3, 4]


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for tests."""
    return np.random.default_rng(42)


@pytest.fixture(params=[64, 384, 1536])
def dim(request: pytest.FixtureRequest) -> int:
    """Common embedding dimensions for parameterized tests."""
    return request.param


@pytest.fixture
def random_vectors(rng: np.random.Generator) -> np.ndarray:
    """100 random unit vectors in 256 dimensions."""
    vectors = rng.standard_normal((100, 256))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def random_query(rng: np.random.Generator) -> np.ndarray:
    """Single random unit vector in 256 dimensions."""
    q = rng.standard_normal(256)
    return q / np.linalg.norm(q)
