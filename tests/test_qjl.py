"""Tests for QJL 1-bit quantizer."""

import numpy as np
import pytest

from turboquant.exceptions import DimensionMismatchError
from turboquant.qjl import QJL


class TestQJLConstruction:
    def test_default_projection_dim(self) -> None:
        qjl = QJL(dim=256)
        assert qjl.projection_dim == 256

    def test_custom_projection_dim(self) -> None:
        qjl = QJL(dim=256, projection_dim=128)
        assert qjl.projection_dim == 128

    def test_deterministic_with_seed(self) -> None:
        q1 = QJL(dim=64, seed=42)
        q2 = QJL(dim=64, seed=42)
        np.testing.assert_array_equal(q1._projection, q2._projection)


class TestQJLQuantize:
    def test_output_type(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        assert compressed.num_vectors == 10
        assert compressed.dim == 64
        assert compressed.bit_width == 1

    def test_dimension_mismatch_raises(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 128))
        with pytest.raises(DimensionMismatchError):
            qjl.quantize(vectors)

    def test_signs_are_binary(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        signs = compressed.extra_arrays["signs"]
        assert set(np.unique(signs)).issubset({-1, 1})

    def test_norms_are_positive(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        assert np.all(compressed.norms > 0)


class TestQJLInnerProduct:
    def test_unbiased_estimator(self) -> None:
        """Over many trials, QJL inner product estimate should be unbiased."""
        dim = 256
        n_trials = 200
        rng = np.random.default_rng(42)

        errors = []
        for trial in range(n_trials):
            qjl = QJL(dim=dim, seed=trial)
            x = rng.standard_normal(dim)
            x = x / np.linalg.norm(x)
            y = rng.standard_normal(dim)

            compressed = qjl.quantize(x.reshape(1, -1))
            estimated = qjl.inner_product(y, compressed)[0]
            true_ip = np.dot(x, y)
            errors.append(estimated - true_ip)

        mean_error = np.mean(errors)
        assert abs(mean_error) < 0.15, f"Mean error {mean_error} too large for unbiased estimator"

    def test_dimension_mismatch_raises(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        bad_query = rng.standard_normal(128)
        with pytest.raises(DimensionMismatchError):
            qjl.inner_product(bad_query, compressed)

    def test_output_shape(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((20, 64))
        query = rng.standard_normal(64)
        compressed = qjl.quantize(vectors)
        scores = qjl.inner_product(query, compressed)
        assert scores.shape == (20,)

    def test_higher_projection_dim_reduces_variance(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        x = rng.standard_normal(dim)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(dim)
        true_ip = np.dot(x, y)

        variances = []
        for proj_dim in [32, 128]:
            errors = []
            for trial in range(100):
                qjl = QJL(dim=dim, projection_dim=proj_dim, seed=trial)
                compressed = qjl.quantize(x.reshape(1, -1))
                estimated = qjl.inner_product(y, compressed)[0]
                errors.append((estimated - true_ip) ** 2)
            variances.append(np.mean(errors))
        assert variances[0] > variances[1], "Higher projection_dim should reduce variance"
