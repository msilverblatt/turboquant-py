"""Tests for NumPy/PyTorch acceleration dispatch."""

import numpy as np

from turboquant._accel import (
    batch_inner_product,
    generate_orthogonal_matrix,
    generate_projection_matrix,
    has_torch,
    matmul,
)


class TestGenerateOrthogonalMatrix:
    def test_returns_square_matrix(self) -> None:
        mat = generate_orthogonal_matrix(64, seed=42)
        assert mat.shape == (64, 64)

    def test_is_orthogonal(self) -> None:
        mat = generate_orthogonal_matrix(64, seed=42)
        product = mat @ mat.T
        np.testing.assert_allclose(product, np.eye(64), atol=1e-10)

    def test_deterministic_with_seed(self) -> None:
        a = generate_orthogonal_matrix(64, seed=42)
        b = generate_orthogonal_matrix(64, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = generate_orthogonal_matrix(64, seed=42)
        b = generate_orthogonal_matrix(64, seed=99)
        assert not np.allclose(a, b)


class TestGenerateProjectionMatrix:
    def test_shape(self) -> None:
        mat = generate_projection_matrix(32, 64, seed=42)
        assert mat.shape == (32, 64)

    def test_rows_are_orthonormal(self) -> None:
        mat = generate_projection_matrix(32, 64, seed=42)
        product = mat @ mat.T
        np.testing.assert_allclose(product, np.eye(32), atol=1e-10)

    def test_deterministic_with_seed(self) -> None:
        a = generate_projection_matrix(32, 64, seed=42)
        b = generate_projection_matrix(32, 64, seed=42)
        np.testing.assert_array_equal(a, b)


class TestMatmul:
    def test_matrix_vector(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 64))
        x = rng.standard_normal(64)
        result = matmul(A, x)
        np.testing.assert_allclose(result, A @ x)

    def test_matrix_matrix(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 64))
        B = rng.standard_normal((64, 10))
        result = matmul(A, B)
        np.testing.assert_allclose(result, A @ B)


class TestBatchInnerProduct:
    def test_single_query_multiple_vectors(self) -> None:
        rng = np.random.default_rng(42)
        query = rng.standard_normal(64)
        vectors = rng.standard_normal((100, 64))
        result = batch_inner_product(query, vectors)
        expected = vectors @ query
        np.testing.assert_allclose(result, expected)

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        query = rng.standard_normal(64)
        vectors = rng.standard_normal((50, 64))
        result = batch_inner_product(query, vectors)
        assert result.shape == (50,)


class TestHasTorch:
    def test_returns_bool(self) -> None:
        assert isinstance(has_torch(), bool)
