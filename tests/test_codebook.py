"""Tests for Lloyd-Max codebook computation and loading."""

import numpy as np
import pytest

from turboquant.codebook import (
    beta_pdf,
    compute_codebook,
    dequantize_scalar,
    get_codebook,
    quantize_scalar,
)


class TestBetaPdf:
    def test_integrates_to_one(self) -> None:
        for dim in [64, 256, 1024]:
            x = np.linspace(-0.999, 0.999, 10000)
            pdf_vals = beta_pdf(x, dim)
            integral = np.trapezoid(pdf_vals, x)
            np.testing.assert_allclose(integral, 1.0, atol=0.01)

    def test_symmetric(self) -> None:
        x = np.linspace(0.01, 0.99, 100)
        for dim in [64, 256, 1024]:
            np.testing.assert_allclose(beta_pdf(x, dim), beta_pdf(-x, dim), atol=1e-10)

    def test_converges_to_gaussian(self) -> None:
        dim = 4096
        x = np.linspace(-0.1, 0.1, 1000)
        beta_vals = beta_pdf(x, dim)
        sigma = 1.0 / np.sqrt(dim)
        gaussian_vals = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)
        beta_vals = beta_vals / np.trapezoid(beta_vals, x)
        gaussian_vals = gaussian_vals / np.trapezoid(gaussian_vals, x)
        np.testing.assert_allclose(beta_vals, gaussian_vals, atol=0.05)


class TestComputeCodebook:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, bit_width: int) -> None:
        centroids, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        assert len(centroids) == 2**bit_width
        assert len(boundaries) == 2**bit_width + 1

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_centroids_sorted(self, bit_width: int) -> None:
        centroids, _ = compute_codebook(dim=256, bit_width=bit_width)
        assert np.all(np.diff(centroids) > 0)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_boundaries_sorted(self, bit_width: int) -> None:
        _, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        assert np.all(np.diff(boundaries) > 0)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_boundaries_span_range(self, bit_width: int) -> None:
        _, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        assert boundaries[0] == -1.0
        assert boundaries[-1] == 1.0

    def test_1bit_centroids_symmetric(self) -> None:
        centroids, _ = compute_codebook(dim=256, bit_width=1)
        np.testing.assert_allclose(centroids[0], -centroids[1], atol=1e-6)

    def test_centroids_within_boundaries(self) -> None:
        for bit_width in [1, 2, 3, 4]:
            centroids, boundaries = compute_codebook(dim=256, bit_width=bit_width)
            for i, c in enumerate(centroids):
                assert boundaries[i] <= c <= boundaries[i + 1]


class TestGetCodebook:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_returns_cached_codebook(self, bit_width: int) -> None:
        c1, b1 = get_codebook(dim=256, bit_width=bit_width)
        c2, b2 = get_codebook(dim=256, bit_width=bit_width)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(b1, b2)


class TestQuantizeDequantize:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_round_trip_reduces_error_with_bit_width(self, bit_width: int) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(1000) / np.sqrt(256)
        values = np.clip(values, -0.99, 0.99)
        centroids, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        indices = quantize_scalar(values, boundaries)
        reconstructed = dequantize_scalar(indices, centroids)
        mse = np.mean((values - reconstructed) ** 2)
        assert 0 < mse < 1.0

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_indices_in_valid_range(self, bit_width: int) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(1000) / np.sqrt(256)
        values = np.clip(values, -0.99, 0.99)
        _, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        indices = quantize_scalar(values, boundaries)
        assert np.all(indices >= 0)
        assert np.all(indices < 2**bit_width)

    def test_mse_decreases_with_bit_width(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(10000) / np.sqrt(256)
        values = np.clip(values, -0.99, 0.99)
        mses = []
        for bw in [1, 2, 3, 4]:
            centroids, boundaries = compute_codebook(dim=256, bit_width=bw)
            indices = quantize_scalar(values, boundaries)
            reconstructed = dequantize_scalar(indices, centroids)
            mses.append(np.mean((values - reconstructed) ** 2))
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], (
                f"MSE did not decrease: bw={i+1}={mses[i]}, bw={i+2}={mses[i+1]}"
            )
