"""Tests for TurboQuant MSE and inner-product quantizers."""

import numpy as np
import pytest

from turboquant.exceptions import DimensionMismatchError, InvalidBitWidthError, InvalidModeError
from turboquant.turboquant import TurboQuant


class TestTurboQuantConstruction:
    def test_valid_construction(self) -> None:
        tq = TurboQuant(dim=256, bit_width=3, mode="mse", seed=42)
        assert tq.dim == 256
        assert tq.bit_width == 3
        assert tq.mode == "mse"

    def test_invalid_bit_width_raises(self) -> None:
        with pytest.raises(InvalidBitWidthError):
            TurboQuant(dim=256, bit_width=0)
        with pytest.raises(InvalidBitWidthError):
            TurboQuant(dim=256, bit_width=5)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(InvalidModeError):
            TurboQuant(dim=256, bit_width=3, mode="bad")

    def test_deterministic_with_seed(self) -> None:
        t1 = TurboQuant(dim=64, bit_width=2, seed=42)
        t2 = TurboQuant(dim=64, bit_width=2, seed=42)
        np.testing.assert_array_equal(t1._rotation, t2._rotation)

    def test_inner_product_mode_needs_bit_width_ge_2(self) -> None:
        with pytest.raises(InvalidBitWidthError):
            TurboQuant(dim=256, bit_width=1, mode="inner_product")


class TestTurboQuantMSE:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_quantize_output_shape(self, bit_width: int) -> None:
        tq = TurboQuant(dim=64, bit_width=bit_width, mode="mse", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = tq.quantize(vectors)
        assert compressed.num_vectors == 10
        assert compressed.dim == 64
        assert compressed.bit_width == bit_width

    def test_dimension_mismatch_raises(self) -> None:
        tq = TurboQuant(dim=64, bit_width=2, mode="mse", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 128))
        with pytest.raises(DimensionMismatchError):
            tq.quantize(vectors)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_dequantize_output_shape(self, bit_width: int) -> None:
        tq = TurboQuant(dim=64, bit_width=bit_width, mode="mse", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = tq.quantize(vectors)
        reconstructed = tq.dequantize(compressed)
        assert reconstructed.shape == (10, 64)

    def test_mse_decreases_with_bit_width(self) -> None:
        dim = 256
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, dim))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        mses = []
        for bw in [1, 2, 3, 4]:
            tq = TurboQuant(dim=dim, bit_width=bw, mode="mse", seed=42)
            compressed = tq.quantize(vectors)
            reconstructed = tq.dequantize(compressed)
            mse = np.mean(np.sum((vectors - reconstructed) ** 2, axis=1))
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], (
                f"MSE did not decrease: bw={i + 1} mse={mses[i]:.6f}, "
                f"bw={i + 2} mse={mses[i + 1]:.6f}"
            )

    def test_distortion_within_theoretical_bound(self) -> None:
        dim = 512
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((200, dim))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        for bw in [2, 3, 4]:
            tq = TurboQuant(dim=dim, bit_width=bw, mode="mse", seed=42)
            compressed = tq.quantize(vectors)
            reconstructed = tq.dequantize(compressed)
            mse = np.mean(np.sum((vectors - reconstructed) ** 2, axis=1))

            upper_bound = np.sqrt(3 * np.pi) / 2 * (1 / 4**bw)
            assert mse < upper_bound * 2, (
                f"MSE {mse:.6f} exceeds 2x theoretical bound {upper_bound:.6f} at bw={bw}"
            )


class TestTurboQuantInnerProduct:
    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_inner_product_output_shape(self, bit_width: int) -> None:
        tq = TurboQuant(dim=64, bit_width=bit_width, mode="inner_product", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((20, 64))
        query = rng.standard_normal(64)
        compressed = tq.quantize(vectors)
        scores = tq.inner_product(query, compressed)
        assert scores.shape == (20,)

    def test_unbiased_estimator(self) -> None:
        dim = 256
        n_trials = 200
        rng = np.random.default_rng(42)

        x = rng.standard_normal(dim)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(dim)
        true_ip = np.dot(x, y)

        errors = []
        for trial in range(n_trials):
            tq = TurboQuant(dim=dim, bit_width=3, mode="inner_product", seed=trial)
            compressed = tq.quantize(x.reshape(1, -1))
            estimated = tq.inner_product(y, compressed)[0]
            errors.append(estimated - true_ip)

        mean_error = np.mean(errors)
        assert abs(mean_error) < 0.15, f"Mean error {mean_error} too large for unbiased estimator"

    def test_dimension_mismatch_on_query_raises(self) -> None:
        tq = TurboQuant(dim=64, bit_width=2, mode="inner_product", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = tq.quantize(vectors)
        bad_query = rng.standard_normal(128)
        with pytest.raises(DimensionMismatchError):
            tq.inner_product(bad_query, compressed)


class TestTurboQuantDeterminism:
    def test_same_seed_same_result(self) -> None:
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((10, 64))

        tq1 = TurboQuant(dim=64, bit_width=3, mode="mse", seed=99)
        tq2 = TurboQuant(dim=64, bit_width=3, mode="mse", seed=99)

        c1 = tq1.quantize(vectors)
        c2 = tq2.quantize(vectors)

        r1 = tq1.dequantize(c1)
        r2 = tq2.dequantize(c2)

        np.testing.assert_array_equal(r1, r2)
