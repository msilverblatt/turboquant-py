"""Tests for outlier channel handling in TurboQuant."""

from pathlib import Path

import numpy as np

from turboquant.storage import CompressedStore, CompressedVectors
from turboquant.turboquant import TurboQuant


class TestOutlierHandling:
    def test_outlier_channels_produces_valid_output(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        vectors[:, :4] *= 10.0

        tq = TurboQuant(
            dim=dim,
            bit_width=2,
            mode="mse",
            seed=42,
            outlier_channels=4,
            outlier_bit_width=3,
        )
        compressed = tq.quantize(vectors)
        assert compressed.num_vectors == 20
        assert compressed.dim == dim

    def test_outlier_reduces_mse_on_outlier_data(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, dim))
        vectors[:, :4] *= 10.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms

        tq_no_outlier = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=42)
        c1 = tq_no_outlier.quantize(unit_vectors)
        r1 = tq_no_outlier.dequantize(c1)
        mse_no_outlier = np.mean(np.sum((unit_vectors - r1) ** 2, axis=1))

        tq_outlier = TurboQuant(
            dim=dim,
            bit_width=2,
            mode="mse",
            seed=42,
            outlier_channels=4,
            outlier_bit_width=3,
        )
        c2 = tq_outlier.quantize(unit_vectors)
        r2 = tq_outlier.dequantize(c2)
        mse_outlier = np.mean(np.sum((unit_vectors - r2) ** 2, axis=1))

        assert mse_outlier < mse_no_outlier, (
            f"Outlier handling MSE {mse_outlier:.6f} should be less than "
            f"no-outlier MSE {mse_no_outlier:.6f}"
        )

    def test_dequantize_round_trip_with_outliers(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        vectors[:, :4] *= 10.0

        tq = TurboQuant(
            dim=dim,
            bit_width=3,
            mode="mse",
            seed=42,
            outlier_channels=4,
            outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        reconstructed = tq.dequantize(compressed)
        assert reconstructed.shape == (20, dim)
        for i in range(20):
            corr = np.corrcoef(vectors[i], reconstructed[i])[0, 1]
            assert corr > 0.5

    def test_inner_product_mode_with_outliers(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        vectors[:, :4] *= 10.0
        query = rng.standard_normal(dim)

        tq = TurboQuant(
            dim=dim,
            bit_width=3,
            mode="inner_product",
            seed=42,
            outlier_channels=4,
            outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        scores = tq.inner_product(query, compressed)
        assert scores.shape == (20,)

    def test_save_load_round_trip_with_outliers(self, tmp_path: Path) -> None:
        """Outlier config must survive save/load and produce same scores."""
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        vectors[:, :4] *= 10.0
        query = rng.standard_normal(dim)

        tq = TurboQuant(
            dim=dim,
            bit_width=3,
            mode="inner_product",
            seed=42,
            outlier_channels=4,
            outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "outlier_store")
        loaded = CompressedVectors.load(tmp_path / "outlier_store")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_compressed_store_search_with_outliers(self, tmp_path: Path) -> None:
        """CompressedStore.search() must work correctly with outlier config."""
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))
        vectors[:, :4] *= 10.0
        query = rng.standard_normal(dim)

        tq = TurboQuant(
            dim=dim,
            bit_width=3,
            mode="mse",
            seed=42,
            outlier_channels=4,
            outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        scores_direct = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "outlier_search_store")
        store = CompressedStore.load(tmp_path / "outlier_search_store")
        results = store.search(query, k=5)

        # Verify store.search returns same top results as direct inner_product
        top_5_direct = np.argsort(scores_direct)[-5:][::-1]
        top_5_store = [idx for idx, _ in results]
        assert set(top_5_direct) == set(top_5_store)
