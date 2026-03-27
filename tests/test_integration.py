"""End-to-end integration tests."""

from pathlib import Path

import numpy as np
import pytest

from turboquant import QJL, CompressedVectors, TurboQuant


class TestQJLEndToEnd:
    def test_quantize_search_recovers_nearest(self) -> None:
        dim = 256
        n = 200
        rng = np.random.default_rng(42)

        vectors = rng.standard_normal((n, dim))
        query = rng.standard_normal(dim)

        true_scores = vectors @ query
        true_best = np.argmax(true_scores)

        qjl = QJL(dim=dim, seed=42)
        compressed = qjl.quantize(vectors)
        estimated_scores = qjl.inner_product(query, compressed)
        top_50 = np.argsort(estimated_scores)[-50:]
        assert true_best in top_50, f"True best {true_best} not in top-50 estimated {top_50}"


class TestTurboQuantEndToEnd:
    @pytest.mark.parametrize("mode", ["mse", "inner_product"])
    def test_quantize_dequantize_round_trip(self, mode: str) -> None:
        dim = 256
        bw = 3
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))

        tq = TurboQuant(dim=dim, bit_width=bw, mode=mode, seed=42)
        compressed = tq.quantize(vectors)
        reconstructed = tq.dequantize(compressed)

        for i in range(50):
            corr = np.corrcoef(vectors[i], reconstructed[i])[0, 1]
            assert corr > 0.5, f"Vector {i} correlation {corr} too low"

    def test_inner_product_search_recovers_nearest(self) -> None:
        dim = 256
        n = 200
        rng = np.random.default_rng(42)

        vectors = rng.standard_normal((n, dim))
        query = rng.standard_normal(dim)

        true_scores = vectors @ query
        true_best = np.argmax(true_scores)

        tq = TurboQuant(dim=dim, bit_width=3, mode="inner_product", seed=42)
        compressed = tq.quantize(vectors)
        estimated_scores = tq.inner_product(query, compressed)

        top_5 = np.argsort(estimated_scores)[-5:]
        assert true_best in top_5, f"True best {true_best} not in top-5 estimated {top_5}"

    @pytest.mark.parametrize("mode", ["mse", "inner_product"])
    def test_save_load_preserves_search(self, tmp_path: Path, mode: str) -> None:
        dim = 64
        bw = 3 if mode == "inner_product" else 2
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((30, dim))
        query = rng.standard_normal(dim)

        tq = TurboQuant(dim=dim, bit_width=bw, mode=mode, seed=42)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / f"test_store_{mode}")
        loaded = CompressedVectors.load(tmp_path / f"test_store_{mode}")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after)


class TestBatchQuantization:
    def test_batched_matches_single(self, tmp_path: Path) -> None:
        dim = 64
        rng = np.random.default_rng(42)
        all_vectors = rng.standard_normal((100, dim))

        tq = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=42)

        single_compressed = tq.quantize(all_vectors)
        single_reconstructed = tq.dequantize(single_compressed)

        def vector_iterator():
            for i in range(0, 100, 25):
                yield all_vectors[i : i + 25]

        tq.quantize_batched(vector_iterator(), batch_size=25, output_path=tmp_path / "batched")
        batched_loaded = CompressedVectors.load(tmp_path / "batched")
        batched_reconstructed = tq.dequantize(batched_loaded)

        np.testing.assert_allclose(single_reconstructed, batched_reconstructed, atol=1e-10)

    def test_batched_inner_product_mode(self, tmp_path: Path) -> None:
        """quantize_batched should work with inner_product mode."""
        dim = 64
        rng = np.random.default_rng(42)
        all_vectors = rng.standard_normal((100, dim))

        tq = TurboQuant(dim=dim, bit_width=3, mode="inner_product", seed=42)

        single_compressed = tq.quantize(all_vectors)
        single_reconstructed = tq.dequantize(single_compressed)

        def vector_iterator():
            for i in range(0, 100, 25):
                yield all_vectors[i : i + 25]

        tq.quantize_batched(vector_iterator(), batch_size=25, output_path=tmp_path / "batched_ip")
        batched_loaded = CompressedVectors.load(tmp_path / "batched_ip")
        batched_reconstructed = tq.dequantize(batched_loaded)

        np.testing.assert_allclose(single_reconstructed, batched_reconstructed, atol=1e-10)


class TestDimensionSweep:
    """Verify quantization works correctly across all spec dimensions."""

    @pytest.mark.parametrize("dim", [384, 512, 768, 1024, 1536, 2048, 3072])
    def test_turboquant_mse_across_dimensions(self, dim: int) -> None:
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms

        tq = TurboQuant(dim=dim, bit_width=3, mode="mse", seed=42)
        compressed = tq.quantize(unit_vectors)
        reconstructed = tq.dequantize(compressed)
        mse = np.mean(np.sum((unit_vectors - reconstructed) ** 2, axis=1))

        upper_bound = np.sqrt(3 * np.pi) / 2 * (1 / 4**3)
        assert mse < upper_bound * 2, f"MSE {mse:.6f} exceeds bound at dim={dim}"

    @pytest.mark.parametrize("dim", [384, 512, 768, 1024, 1536, 2048, 3072])
    def test_qjl_across_dimensions(self, dim: int) -> None:
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))
        query = rng.standard_normal(dim)

        qjl = QJL(dim=dim, seed=42)
        compressed = qjl.quantize(vectors)
        scores = qjl.inner_product(query, compressed)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))
