"""Adversarial tests targeting silent failure modes.

These tests are designed to catch bugs that produce wrong results while
still passing existing tests -- e.g., outlier handling breaking after
save/load, or MSE mode's inner_product method returning garbage.
"""

from pathlib import Path

import numpy as np
import pytest

from turboquant.qjl import QJL
from turboquant.storage import CompressedStore, CompressedVectors
from turboquant.turboquant import TurboQuant

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 128
SEED = 42
N_VECTORS = 30


def _make_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    return rng.standard_normal((n, dim))


def _make_query(rng: np.random.Generator, dim: int) -> np.ndarray:
    return rng.standard_normal(dim)


def _make_outlier_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    vectors = rng.standard_normal((n, dim))
    vectors[:, :4] *= 10.0
    return vectors


# ---------------------------------------------------------------------------
# 1. Cross-mode consistency
# ---------------------------------------------------------------------------


class TestCrossModeConsistency:
    """Inner-product mode uses MSE at (b-1) bits; indices should match."""

    def test_mse_indices_match_inner_product_mse_component(self) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)

        tq_mse = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        tq_ip = TurboQuant(dim=DIM, bit_width=3, mode="inner_product", seed=SEED)

        c_mse = tq_mse.quantize(vectors)
        c_ip = tq_ip.quantize(vectors)

        # inner_product at bw=3 uses MSE at bw=2 internally; indices must match
        np.testing.assert_array_equal(
            c_mse.indices,
            c_ip.indices,
            err_msg="MSE(bw=2) indices differ from inner_product(bw=3) MSE component",
        )

    @pytest.mark.parametrize("ip_bw", [2, 3, 4])
    def test_mse_indices_match_at_various_bit_widths(self, ip_bw: int) -> None:
        mse_bw = ip_bw - 1
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)

        tq_mse = TurboQuant(dim=DIM, bit_width=mse_bw, mode="mse", seed=SEED)
        tq_ip = TurboQuant(dim=DIM, bit_width=ip_bw, mode="inner_product", seed=SEED)

        c_mse = tq_mse.quantize(vectors)
        c_ip = tq_ip.quantize(vectors)

        np.testing.assert_array_equal(c_mse.indices, c_ip.indices)


# ---------------------------------------------------------------------------
# 2. Save/load score equivalence (EVERY mode)
# ---------------------------------------------------------------------------


class TestSaveLoadScoreEquivalence:
    """Scores computed in-memory must EXACTLY match scores after save/load."""

    def test_qjl_scores_exact_after_save_load(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        qjl = QJL(dim=DIM, seed=SEED)
        compressed = qjl.quantize(vectors)
        scores_before = qjl.inner_product(query, compressed)

        compressed.save(tmp_path / "qjl")
        loaded = CompressedVectors.load(tmp_path / "qjl")
        scores_after = qjl.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_turboquant_mse_scores_exact_after_save_load(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "mse")
        loaded = CompressedVectors.load(tmp_path / "mse")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_turboquant_ip_scores_exact_after_save_load(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=3, mode="inner_product", seed=SEED)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "ip")
        loaded = CompressedVectors.load(tmp_path / "ip")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_turboquant_mse_outlier_scores_exact_after_save_load(
        self, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_outlier_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(
            dim=DIM,
            bit_width=2,
            mode="mse",
            seed=SEED,
            outlier_channels=4,
            outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "mse_outlier")
        loaded = CompressedVectors.load(tmp_path / "mse_outlier")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_turboquant_ip_outlier_scores_exact_after_save_load(
        self, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_outlier_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(
            dim=DIM,
            bit_width=3,
            mode="inner_product",
            seed=SEED,
            outlier_channels=4,
            outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "ip_outlier")
        loaded = CompressedVectors.load(tmp_path / "ip_outlier")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. CompressedStore.search consistency
# ---------------------------------------------------------------------------


class TestCompressedStoreSearchConsistency:
    """store.search() must return the same top-k as manual inner_product."""

    @pytest.mark.parametrize(
        "mode,bit_width,outlier_channels,outlier_bit_width",
        [
            ("mse", 2, 0, None),
            ("inner_product", 3, 0, None),
            ("mse", 2, 4, 4),
            ("inner_product", 3, 4, 4),
        ],
    )
    def test_store_search_matches_manual_inner_product(
        self,
        tmp_path: Path,
        mode: str,
        bit_width: int,
        outlier_channels: int,
        outlier_bit_width: int | None,
    ) -> None:
        rng = np.random.default_rng(SEED)
        make = _make_outlier_vectors if outlier_channels else _make_vectors
        vectors = make(rng, 50, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(
            dim=DIM,
            bit_width=bit_width,
            mode=mode,
            seed=SEED,
            outlier_channels=outlier_channels,
            outlier_bit_width=outlier_bit_width,
        )
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")
        k = 5
        search_results = store.search(query, k=k)
        search_indices = [idx for idx, _ in search_results]
        search_scores = [score for _, score in search_results]

        # Manually compute using loaded vectors
        loaded = CompressedVectors.load(tmp_path / "store")
        tq_fresh = TurboQuant(
            dim=DIM,
            bit_width=bit_width,
            mode=mode,
            seed=SEED,
            outlier_channels=outlier_channels,
            outlier_bit_width=outlier_bit_width,
        )
        manual_scores = tq_fresh.inner_product(query, loaded)
        manual_top_k = np.argsort(manual_scores)[-k:][::-1].tolist()

        assert search_indices == manual_top_k, (
            f"Store search indices {search_indices} != manual top-k {manual_top_k}"
        )
        for s_score, m_idx in zip(search_scores, manual_top_k, strict=True):
            np.testing.assert_allclose(s_score, manual_scores[m_idx], atol=1e-10)

    def test_store_search_qjl_matches_manual(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, 50, DIM)
        query = _make_query(rng, DIM)

        qjl = QJL(dim=DIM, seed=SEED)
        compressed = qjl.quantize(vectors)
        compressed.save(tmp_path / "qjl_store")

        store = CompressedStore.load(tmp_path / "qjl_store")
        k = 5
        search_results = store.search(query, k=k)
        search_indices = [idx for idx, _ in search_results]

        loaded = CompressedVectors.load(tmp_path / "qjl_store")
        qjl_fresh = QJL(dim=DIM, seed=SEED)
        manual_scores = qjl_fresh.inner_product(query, loaded)
        manual_top_k = np.argsort(manual_scores)[-k:][::-1].tolist()

        assert search_indices == manual_top_k


# ---------------------------------------------------------------------------
# 4. Bit-packing does not corrupt data
# ---------------------------------------------------------------------------


class TestBitPackingIntegrity:
    """Dequantized results must be identical before save vs after load."""

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_dequantize_identical_after_save_load(
        self, tmp_path: Path, bit_width: int
    ) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)

        tq = TurboQuant(dim=DIM, bit_width=bit_width, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        deq_before = tq.dequantize(compressed)

        compressed.save(tmp_path / f"bw{bit_width}")
        loaded = CompressedVectors.load(tmp_path / f"bw{bit_width}")
        deq_after = tq.dequantize(loaded)

        np.testing.assert_array_equal(
            deq_before,
            deq_after,
            err_msg=f"Dequantized results differ after save/load at bit_width={bit_width}",
        )

    @pytest.mark.parametrize(
        "bit_width,outlier_bit_width",
        [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ],
    )
    def test_outlier_bit_packing_no_truncation(
        self, tmp_path: Path, bit_width: int, outlier_bit_width: int
    ) -> None:
        """When outlier_bit_width > bit_width, indices must not be truncated."""
        rng = np.random.default_rng(SEED)
        vectors = _make_outlier_vectors(rng, N_VECTORS, DIM)

        tq = TurboQuant(
            dim=DIM,
            bit_width=bit_width,
            mode="mse",
            seed=SEED,
            outlier_channels=4,
            outlier_bit_width=outlier_bit_width,
        )
        compressed = tq.quantize(vectors)
        deq_before = tq.dequantize(compressed)

        compressed.save(tmp_path / f"outlier_bw{bit_width}_obw{outlier_bit_width}")
        loaded = CompressedVectors.load(
            tmp_path / f"outlier_bw{bit_width}_obw{outlier_bit_width}"
        )
        deq_after = tq.dequantize(loaded)

        np.testing.assert_array_equal(
            deq_before,
            deq_after,
            err_msg=(
                f"Dequantized results differ after save/load with "
                f"bit_width={bit_width}, outlier_bit_width={outlier_bit_width}"
            ),
        )

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_raw_indices_survive_pack_unpack(
        self, tmp_path: Path, bit_width: int
    ) -> None:
        """Raw indices must be identical after pack/save/load/unpack."""
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)

        tq = TurboQuant(dim=DIM, bit_width=bit_width, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        indices_before = compressed.indices.copy()

        compressed.save(tmp_path / f"idx_bw{bit_width}")
        loaded = CompressedVectors.load(tmp_path / f"idx_bw{bit_width}")

        np.testing.assert_array_equal(loaded.indices, indices_before)


# ---------------------------------------------------------------------------
# 5. Seed determinism across save/load
# ---------------------------------------------------------------------------


class TestSeedDeterminismAcrossSaveLoad:
    """A fresh quantizer with the same seed must produce same scores on loaded data."""

    def test_fresh_quantizer_same_seed_same_scores(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        tq1 = TurboQuant(dim=DIM, bit_width=3, mode="inner_product", seed=SEED)
        compressed = tq1.quantize(vectors)
        scores_original = tq1.inner_product(query, compressed)

        compressed.save(tmp_path / "seed_test")

        # Create a completely new quantizer with the same seed
        tq2 = TurboQuant(dim=DIM, bit_width=3, mode="inner_product", seed=SEED)
        loaded = CompressedVectors.load(tmp_path / "seed_test")
        scores_loaded = tq2.inner_product(query, loaded)

        np.testing.assert_allclose(scores_original, scores_loaded, atol=1e-10)

    def test_fresh_quantizer_mse_seed_determinism(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        tq1 = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        compressed = tq1.quantize(vectors)
        scores_original = tq1.inner_product(query, compressed)

        compressed.save(tmp_path / "mse_seed_test")

        tq2 = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        loaded = CompressedVectors.load(tmp_path / "mse_seed_test")
        scores_loaded = tq2.inner_product(query, loaded)

        np.testing.assert_allclose(scores_original, scores_loaded, atol=1e-10)

    def test_fresh_qjl_seed_determinism(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, N_VECTORS, DIM)
        query = _make_query(rng, DIM)

        qjl1 = QJL(dim=DIM, seed=SEED)
        compressed = qjl1.quantize(vectors)
        scores_original = qjl1.inner_product(query, compressed)

        compressed.save(tmp_path / "qjl_seed_test")

        qjl2 = QJL(dim=DIM, seed=SEED)
        loaded = CompressedVectors.load(tmp_path / "qjl_seed_test")
        scores_loaded = qjl2.inner_product(query, loaded)

        np.testing.assert_allclose(scores_original, scores_loaded, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. quantize_batched equivalence for all modes
# ---------------------------------------------------------------------------


class TestQuantizeBatchedEquivalence:
    """Batched quantization must produce identical results to single-shot."""

    @pytest.mark.parametrize("mode", ["mse", "inner_product"])
    def test_batched_dequantize_matches_single(
        self, tmp_path: Path, mode: str
    ) -> None:
        bit_width = 3 if mode == "inner_product" else 2
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, 80, DIM)

        tq = TurboQuant(dim=DIM, bit_width=bit_width, mode=mode, seed=SEED)

        single_compressed = tq.quantize(vectors)
        single_deq = tq.dequantize(single_compressed)

        def batch_iter():
            for i in range(0, 80, 20):
                yield vectors[i : i + 20]

        tq.quantize_batched(batch_iter(), output_path=tmp_path / "batched")
        batched_loaded = CompressedVectors.load(tmp_path / "batched")
        batched_deq = tq.dequantize(batched_loaded)

        np.testing.assert_allclose(single_deq, batched_deq, atol=1e-10)

    def test_batched_with_outliers_matches_single(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_outlier_vectors(rng, 80, DIM)

        tq = TurboQuant(
            dim=DIM,
            bit_width=2,
            mode="mse",
            seed=SEED,
            outlier_channels=4,
            outlier_bit_width=3,
        )

        single_compressed = tq.quantize(vectors)
        single_deq = tq.dequantize(single_compressed)

        def batch_iter():
            for i in range(0, 80, 20):
                yield vectors[i : i + 20]

        tq.quantize_batched(batch_iter(), output_path=tmp_path / "batched_outlier")
        batched_loaded = CompressedVectors.load(tmp_path / "batched_outlier")
        batched_deq = tq.dequantize(batched_loaded)

        np.testing.assert_allclose(single_deq, batched_deq, atol=1e-10)

    @pytest.mark.parametrize("mode", ["mse", "inner_product"])
    def test_batched_scores_match_single(self, tmp_path: Path, mode: str) -> None:
        """Inner product scores from batched must match single-shot."""
        bit_width = 3 if mode == "inner_product" else 2
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, 80, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=bit_width, mode=mode, seed=SEED)

        single_compressed = tq.quantize(vectors)
        single_scores = tq.inner_product(query, single_compressed)

        def batch_iter():
            for i in range(0, 80, 20):
                yield vectors[i : i + 20]

        tq.quantize_batched(
            batch_iter(), output_path=tmp_path / f"batched_scores_{mode}"
        )
        batched_loaded = CompressedVectors.load(tmp_path / f"batched_scores_{mode}")
        batched_scores = tq.inner_product(query, batched_loaded)

        np.testing.assert_allclose(single_scores, batched_scores, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that could silently produce wrong results."""

    def test_single_vector(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, 1, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "single_vec")
        loaded = CompressedVectors.load(tmp_path / "single_vec")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)
        assert scores_after.shape == (1,)

    def test_high_dimensional(self, tmp_path: Path) -> None:
        dim = 4096
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, 5, dim)
        query = _make_query(rng, dim)

        tq = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "high_dim")
        loaded = CompressedVectors.load(tmp_path / "high_dim")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_all_zeros_vector(self) -> None:
        """All-zeros vector should not cause NaN or crash."""
        rng = np.random.default_rng(SEED)
        vectors = np.zeros((1, DIM))
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        scores = tq.inner_product(query, compressed)

        assert not np.any(np.isnan(scores)), "All-zeros vector produced NaN scores"
        assert not np.any(np.isinf(scores)), "All-zeros vector produced Inf scores"

    def test_all_same_vector(self, tmp_path: Path) -> None:
        """Vector with all identical elements."""
        vectors = np.full((5, DIM), 3.14)
        rng = np.random.default_rng(SEED)
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "all_same")
        loaded = CompressedVectors.load(tmp_path / "all_same")
        scores_after = tq.inner_product(query, loaded)

        assert not np.any(np.isnan(scores_before))
        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_extreme_norm_disparity(self, tmp_path: Path) -> None:
        """Mix of very large and very small norm vectors in the same batch."""
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, 20, DIM)
        # First 10 have tiny norms, last 10 have huge norms
        vectors[:10] *= 1e-8
        vectors[10:] *= 1e6
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=2, mode="mse", seed=SEED)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "extreme_norms")
        loaded = CompressedVectors.load(tmp_path / "extreme_norms")
        scores_after = tq.inner_product(query, loaded)

        assert not np.any(np.isnan(scores_before))
        assert not np.any(np.isinf(scores_before))
        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_single_vector_inner_product_mode(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SEED)
        vectors = _make_vectors(rng, 1, DIM)
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=3, mode="inner_product", seed=SEED)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "single_vec_ip")
        loaded = CompressedVectors.load(tmp_path / "single_vec_ip")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-10)

    def test_all_zeros_inner_product_mode(self) -> None:
        rng = np.random.default_rng(SEED)
        vectors = np.zeros((3, DIM))
        query = _make_query(rng, DIM)

        tq = TurboQuant(dim=DIM, bit_width=3, mode="inner_product", seed=SEED)
        compressed = tq.quantize(vectors)
        scores = tq.inner_product(query, compressed)

        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))
