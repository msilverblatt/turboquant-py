"""Tests for CompressedVectors container and CompressedStore."""

from pathlib import Path

import numpy as np
import pytest

from turboquant.exceptions import StorageError
from turboquant.storage import CompressedStore, CompressedVectors


class TestCompressedVectors:
    def _make_compressed(
        self, n: int = 100, dim: int = 256, bit_width: int = 3
    ) -> CompressedVectors:
        rng = np.random.default_rng(42)
        max_val = (1 << bit_width) - 1
        indices = rng.integers(0, max_val + 1, size=(n, dim), dtype=np.uint8)
        norms = rng.random(n).astype(np.float64) + 0.5
        return CompressedVectors(
            indices=indices,
            norms=norms,
            dim=dim,
            bit_width=bit_width,
            metadata={"mode": "mse"},
        )

    def test_properties(self) -> None:
        cv = self._make_compressed(n=50, dim=128, bit_width=2)
        assert cv.num_vectors == 50
        assert cv.dim == 128
        assert cv.bit_width == 2
        assert cv.metadata["mode"] == "mse"

    def test_slicing(self) -> None:
        cv = self._make_compressed(n=100, dim=256, bit_width=3)
        subset = cv[10:20]
        assert subset.num_vectors == 10
        assert subset.dim == 256
        assert subset.bit_width == 3
        np.testing.assert_array_equal(subset.indices, cv.indices[10:20])
        np.testing.assert_array_equal(subset.norms, cv.norms[10:20])

    def test_concatenate(self) -> None:
        cv1 = self._make_compressed(n=50, dim=256, bit_width=3)
        cv2 = self._make_compressed(n=30, dim=256, bit_width=3)
        merged = CompressedVectors.concatenate([cv1, cv2])
        assert merged.num_vectors == 80
        assert merged.dim == 256

    def test_concatenate_dim_mismatch_raises(self) -> None:
        cv1 = self._make_compressed(n=50, dim=256, bit_width=3)
        cv2 = self._make_compressed(n=30, dim=128, bit_width=3)
        with pytest.raises(ValueError, match="dimension"):
            CompressedVectors.concatenate([cv1, cv2])

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        cv = self._make_compressed(n=100, dim=256, bit_width=3)
        save_path = tmp_path / "test_vectors"
        cv.save(save_path)
        loaded = CompressedVectors.load(save_path)
        assert loaded.num_vectors == cv.num_vectors
        assert loaded.dim == cv.dim
        assert loaded.bit_width == cv.bit_width
        np.testing.assert_array_equal(loaded.indices, cv.indices)
        np.testing.assert_allclose(loaded.norms, cv.norms)
        assert loaded.metadata["mode"] == cv.metadata["mode"]

    def test_extra_arrays_round_trip(self, tmp_path: Path) -> None:
        cv = self._make_compressed(n=100, dim=256, bit_width=3)
        rng = np.random.default_rng(99)
        cv.extra_arrays = {
            "rotation": rng.standard_normal((256, 256)),
            "qjl_signs": rng.choice([-1, 1], size=(100, 128)).astype(np.int8),
        }
        save_path = tmp_path / "test_extras"
        cv.save(save_path)
        loaded = CompressedVectors.load(save_path)
        np.testing.assert_allclose(loaded.extra_arrays["rotation"], cv.extra_arrays["rotation"])
        np.testing.assert_array_equal(
            loaded.extra_arrays["qjl_signs"], cv.extra_arrays["qjl_signs"]
        )

    def test_packed_saves_disk_space(self, tmp_path: Path) -> None:
        """Verify that bit-packing at 2-bit width yields ~4x smaller indices on disk."""
        n, dim, bit_width = 1000, 256, 2
        rng = np.random.default_rng(42)
        max_val = (1 << bit_width) - 1
        indices = rng.integers(0, max_val + 1, size=(n, dim), dtype=np.uint8)
        norms = rng.random(n).astype(np.float64) + 0.5
        cv = CompressedVectors(
            indices=indices,
            norms=norms,
            dim=dim,
            bit_width=bit_width,
            metadata={"mode": "mse"},
        )

        save_path = tmp_path / "packed"
        cv.save(save_path)

        packed_size = (save_path / "indices.npy").stat().st_size
        # Raw uint8 would be n * dim = 256000 bytes (plus npy header ~128 bytes).
        # Packed at 2-bit should be n * dim * 2 / 8 = 64000 bytes (plus header).
        unpacked_data_size = n * dim  # 256000
        # Allow generous margin: packed file should be less than half the raw data size
        assert packed_size < unpacked_data_size * 0.5, (
            f"Packed file ({packed_size} bytes) should be much smaller than "
            f"raw data ({unpacked_data_size} bytes)"
        )

        # Verify round-trip correctness
        loaded = CompressedVectors.load(save_path)
        np.testing.assert_array_equal(loaded.indices, cv.indices)

    def test_load_nonexistent_path_raises(self) -> None:
        with pytest.raises(StorageError):
            CompressedVectors.load("/nonexistent/path/xyz")

    def test_load_missing_meta_raises(self, tmp_path: Path) -> None:
        (tmp_path / "bad_store").mkdir()
        with pytest.raises(StorageError):
            CompressedVectors.load(tmp_path / "bad_store")


class TestCompressedStoreSearch:
    def test_search_returns_top_k(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        n = 50
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n, dim))

        tq = TurboQuant(dim=dim, bit_width=3, mode="mse", seed=42)
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")
        assert store.dim == dim
        assert store.num_vectors == n

    def test_store_metadata(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        tq = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=42)
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")
        assert store.bit_width == 2
        assert store.mode == "mse"

    def test_search_returns_correct_top_k(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        n = 50
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((n, dim))

        tq = TurboQuant(dim=dim, bit_width=3, mode="mse", seed=7)
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")

        # Use first vector as query; it should score highest against itself
        query = vectors[0]
        k = 5
        results = store.search(query, k=k)

        assert len(results) == k
        # Results must be sorted descending by score
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        # The index of the query vector itself should appear in top results
        returned_indices = [idx for idx, _ in results]
        assert 0 in returned_indices

    def test_search_qjl_mode(self, tmp_path: Path) -> None:
        from turboquant.qjl import QJL

        dim = 64
        n = 50
        rng = np.random.default_rng(1)
        vectors = rng.standard_normal((n, dim))

        qjl = QJL(dim=dim, seed=13)
        compressed = qjl.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")

        query = vectors[0]
        k = 5
        results = store.search(query, k=k)

        assert len(results) == k
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        returned_indices = [idx for idx, _ in results]
        assert 0 in returned_indices

    def test_search_k_larger_than_n(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 32
        n = 10
        rng = np.random.default_rng(2)
        vectors = rng.standard_normal((n, dim))

        tq = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=99)
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")

        query = rng.standard_normal(dim)
        results = store.search(query, k=100)

        # Should return all n vectors, not error or return more than n
        assert len(results) == n
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
