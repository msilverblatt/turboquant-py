"""Tests for CompressedVectors container and CompressedStore."""

from pathlib import Path

import numpy as np
import pytest

from turboquant.storage import CompressedVectors


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
