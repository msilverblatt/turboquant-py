"""Tests for entropy encoding of codebook indices."""

from pathlib import Path

import numpy as np
import pytest

from turboquant._entropy import (
    build_huffman_table,
    compute_symbol_probabilities,
    compute_theoretical_savings,
    huffman_decode,
    huffman_encode,
)
from turboquant.storage import CompressedVectors


class TestSymbolProbabilities:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    @pytest.mark.parametrize("dim", [64, 256, 1024])
    def test_probabilities_sum_to_one(self, bit_width: int, dim: int) -> None:
        probs = compute_symbol_probabilities(dim, bit_width)
        assert len(probs) == (1 << bit_width)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_all_probabilities_positive(self, bit_width: int) -> None:
        probs = compute_symbol_probabilities(256, bit_width)
        assert np.all(probs > 0)

    def test_symmetric_distribution(self) -> None:
        """The Beta distribution is symmetric, so probabilities should be symmetric."""
        probs = compute_symbol_probabilities(256, 4)
        np.testing.assert_allclose(probs, probs[::-1], atol=1e-6)


class TestHuffmanTable:
    def test_all_symbols_have_codes(self) -> None:
        probs = compute_symbol_probabilities(256, 3)
        table = build_huffman_table(probs)
        assert len(table) == len(probs)
        for i in range(len(probs)):
            assert i in table

    def test_prefix_free(self) -> None:
        """No code should be a prefix of another."""
        probs = compute_symbol_probabilities(256, 4)
        table = build_huffman_table(probs)
        codes = list(table.values())
        for i, c1 in enumerate(codes):
            for j, c2 in enumerate(codes):
                if i != j:
                    assert not c2.startswith(c1), f"Code {c1} is a prefix of {c2}"

    def test_single_symbol(self) -> None:
        probs = np.array([1.0])
        table = build_huffman_table(probs)
        assert table == {0: "0"}

    def test_two_symbols(self) -> None:
        probs = np.array([0.5, 0.5])
        table = build_huffman_table(probs)
        assert len(table) == 2
        assert all(len(v) == 1 for v in table.values())


class TestHuffmanRoundTrip:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_encode_decode_random(self, bit_width: int) -> None:
        rng = np.random.default_rng(42)
        max_val = (1 << bit_width) - 1
        indices = rng.integers(0, max_val + 1, size=1000, dtype=np.uint8)

        probs = compute_symbol_probabilities(256, bit_width)
        table = build_huffman_table(probs)

        encoded = huffman_encode(indices, table)
        decoded = huffman_decode(encoded, table, len(indices))
        np.testing.assert_array_equal(decoded, indices)

    def test_encode_decode_all_same(self) -> None:
        indices = np.zeros(500, dtype=np.uint8)
        probs = compute_symbol_probabilities(256, 3)
        table = build_huffman_table(probs)

        encoded = huffman_encode(indices, table)
        decoded = huffman_decode(encoded, table, len(indices))
        np.testing.assert_array_equal(decoded, indices)

    def test_encode_decode_single_element(self) -> None:
        indices = np.array([5], dtype=np.uint8)
        probs = compute_symbol_probabilities(256, 4)
        table = build_huffman_table(probs)

        encoded = huffman_encode(indices, table)
        decoded = huffman_decode(encoded, table, 1)
        np.testing.assert_array_equal(decoded, indices)


class TestSaveLoadWithEntropy:
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

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_save_load_round_trip(self, tmp_path: Path, bit_width: int) -> None:
        cv = self._make_compressed(bit_width=bit_width)
        save_path = tmp_path / f"entropy_bw{bit_width}"
        cv.save(save_path, entropy_encode=True)
        loaded = CompressedVectors.load(save_path)
        np.testing.assert_array_equal(loaded.indices, cv.indices)
        np.testing.assert_allclose(loaded.norms, cv.norms)
        assert loaded.dim == cv.dim
        assert loaded.bit_width == cv.bit_width

    def test_identical_results_with_and_without(self, tmp_path: Path) -> None:
        cv = self._make_compressed()
        path_normal = tmp_path / "normal"
        path_entropy = tmp_path / "entropy"
        cv.save(path_normal, entropy_encode=False)
        cv.save(path_entropy, entropy_encode=True)

        loaded_normal = CompressedVectors.load(path_normal)
        loaded_entropy = CompressedVectors.load(path_entropy)

        np.testing.assert_array_equal(loaded_normal.indices, loaded_entropy.indices)
        np.testing.assert_allclose(loaded_normal.norms, loaded_entropy.norms)

    def test_entropy_files_exist(self, tmp_path: Path) -> None:
        cv = self._make_compressed()
        save_path = tmp_path / "entropy_files"
        cv.save(save_path, entropy_encode=True)

        assert (save_path / "indices.huffman").exists()
        assert (save_path / "huffman_table.json").exists()
        assert (save_path / "meta.json").exists()
        assert (save_path / "norms.npy").exists()
        # indices.npy should NOT exist when entropy encoded
        assert not (save_path / "indices.npy").exists()

    def test_file_size_reduction(self, tmp_path: Path) -> None:
        """Entropy encoding should be smaller than raw uint8 storage."""
        rng = np.random.default_rng(42)
        n, dim, bw = 10_000, 256, 4
        max_val = (1 << bw) - 1
        indices = rng.integers(0, max_val + 1, size=(n, dim), dtype=np.uint8)
        norms = rng.random(n).astype(np.float64) + 0.5
        meta = {"mode": "mse", "seed": 42}
        cv = CompressedVectors(
            indices=indices, norms=norms, dim=dim, bit_width=bw, metadata=meta
        )

        cv.save(tmp_path / "entropy", entropy_encode=True)

        entropy_size = (tmp_path / "entropy" / "indices.huffman").stat().st_size
        raw_size = n * dim  # 1 byte per index without any packing

        assert entropy_size < raw_size, (
            f"Entropy encoded ({entropy_size}) should be smaller than raw ({raw_size})"
        )

    def test_extra_arrays_preserved(self, tmp_path: Path) -> None:
        cv = self._make_compressed()
        rng = np.random.default_rng(99)
        cv.extra_arrays = {
            "rotation": rng.standard_normal((256, 256)),
        }
        save_path = tmp_path / "entropy_extras"
        cv.save(save_path, entropy_encode=True)
        loaded = CompressedVectors.load(save_path)
        np.testing.assert_allclose(loaded.extra_arrays["rotation"], cv.extra_arrays["rotation"])


class TestEntropyWithModes:
    def test_mse_mode(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))
        tq = TurboQuant(dim=dim, bit_width=3, mode="mse", seed=42)
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "mse_entropy", entropy_encode=True)
        loaded = CompressedVectors.load(tmp_path / "mse_entropy")
        np.testing.assert_array_equal(loaded.indices, compressed.indices)

    def test_inner_product_mode(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))
        tq = TurboQuant(dim=dim, bit_width=3, mode="inner_product", seed=42)
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "ip_entropy", entropy_encode=True)
        loaded = CompressedVectors.load(tmp_path / "ip_entropy")
        np.testing.assert_array_equal(loaded.indices, compressed.indices)

    def test_qjl_mode(self, tmp_path: Path) -> None:
        from turboquant.qjl import QJL

        dim = 64
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))
        qjl = QJL(dim=dim, seed=42)
        compressed = qjl.quantize(vectors)
        compressed.save(tmp_path / "qjl_entropy", entropy_encode=True)
        loaded = CompressedVectors.load(tmp_path / "qjl_entropy")
        np.testing.assert_array_equal(loaded.indices, compressed.indices)

    def test_outlier_mode(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))
        tq = TurboQuant(
            dim=dim,
            bit_width=3,
            mode="mse",
            seed=42,
            outlier_channels=4,
            outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "outlier_entropy", entropy_encode=True)
        loaded = CompressedVectors.load(tmp_path / "outlier_entropy")
        np.testing.assert_array_equal(loaded.indices, compressed.indices)


class TestTheoreticalSavings:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_entropy_less_than_bit_width(self, bit_width: int) -> None:
        savings = compute_theoretical_savings(dim=256, bit_width=bit_width)
        assert savings["entropy"] <= bit_width
        assert savings["entropy"] > 0

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_huffman_avg_less_than_bit_width(self, bit_width: int) -> None:
        savings = compute_theoretical_savings(dim=256, bit_width=bit_width)
        assert savings["avg_bits_huffman"] <= bit_width
        assert savings["avg_bits_huffman"] >= savings["entropy"]

    def test_positive_savings(self) -> None:
        savings = compute_theoretical_savings(dim=256, bit_width=4)
        assert savings["savings_pct"] > 0

    def test_savings_dict_keys(self) -> None:
        savings = compute_theoretical_savings(dim=256, bit_width=3)
        assert "entropy" in savings
        assert "avg_bits_huffman" in savings
        assert "savings_pct" in savings


class TestQuantizeBatchedEntropy:
    def test_quantize_batched_with_entropy(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        rng = np.random.default_rng(42)
        batches = [rng.standard_normal((25, dim)) for _ in range(4)]

        tq = TurboQuant(dim=dim, bit_width=3, mode="mse", seed=42)
        output = tmp_path / "batched_entropy"
        tq.quantize_batched(batches, output_path=output, entropy_encode=True)

        loaded = CompressedVectors.load(output)
        assert loaded.num_vectors == 100
        assert (output / "indices.huffman").exists()
        assert (output / "huffman_table.json").exists()
