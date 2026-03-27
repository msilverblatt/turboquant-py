"""Tests for bit-packing utilities."""

import numpy as np
import pytest

from turboquant._bitpack import pack_indices, unpack_indices


class TestBitPacking:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_round_trip(self, bit_width: int) -> None:
        rng = np.random.default_rng(42)
        n_values = 1000
        max_val = (1 << bit_width) - 1
        indices = rng.integers(0, max_val + 1, size=n_values, dtype=np.uint8)
        packed = pack_indices(indices, bit_width)
        unpacked = unpack_indices(packed, bit_width, n_values)
        np.testing.assert_array_equal(indices, unpacked)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_compression_ratio(self, bit_width: int) -> None:
        n_values = 1024
        indices = np.zeros(n_values, dtype=np.uint8)
        packed = pack_indices(indices, bit_width)
        expected_bytes = (n_values * bit_width + 7) // 8
        assert packed.nbytes == expected_bytes

    def test_1bit_packing_specific(self) -> None:
        indices = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        packed = pack_indices(indices, bit_width=1)
        assert packed.nbytes == 1
        unpacked = unpack_indices(packed, bit_width=1, n_values=8)
        np.testing.assert_array_equal(indices, unpacked)

    def test_4bit_packing_specific(self) -> None:
        indices = np.array([0, 15, 7, 8], dtype=np.uint8)
        packed = pack_indices(indices, bit_width=4)
        assert packed.nbytes == 2
        unpacked = unpack_indices(packed, bit_width=4, n_values=4)
        np.testing.assert_array_equal(indices, unpacked)

    def test_non_aligned_length(self) -> None:
        rng = np.random.default_rng(42)
        for bit_width in [1, 2, 3, 4]:
            for n_values in [7, 13, 100, 255]:
                max_val = (1 << bit_width) - 1
                indices = rng.integers(0, max_val + 1, size=n_values, dtype=np.uint8)
                packed = pack_indices(indices, bit_width)
                unpacked = unpack_indices(packed, bit_width, n_values)
                np.testing.assert_array_equal(indices, unpacked)

    def test_large_array(self) -> None:
        rng = np.random.default_rng(42)
        n_values = 100_000
        indices = rng.integers(0, 8, size=n_values, dtype=np.uint8)
        packed = pack_indices(indices, bit_width=3)
        unpacked = unpack_indices(packed, bit_width=3, n_values=n_values)
        np.testing.assert_array_equal(indices, unpacked)
