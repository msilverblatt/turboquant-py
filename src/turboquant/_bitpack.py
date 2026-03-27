"""Bit-packing utilities for storing quantized indices at sub-byte bit-widths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["pack_indices", "unpack_indices"]


def pack_indices(indices: NDArray[np.uint8], bit_width: int) -> NDArray[np.uint8]:
    """Pack an array of small integers into a bit-packed byte array.

    Parameters
    ----------
    indices : NDArray[np.uint8]
        Array of indices, each in range [0, 2^bit_width - 1].
    bit_width : int
        Number of bits per index (1, 2, 3, or 4).

    Returns
    -------
    NDArray[np.uint8]
        Bit-packed byte array of size ceil(len(indices) * bit_width / 8).
    """
    n = len(indices)
    total_bits = n * bit_width
    n_bytes = (total_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    if bit_width == 8:
        return indices.copy()

    bit_pos = 0
    for i in range(n):
        val = int(indices[i])
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        packed[byte_idx] |= np.uint8((val << bit_offset) & 0xFF)
        if bit_offset + bit_width > 8 and byte_idx + 1 < n_bytes:
            packed[byte_idx + 1] |= np.uint8(val >> (8 - bit_offset))

        bit_pos += bit_width

    return packed


def unpack_indices(packed: NDArray[np.uint8], bit_width: int, n_values: int) -> NDArray[np.uint8]:
    """Unpack a bit-packed byte array into an array of small integers.

    Parameters
    ----------
    packed : NDArray[np.uint8]
        Bit-packed byte array produced by ``pack_indices``.
    bit_width : int
        Number of bits per index (1, 2, 3, or 4).
    n_values : int
        Number of values to unpack.

    Returns
    -------
    NDArray[np.uint8]
        Array of indices of length n_values, each in range [0, 2^bit_width - 1].
    """
    mask = (1 << bit_width) - 1
    result = np.empty(n_values, dtype=np.uint8)

    if bit_width == 8:
        return packed[:n_values].copy()

    bit_pos = 0
    for i in range(n_values):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        val = int(packed[byte_idx]) >> bit_offset
        if bit_offset + bit_width > 8 and byte_idx + 1 < len(packed):
            val |= int(packed[byte_idx + 1]) << (8 - bit_offset)

        result[i] = val & mask
        bit_pos += bit_width

    return result
