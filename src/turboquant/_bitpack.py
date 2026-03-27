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
    if bit_width == 8:
        return indices.copy()

    total_bits = n * bit_width
    n_bytes = (total_bits + 7) // 8

    # Expand each index into its individual bits (LSB first)
    bit_positions = np.arange(bit_width, dtype=np.uint8)
    bits = ((indices[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).ravel()

    # Pad to full byte boundary
    padded = np.zeros(n_bytes * 8, dtype=np.uint8)
    padded[: len(bits)] = bits

    return np.packbits(padded, bitorder="little")


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
    if bit_width == 8:
        return packed[:n_values].copy()

    # Unpack all bytes into individual bits (LSB first)
    bits = np.unpackbits(packed, bitorder="little")

    # Take only the bits we need and reshape to (n_values, bit_width)
    total_bits = n_values * bit_width
    bits = bits[:total_bits].reshape(n_values, bit_width)

    # Reconstruct values: multiply each bit by its power of 2 and sum
    powers = (1 << np.arange(bit_width, dtype=np.uint8)).astype(np.uint8)
    return (bits * powers).sum(axis=1).astype(np.uint8)
