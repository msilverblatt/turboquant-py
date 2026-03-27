"""Entropy encoding of codebook indices using Huffman coding.

Implements lossless compression of quantization indices by exploiting
the non-uniform distribution of codebook assignments. For the Beta
distribution underlying TurboQuant's scalar quantizer, this provides
a bit-width reduction of approximately 5% at b=4.

Reference: TurboQuant paper, Section 3.1 "Entropy Encoding Codebook Pointers".
"""

from __future__ import annotations

import heapq
import struct
from typing import TYPE_CHECKING

import numpy as np
from scipy import integrate

from turboquant.codebook import beta_pdf, get_codebook

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "build_huffman_table",
    "compute_symbol_probabilities",
    "compute_theoretical_savings",
    "huffman_decode",
    "huffman_encode",
]


def compute_symbol_probabilities(dim: int, bit_width: int) -> NDArray[np.float64]:
    """Compute the probability of each codebook index for the Beta distribution.

    For each partition interval [boundary_i, boundary_{i+1}], compute:
        p_i = integral of f_X(x) over the interval

    Parameters
    ----------
    dim : int
        Ambient dimension.
    bit_width : int
        Bits per quantized value.

    Returns
    -------
    NDArray[np.float64]
        Array of probabilities summing to 1, one per codebook entry.
    """
    _, boundaries = get_codebook(dim, bit_width)
    n_centroids = 1 << bit_width
    probs = np.empty(n_centroids, dtype=np.float64)

    for i in range(n_centroids):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        val, _ = integrate.quad(lambda x: beta_pdf(np.array([x]), dim)[0], lo, hi)
        probs[i] = val

    # Normalize to ensure they sum to exactly 1
    probs /= probs.sum()
    return probs


class _HuffmanNode:
    """Internal node for Huffman tree construction."""

    __slots__ = ("freq", "left", "right", "symbol")

    def __init__(
        self,
        freq: float,
        symbol: int | None = None,
        left: _HuffmanNode | None = None,
        right: _HuffmanNode | None = None,
    ) -> None:
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other: _HuffmanNode) -> bool:
        return self.freq < other.freq


def build_huffman_table(probabilities: NDArray[np.float64]) -> dict[int, str]:
    """Build a Huffman coding table from symbol probabilities.

    Parameters
    ----------
    probabilities : NDArray[np.float64]
        Probability of each symbol. Must sum to 1.

    Returns
    -------
    dict[int, str]
        Mapping from symbol index to binary code string (e.g. '010').
    """
    n = len(probabilities)
    if n == 1:
        return {0: "0"}

    # Build priority queue of leaf nodes
    heap: list[_HuffmanNode] = []
    for i in range(n):
        heapq.heappush(heap, _HuffmanNode(freq=probabilities[i], symbol=i))

    # Build tree by merging two lowest-frequency nodes
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = _HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    # Traverse tree to assign codes
    root = heap[0]
    table: dict[int, str] = {}

    def _traverse(node: _HuffmanNode, code: str) -> None:
        if node.symbol is not None:
            table[node.symbol] = code if code else "0"
            return
        if node.left is not None:
            _traverse(node.left, code + "0")
        if node.right is not None:
            _traverse(node.right, code + "1")

    _traverse(root, "")
    return table


def huffman_encode(indices: NDArray[np.uint8], table: dict[int, str]) -> bytes:
    """Encode an array of indices using Huffman coding.

    Parameters
    ----------
    indices : NDArray[np.uint8]
        Array of codebook indices to encode.
    table : dict[int, str]
        Huffman coding table mapping symbol -> binary code string.

    Returns
    -------
    bytes
        Compressed data. The first 4 bytes store the total number of bits
        as a big-endian uint32, followed by the packed bit data.
    """
    # Build the bitstring
    bits: list[str] = []
    for idx in indices.ravel():
        bits.append(table[int(idx)])
    bitstring = "".join(bits)

    total_bits = len(bitstring)

    # Pad to multiple of 8
    padding = (8 - total_bits % 8) % 8
    bitstring += "0" * padding

    # Convert to bytes
    data = bytearray()
    for i in range(0, len(bitstring), 8):
        data.append(int(bitstring[i : i + 8], 2))

    # Prepend total bit count (4 bytes, big-endian)
    header = struct.pack(">I", total_bits)
    return bytes(header) + bytes(data)


def huffman_decode(data: bytes, table: dict[int, str], n_symbols: int) -> NDArray[np.uint8]:
    """Decode Huffman-encoded bytes back to indices.

    Parameters
    ----------
    data : bytes
        Compressed data from ``huffman_encode``.
    table : dict[int, str]
        Huffman coding table (same one used for encoding).
    n_symbols : int
        Number of symbols to decode.

    Returns
    -------
    NDArray[np.uint8]
        Array of decoded indices.
    """
    # Read total bit count from header
    total_bits = struct.unpack(">I", data[:4])[0]
    payload = data[4:]

    # Build reverse lookup: code string -> symbol
    reverse_table = {code: sym for sym, code in table.items()}

    # Convert bytes to bitstring
    bitstring = "".join(f"{byte:08b}" for byte in payload)
    bitstring = bitstring[:total_bits]

    result = np.empty(n_symbols, dtype=np.uint8)
    pos = 0
    for i in range(n_symbols):
        code = ""
        while pos < len(bitstring):
            code += bitstring[pos]
            pos += 1
            if code in reverse_table:
                result[i] = reverse_table[code]
                break
    return result


def compute_theoretical_savings(dim: int, bit_width: int) -> dict[str, float]:
    """Compute theoretical entropy and savings for a given configuration.

    Parameters
    ----------
    dim : int
        Ambient dimension.
    bit_width : int
        Bits per quantized value.

    Returns
    -------
    dict[str, float]
        Dictionary with:
        - entropy: Shannon entropy of the distribution (bits)
        - avg_bits_huffman: average bits per symbol with Huffman coding
        - savings_pct: percentage reduction vs fixed-width encoding
    """
    probs = compute_symbol_probabilities(dim, bit_width)

    # Shannon entropy: H = -sum(p * log2(p))
    nonzero = probs > 0
    entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))

    # Build Huffman table and compute average code length
    table = build_huffman_table(probs)
    avg_bits = sum(probs[sym] * len(code) for sym, code in table.items())

    savings_pct = (1.0 - avg_bits / bit_width) * 100.0

    return {
        "entropy": float(entropy),
        "avg_bits_huffman": float(avg_bits),
        "savings_pct": float(savings_pct),
    }


def table_to_serializable(table: dict[int, str]) -> dict[str, str]:
    """Convert Huffman table to JSON-serializable format (string keys)."""
    return {str(k): v for k, v in table.items()}


def table_from_serializable(data: dict[str, str]) -> dict[int, str]:
    """Restore Huffman table from JSON-deserialized format."""
    return {int(k): v for k, v in data.items()}
