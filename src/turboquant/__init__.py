"""TurboQuant: Vector quantization with near-optimal distortion rates.

Implements the TurboQuant and QJL algorithms for compressing high-dimensional
vectors while preserving inner products and distances.
"""

from turboquant.codebook import compute_codebook, get_codebook
from turboquant.exceptions import (
    DimensionMismatchError,
    InvalidBitWidthError,
    InvalidModeError,
    StorageError,
    TurboQuantError,
)
from turboquant.storage import CompressedVectors, CompressedStore

__all__ = [
    "CompressedStore",
    "CompressedVectors",
    "DimensionMismatchError",
    "InvalidBitWidthError",
    "InvalidModeError",
    "StorageError",
    "TurboQuantError",
    "compute_codebook",
    "get_codebook",
]

__version__ = "0.1.0"
