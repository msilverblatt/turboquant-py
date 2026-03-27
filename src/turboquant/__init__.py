"""TurboQuant: Vector quantization with near-optimal distortion rates.

Implements the TurboQuant and QJL algorithms for compressing high-dimensional
vectors while preserving inner products and distances.
"""

from turboquant.exceptions import (
    DimensionMismatchError,
    InvalidBitWidthError,
    InvalidModeError,
    StorageError,
    TurboQuantError,
)

__all__ = [
    "DimensionMismatchError",
    "InvalidBitWidthError",
    "InvalidModeError",
    "StorageError",
    "TurboQuantError",
]

__version__ = "0.1.0"
