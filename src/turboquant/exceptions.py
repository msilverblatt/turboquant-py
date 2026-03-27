"""Custom exception hierarchy for TurboQuant."""

__all__ = [
    "DimensionMismatchError",
    "InvalidBitWidthError",
    "InvalidModeError",
    "StorageError",
    "TurboQuantError",
]


class TurboQuantError(Exception):
    """Base exception for all TurboQuant errors."""


class DimensionMismatchError(TurboQuantError):
    """Raised when vector dimensions do not match the quantizer configuration."""

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(
            f"Dimension mismatch: quantizer configured for dim={expected}, got dim={got}"
        )


class InvalidBitWidthError(TurboQuantError):
    """Raised when an unsupported bit-width is requested."""

    def __init__(self, bit_width: int, valid_range: tuple[int, int] = (1, 4)) -> None:
        self.bit_width = bit_width
        self.valid_range = valid_range
        super().__init__(
            f"Invalid bit_width={bit_width}. Must be in range [{valid_range[0]}, {valid_range[1]}]"
        )


class InvalidModeError(TurboQuantError):
    """Raised when an unsupported quantization mode is requested."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        super().__init__(f"Invalid mode='{mode}'. Must be 'mse' or 'inner_product'")


class StorageError(TurboQuantError):
    """Raised when storage operations fail (load, save, corrupt data)."""
