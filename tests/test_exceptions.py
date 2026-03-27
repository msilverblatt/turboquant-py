"""Tests for the exception hierarchy."""

from turboquant import (
    DimensionMismatchError,
    InvalidBitWidthError,
    InvalidModeError,
    StorageError,
    TurboQuantError,
)


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_base(self) -> None:
        assert issubclass(DimensionMismatchError, TurboQuantError)
        assert issubclass(InvalidBitWidthError, TurboQuantError)
        assert issubclass(InvalidModeError, TurboQuantError)
        assert issubclass(StorageError, TurboQuantError)

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(TurboQuantError, Exception)

    def test_dimension_mismatch_message(self) -> None:
        err = DimensionMismatchError(expected=1536, got=768)
        assert "1536" in str(err)
        assert "768" in str(err)
        assert err.expected == 1536
        assert err.got == 768

    def test_invalid_bit_width_message(self) -> None:
        err = InvalidBitWidthError(bit_width=5)
        assert "5" in str(err)
        assert err.bit_width == 5

    def test_invalid_mode_message(self) -> None:
        err = InvalidModeError(mode="bad")
        assert "bad" in str(err)
        assert err.mode == "bad"
