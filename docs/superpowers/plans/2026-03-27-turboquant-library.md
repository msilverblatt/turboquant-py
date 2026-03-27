# TurboQuant Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python library implementing TurboQuant and QJL vector quantization algorithms with NumPy-first implementation and optional PyTorch acceleration.

**Architecture:** NumPy-first with optional PyTorch dispatch. Two quantizer classes (QJL, TurboQuant) sharing a CompressedVectors container and CompressedStore persistence layer. Precomputed Lloyd-Max codebooks shipped as NumPy arrays.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pytest, ruff

**Spec:** `docs/superpowers/specs/2026-03-27-turboquant-library-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package metadata, dependencies, ruff config |
| `src/turboquant/__init__.py` | Public API re-exports, `__all__` |
| `src/turboquant/py.typed` | PEP 561 marker (empty file) |
| `src/turboquant/exceptions.py` | Custom exception hierarchy |
| `src/turboquant/_accel.py` | NumPy/PyTorch dispatch for matrix ops |
| `src/turboquant/_bitpack.py` | Bit-packing utilities for quantized indices |
| `src/turboquant/codebook.py` | Lloyd-Max codebook computation and loading |
| `src/turboquant/qjl.py` | QJL 1-bit quantizer |
| `src/turboquant/turboquant.py` | TurboQuant MSE and inner-product quantizers |
| `src/turboquant/storage.py` | CompressedVectors container, CompressedStore persistence |
| `src/turboquant/codebooks/` | Directory for precomputed `.npy` codebook arrays |
| `tests/conftest.py` | Shared fixtures (random vectors, dimensions list) |
| `tests/test_exceptions.py` | Exception hierarchy tests |
| `tests/test_accel.py` | Acceleration dispatch tests |
| `tests/test_bitpack.py` | Bit-packing round-trip tests |
| `tests/test_codebook.py` | Codebook computation and loading tests |
| `tests/test_qjl.py` | QJL quantizer tests |
| `tests/test_turboquant.py` | TurboQuant quantizer tests |
| `tests/test_storage.py` | Storage round-trip and mmap tests |
| `tests/test_integration.py` | End-to-end pipeline tests |
| `benchmarks/bench_all.py` | Performance benchmarks |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/turboquant/__init__.py`
- Create: `src/turboquant/py.typed`
- Create: `src/turboquant/exceptions.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_exceptions.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
name = "turboquant"
version = "0.1.0"
description = "Vector quantization library implementing TurboQuant and QJL algorithms"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "ruff>=0.4",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
target-version = "py310"
line-length = 99

[tool.ruff.lint]
select = [
    "E", "F", "W",   # pyflakes + pycodestyle
    "I",              # isort
    "UP",             # pyupgrade
    "B",              # bugbear
    "SIM",            # simplify
    "TCH",            # type-checking imports
    "RUF",            # ruff-specific
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create src/turboquant/py.typed**

Empty file (PEP 561 marker):
```
```

- [ ] **Step 3: Create src/turboquant/exceptions.py**

```python
"""Custom exception hierarchy for TurboQuant."""

__all__ = [
    "TurboQuantError",
    "DimensionMismatchError",
    "InvalidBitWidthError",
    "InvalidModeError",
    "StorageError",
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
        super().__init__(
            f"Invalid mode='{mode}'. Must be 'mse' or 'inner_product'"
        )


class StorageError(TurboQuantError):
    """Raised when storage operations fail (load, save, corrupt data)."""
```

- [ ] **Step 4: Create src/turboquant/__init__.py**

```python
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
    # Exceptions
    "TurboQuantError",
    "DimensionMismatchError",
    "InvalidBitWidthError",
    "InvalidModeError",
    "StorageError",
]

__version__ = "0.1.0"
```

- [ ] **Step 5: Create tests/__init__.py and tests/conftest.py**

`tests/__init__.py` — empty file.

`tests/conftest.py`:
```python
"""Shared test fixtures for TurboQuant tests."""

import numpy as np
import pytest

DIMS = [384, 512, 768, 1024, 1536, 2048, 3072]
BIT_WIDTHS = [1, 2, 3, 4]


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for tests."""
    return np.random.default_rng(42)


@pytest.fixture(params=[64, 384, 1536])
def dim(request: pytest.FixtureRequest) -> int:
    """Common embedding dimensions for parameterized tests.

    Uses 64 for fast tests, 384 and 1536 for realistic dimensions.
    """
    return request.param


@pytest.fixture
def random_vectors(rng: np.random.Generator) -> np.ndarray:
    """100 random unit vectors in 256 dimensions."""
    vectors = rng.standard_normal((100, 256))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def random_query(rng: np.random.Generator) -> np.ndarray:
    """Single random unit vector in 256 dimensions."""
    q = rng.standard_normal(256)
    return q / np.linalg.norm(q)
```

- [ ] **Step 6: Write exception tests**

`tests/test_exceptions.py`:
```python
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
```

- [ ] **Step 7: Install package in dev mode and run tests**

Run:
```bash
cd /Users/msilverblatt/Projects/piedpiper
python -m pip install -e ".[dev]"
pytest tests/test_exceptions.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 8: Run ruff**

Run:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
```
Expected: No errors.

- [ ] **Step 9: Initialize git repo and commit**

```bash
git init
echo "__pycache__/" > .gitignore
echo "*.egg-info/" >> .gitignore
echo ".eggs/" >> .gitignore
echo "dist/" >> .gitignore
echo "build/" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo ".ruff_cache/" >> .gitignore
echo "*.pyc" >> .gitignore
git add pyproject.toml .gitignore src/ tests/ docs/
git commit -m "feat: project scaffolding with exceptions, pyproject.toml, and test infrastructure"
```

---

### Task 2: Acceleration Dispatch Layer

**Files:**
- Create: `src/turboquant/_accel.py`
- Create: `tests/test_accel.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_accel.py`:
```python
"""Tests for NumPy/PyTorch acceleration dispatch."""

import numpy as np
import pytest

from turboquant._accel import (
    generate_orthogonal_matrix,
    generate_projection_matrix,
    matmul,
    batch_inner_product,
    has_torch,
)


class TestGenerateOrthogonalMatrix:
    def test_returns_square_matrix(self) -> None:
        mat = generate_orthogonal_matrix(64, seed=42)
        assert mat.shape == (64, 64)

    def test_is_orthogonal(self) -> None:
        mat = generate_orthogonal_matrix(64, seed=42)
        product = mat @ mat.T
        np.testing.assert_allclose(product, np.eye(64), atol=1e-10)

    def test_deterministic_with_seed(self) -> None:
        a = generate_orthogonal_matrix(64, seed=42)
        b = generate_orthogonal_matrix(64, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = generate_orthogonal_matrix(64, seed=42)
        b = generate_orthogonal_matrix(64, seed=99)
        assert not np.allclose(a, b)


class TestGenerateProjectionMatrix:
    def test_shape(self) -> None:
        mat = generate_projection_matrix(32, 64, seed=42)
        assert mat.shape == (32, 64)

    def test_rows_are_orthonormal(self) -> None:
        mat = generate_projection_matrix(32, 64, seed=42)
        product = mat @ mat.T
        np.testing.assert_allclose(product, np.eye(32), atol=1e-10)

    def test_deterministic_with_seed(self) -> None:
        a = generate_projection_matrix(32, 64, seed=42)
        b = generate_projection_matrix(32, 64, seed=42)
        np.testing.assert_array_equal(a, b)


class TestMatmul:
    def test_matrix_vector(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 64))
        x = rng.standard_normal(64)
        result = matmul(A, x)
        np.testing.assert_allclose(result, A @ x)

    def test_matrix_matrix(self) -> None:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((32, 64))
        B = rng.standard_normal((64, 10))
        result = matmul(A, B)
        np.testing.assert_allclose(result, A @ B)


class TestBatchInnerProduct:
    def test_single_query_multiple_vectors(self) -> None:
        rng = np.random.default_rng(42)
        query = rng.standard_normal(64)
        vectors = rng.standard_normal((100, 64))
        result = batch_inner_product(query, vectors)
        expected = vectors @ query
        np.testing.assert_allclose(result, expected)

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        query = rng.standard_normal(64)
        vectors = rng.standard_normal((50, 64))
        result = batch_inner_product(query, vectors)
        assert result.shape == (50,)


class TestHasTorch:
    def test_returns_bool(self) -> None:
        assert isinstance(has_torch(), bool)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_accel.py -v`
Expected: FAIL with `ImportError` — `_accel` module does not exist.

- [ ] **Step 3: Write the implementation**

`src/turboquant/_accel.py`:
```python
"""Acceleration dispatch layer for NumPy/PyTorch.

All matrix operations go through this module. If PyTorch is installed,
operations dispatch to torch tensors for potential speedup. Otherwise,
falls back to NumPy.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "generate_orthogonal_matrix",
    "generate_projection_matrix",
    "matmul",
    "batch_inner_product",
    "has_torch",
]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def has_torch() -> bool:
    """Check if PyTorch is available at runtime."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def generate_orthogonal_matrix(dim: int, seed: int | None = None) -> NDArray[np.float64]:
    """Generate a random orthogonal matrix via QR decomposition.

    Parameters
    ----------
    dim : int
        Matrix dimension (produces dim x dim orthogonal matrix).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Orthogonal matrix of shape (dim, dim) where Q @ Q.T == I.
    """
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(gaussian)
    # Ensure uniqueness: multiply by sign of diagonal of R
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs[np.newaxis, :]
    return q


def generate_projection_matrix(
    rows: int, cols: int, seed: int | None = None
) -> NDArray[np.float64]:
    """Generate a random projection matrix with orthonormal rows.

    Parameters
    ----------
    rows : int
        Number of rows (projection dimension). Must be <= cols.
    cols : int
        Number of columns (input dimension).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Matrix of shape (rows, cols) with orthonormal rows.
    """
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((cols, cols))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs[np.newaxis, :]
    return q[:rows, :].copy()


def matmul(a: NDArray, b: NDArray) -> NDArray:
    """Matrix multiplication with optional PyTorch dispatch.

    Parameters
    ----------
    a : NDArray
        Left operand.
    b : NDArray
        Right operand.

    Returns
    -------
    NDArray
        Result of a @ b.
    """
    if has_torch():
        import torch

        device = "cpu"
        ta = torch.from_numpy(np.ascontiguousarray(a)).to(device)
        tb = torch.from_numpy(np.ascontiguousarray(b)).to(device)
        result = torch.matmul(ta, tb)
        return result.numpy()
    return a @ b


def batch_inner_product(query: NDArray, vectors: NDArray) -> NDArray[np.float64]:
    """Compute inner products between a query and a batch of vectors.

    Parameters
    ----------
    query : NDArray
        Query vector of shape (dim,).
    vectors : NDArray
        Matrix of shape (n, dim).

    Returns
    -------
    NDArray[np.float64]
        Inner products of shape (n,).
    """
    if has_torch():
        import torch

        device = "cpu"
        tq = torch.from_numpy(np.ascontiguousarray(query)).to(device)
        tv = torch.from_numpy(np.ascontiguousarray(vectors)).to(device)
        result = torch.mv(tv, tq)
        return result.numpy()
    return vectors @ query
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accel.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/turboquant/_accel.py tests/test_accel.py
git commit -m "feat: add acceleration dispatch layer with NumPy/PyTorch support"
```

---

### Task 3: Bit-Packing Utilities

**Files:**
- Create: `src/turboquant/_bitpack.py`
- Create: `tests/test_bitpack.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_bitpack.py`:
```python
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
        # packed should use ceil(n_values * bit_width / 8) bytes
        expected_bytes = (n_values * bit_width + 7) // 8
        assert packed.nbytes == expected_bytes

    def test_1bit_packing_specific(self) -> None:
        # 8 values of alternating 0 and 1
        indices = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        packed = pack_indices(indices, bit_width=1)
        assert packed.nbytes == 1
        unpacked = unpack_indices(packed, bit_width=1, n_values=8)
        np.testing.assert_array_equal(indices, unpacked)

    def test_4bit_packing_specific(self) -> None:
        indices = np.array([0, 15, 7, 8], dtype=np.uint8)
        packed = pack_indices(indices, bit_width=4)
        assert packed.nbytes == 2  # 4 values * 4 bits = 16 bits = 2 bytes
        unpacked = unpack_indices(packed, bit_width=4, n_values=4)
        np.testing.assert_array_equal(indices, unpacked)

    def test_non_aligned_length(self) -> None:
        """Test with a number of values that doesn't align to byte boundaries."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_bitpack.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

`src/turboquant/_bitpack.py`:
```python
"""Bit-packing utilities for storing quantized indices at sub-byte bit-widths."""

from __future__ import annotations

import numpy as np
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

        # Value may span two bytes
        packed[byte_idx] |= np.uint8((val << bit_offset) & 0xFF)
        if bit_offset + bit_width > 8 and byte_idx + 1 < n_bytes:
            packed[byte_idx + 1] |= np.uint8(val >> (8 - bit_offset))

        bit_pos += bit_width

    return packed


def unpack_indices(
    packed: NDArray[np.uint8], bit_width: int, n_values: int
) -> NDArray[np.uint8]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bitpack.py -v`
Expected: All 7 tests PASS (some are parameterized, so more test cases).

- [ ] **Step 5: Commit**

```bash
git add src/turboquant/_bitpack.py tests/test_bitpack.py
git commit -m "feat: add bit-packing utilities for sub-byte index storage"
```

---

### Task 4: Lloyd-Max Codebook Computation

**Files:**
- Create: `src/turboquant/codebook.py`
- Create: `src/turboquant/codebooks/` (directory)
- Create: `tests/test_codebook.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_codebook.py`:
```python
"""Tests for Lloyd-Max codebook computation and loading."""

import numpy as np
import pytest

from turboquant.codebook import (
    beta_pdf,
    compute_codebook,
    get_codebook,
    quantize_scalar,
    dequantize_scalar,
)


class TestBetaPdf:
    def test_integrates_to_one(self) -> None:
        """The Beta distribution PDF should integrate to ~1 over [-1, 1]."""
        for dim in [64, 256, 1024]:
            x = np.linspace(-0.999, 0.999, 10000)
            pdf_vals = beta_pdf(x, dim)
            integral = np.trapz(pdf_vals, x)
            np.testing.assert_allclose(integral, 1.0, atol=0.01)

    def test_symmetric(self) -> None:
        x = np.linspace(0.01, 0.99, 100)
        for dim in [64, 256, 1024]:
            np.testing.assert_allclose(beta_pdf(x, dim), beta_pdf(-x, dim), atol=1e-10)

    def test_converges_to_gaussian(self) -> None:
        """In high dimensions, the Beta distribution converges to N(0, 1/d)."""
        dim = 4096
        x = np.linspace(-0.1, 0.1, 1000)
        beta_vals = beta_pdf(x, dim)
        sigma = 1.0 / np.sqrt(dim)
        gaussian_vals = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)
        # Normalize both for comparison
        beta_vals = beta_vals / np.trapz(beta_vals, x)
        gaussian_vals = gaussian_vals / np.trapz(gaussian_vals, x)
        np.testing.assert_allclose(beta_vals, gaussian_vals, atol=0.05)


class TestComputeCodebook:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, bit_width: int) -> None:
        centroids, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        assert len(centroids) == 2**bit_width
        assert len(boundaries) == 2**bit_width + 1

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_centroids_sorted(self, bit_width: int) -> None:
        centroids, _ = compute_codebook(dim=256, bit_width=bit_width)
        assert np.all(np.diff(centroids) > 0)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_boundaries_sorted(self, bit_width: int) -> None:
        _, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        assert np.all(np.diff(boundaries) > 0)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_boundaries_span_range(self, bit_width: int) -> None:
        _, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        assert boundaries[0] == -1.0
        assert boundaries[-1] == 1.0

    def test_1bit_centroids_symmetric(self) -> None:
        """For 1-bit, centroids should be symmetric around 0."""
        centroids, _ = compute_codebook(dim=256, bit_width=1)
        np.testing.assert_allclose(centroids[0], -centroids[1], atol=1e-6)

    def test_centroids_within_boundaries(self) -> None:
        for bit_width in [1, 2, 3, 4]:
            centroids, boundaries = compute_codebook(dim=256, bit_width=bit_width)
            for i, c in enumerate(centroids):
                assert boundaries[i] <= c <= boundaries[i + 1]


class TestGetCodebook:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_returns_cached_codebook(self, bit_width: int) -> None:
        c1, b1 = get_codebook(dim=256, bit_width=bit_width)
        c2, b2 = get_codebook(dim=256, bit_width=bit_width)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(b1, b2)


class TestQuantizeDequantize:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_round_trip_reduces_error_with_bit_width(self, bit_width: int) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(1000) / np.sqrt(256)  # scale like Beta dist
        values = np.clip(values, -0.99, 0.99)
        centroids, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        indices = quantize_scalar(values, boundaries)
        reconstructed = dequantize_scalar(indices, centroids)
        mse = np.mean((values - reconstructed) ** 2)
        # Just check it's finite and positive
        assert 0 < mse < 1.0

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_indices_in_valid_range(self, bit_width: int) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(1000) / np.sqrt(256)
        values = np.clip(values, -0.99, 0.99)
        _, boundaries = compute_codebook(dim=256, bit_width=bit_width)
        indices = quantize_scalar(values, boundaries)
        assert np.all(indices >= 0)
        assert np.all(indices < 2**bit_width)

    def test_mse_decreases_with_bit_width(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.standard_normal(10000) / np.sqrt(256)
        values = np.clip(values, -0.99, 0.99)
        mses = []
        for bw in [1, 2, 3, 4]:
            centroids, boundaries = compute_codebook(dim=256, bit_width=bw)
            indices = quantize_scalar(values, boundaries)
            reconstructed = dequantize_scalar(indices, centroids)
            mses.append(np.mean((values - reconstructed) ** 2))
        # Each bit width should give strictly lower MSE
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], f"MSE did not decrease: bw={i+1}={mses[i]}, bw={i+2}={mses[i+1]}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_codebook.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

`src/turboquant/codebook.py`:
```python
"""Lloyd-Max codebook computation for the Beta distribution.

Computes optimal scalar quantizers for coordinates of randomly rotated
unit vectors. In high dimensions, each coordinate follows a Beta distribution
that converges to Gaussian N(0, 1/d).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

__all__ = [
    "beta_pdf",
    "compute_codebook",
    "get_codebook",
    "quantize_scalar",
    "dequantize_scalar",
]

logger = logging.getLogger(__name__)


def beta_pdf(x: NDArray[np.float64], dim: int) -> NDArray[np.float64]:
    """Evaluate the PDF of a coordinate of a uniformly random unit vector.

    For a random point on S^{d-1}, each coordinate follows:
        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    Parameters
    ----------
    x : NDArray[np.float64]
        Points in [-1, 1] at which to evaluate the PDF.
    dim : int
        Ambient dimension d.

    Returns
    -------
    NDArray[np.float64]
        PDF values at each point.
    """
    x = np.asarray(x, dtype=np.float64)
    # Use log-gamma for numerical stability
    log_coeff = gammaln(dim / 2) - 0.5 * np.log(np.pi) - gammaln((dim - 1) / 2)
    exponent = (dim - 3) / 2

    result = np.zeros_like(x)
    valid = np.abs(x) < 1.0
    result[valid] = np.exp(log_coeff + exponent * np.log(1.0 - x[valid] ** 2))
    return result


def compute_codebook(
    dim: int, bit_width: int, max_iter: int = 200, tol: float = 1e-10
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute optimal Lloyd-Max codebook for the Beta distribution.

    Solves the 1D continuous k-means problem by iterating:
    1. Update boundaries to midpoints of consecutive centroids.
    2. Update centroids to conditional expectations within each partition.

    Parameters
    ----------
    dim : int
        Ambient dimension (determines the Beta distribution shape).
    bit_width : int
        Number of bits per quantized value. Produces 2^bit_width centroids.
    max_iter : int
        Maximum Lloyd-Max iterations.
    tol : float
        Convergence tolerance on centroid movement.

    Returns
    -------
    centroids : NDArray[np.float64]
        Sorted array of 2^bit_width centroid values.
    boundaries : NDArray[np.float64]
        Sorted array of 2^bit_width + 1 boundary values, starting at -1.0 and ending at 1.0.
    """
    n_centroids = 1 << bit_width
    n_points = 10000  # quadrature resolution

    # Initialize centroids uniformly in [-1, 1]
    centroids = np.linspace(-1 + 1 / n_centroids, 1 - 1 / n_centroids, n_centroids)

    x_grid = np.linspace(-0.9999, 0.9999, n_points)
    pdf_vals = beta_pdf(x_grid, dim)

    for iteration in range(max_iter):
        # Update boundaries: midpoints of consecutive centroids
        boundaries = np.empty(n_centroids + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_centroids - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

        # Update centroids: conditional expectation within each partition
        new_centroids = np.empty(n_centroids)
        for i in range(n_centroids):
            mask = (x_grid >= boundaries[i]) & (x_grid < boundaries[i + 1])
            if i == n_centroids - 1:
                mask = (x_grid >= boundaries[i]) & (x_grid <= boundaries[i + 1])
            if np.any(mask):
                weights = pdf_vals[mask]
                total_weight = np.trapz(weights, x_grid[mask])
                if total_weight > 0:
                    new_centroids[i] = np.trapz(x_grid[mask] * weights, x_grid[mask]) / total_weight
                else:
                    new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        max_shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if max_shift < tol:
            logger.debug(
                "Lloyd-Max converged after %d iterations (max_shift=%.2e)",
                iteration + 1,
                max_shift,
            )
            break

    # Final boundaries
    boundaries = np.empty(n_centroids + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(n_centroids - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

    return centroids, boundaries


@lru_cache(maxsize=32)
def get_codebook(
    dim: int, bit_width: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get a codebook, computing it if not cached.

    Parameters
    ----------
    dim : int
        Ambient dimension.
    bit_width : int
        Bits per quantized value.

    Returns
    -------
    centroids : NDArray[np.float64]
        Sorted centroid values.
    boundaries : NDArray[np.float64]
        Sorted boundary values.
    """
    return compute_codebook(dim, bit_width)


def quantize_scalar(
    values: NDArray[np.float64], boundaries: NDArray[np.float64]
) -> NDArray[np.uint8]:
    """Quantize scalar values using precomputed boundaries.

    Parameters
    ----------
    values : NDArray[np.float64]
        Values to quantize.
    boundaries : NDArray[np.float64]
        Sorted boundary array of length n_centroids + 1.

    Returns
    -------
    NDArray[np.uint8]
        Indices into the centroid array.
    """
    # np.searchsorted gives the index of the right boundary
    # Subtract 1 to get the centroid index, clip to valid range
    indices = np.searchsorted(boundaries, values, side="right") - 1
    n_centroids = len(boundaries) - 1
    return np.clip(indices, 0, n_centroids - 1).astype(np.uint8)


def dequantize_scalar(
    indices: NDArray[np.uint8], centroids: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Dequantize indices back to scalar values using centroids.

    Parameters
    ----------
    indices : NDArray[np.uint8]
        Centroid indices.
    centroids : NDArray[np.float64]
        Centroid values.

    Returns
    -------
    NDArray[np.float64]
        Reconstructed values (centroid lookups).
    """
    return centroids[indices]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_codebook.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Update __init__.py exports**

Add to `src/turboquant/__init__.py`:
```python
from turboquant.codebook import compute_codebook, get_codebook
```

Update `__all__` to include `"compute_codebook"`, `"get_codebook"`.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant/codebook.py src/turboquant/codebooks/ src/turboquant/__init__.py tests/test_codebook.py
git commit -m "feat: add Lloyd-Max codebook computation for Beta distribution"
```

---

### Task 5: CompressedVectors Container

**Files:**
- Create: `src/turboquant/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_storage.py`:
```python
"""Tests for CompressedVectors container and CompressedStore."""

import numpy as np
import pytest
from pathlib import Path

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
        """Test that additional arrays (rotation, projection, etc.) survive save/load."""
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
        np.testing.assert_array_equal(loaded.extra_arrays["qjl_signs"], cv.extra_arrays["qjl_signs"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

`src/turboquant/storage.py`:
```python
"""Compressed vector storage: in-memory container and on-disk persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from turboquant.exceptions import StorageError

__all__ = ["CompressedVectors", "CompressedStore"]

logger = logging.getLogger(__name__)


class CompressedVectors:
    """In-memory container for quantized vectors.

    Parameters
    ----------
    indices : NDArray[np.uint8]
        Quantization indices of shape (n, dim). Each value in [0, 2^bit_width - 1].
    norms : NDArray[np.float64]
        Per-vector L2 norms of shape (n,).
    dim : int
        Original vector dimensionality.
    bit_width : int
        Bits per quantized coordinate.
    metadata : dict[str, Any]
        Arbitrary metadata (mode, seed, outlier config, etc.).
    extra_arrays : dict[str, NDArray] or None
        Additional named arrays (rotation matrix, QJL signs, residual norms, etc.).
    """

    def __init__(
        self,
        indices: NDArray[np.uint8],
        norms: NDArray[np.float64],
        dim: int,
        bit_width: int,
        metadata: dict[str, Any] | None = None,
        extra_arrays: dict[str, NDArray] | None = None,
    ) -> None:
        self.indices = indices
        self.norms = norms
        self.dim = dim
        self.bit_width = bit_width
        self.metadata: dict[str, Any] = metadata or {}
        self.extra_arrays: dict[str, NDArray] = extra_arrays or {}

    @property
    def num_vectors(self) -> int:
        """Number of compressed vectors."""
        return len(self.norms)

    def __getitem__(self, key: slice) -> CompressedVectors:
        """Slice the compressed vectors."""
        sliced_extras = {k: v[key] if v.shape[0] == self.num_vectors else v
                         for k, v in self.extra_arrays.items()}
        return CompressedVectors(
            indices=self.indices[key],
            norms=self.norms[key],
            dim=self.dim,
            bit_width=self.bit_width,
            metadata=self.metadata.copy(),
            extra_arrays=sliced_extras,
        )

    @classmethod
    def concatenate(cls, parts: list[CompressedVectors]) -> CompressedVectors:
        """Concatenate multiple CompressedVectors into one.

        Parameters
        ----------
        parts : list[CompressedVectors]
            Parts to concatenate. Must have matching dim and bit_width.

        Returns
        -------
        CompressedVectors
            Merged container.

        Raises
        ------
        ValueError
            If dimensions or bit-widths do not match.
        """
        if not parts:
            raise ValueError("Cannot concatenate empty list")
        ref = parts[0]
        for i, p in enumerate(parts[1:], 1):
            if p.dim != ref.dim:
                raise ValueError(
                    f"Cannot concatenate: dimension mismatch at index {i} "
                    f"(expected {ref.dim}, got {p.dim})"
                )
            if p.bit_width != ref.bit_width:
                raise ValueError(
                    f"Cannot concatenate: bit_width mismatch at index {i} "
                    f"(expected {ref.bit_width}, got {p.bit_width})"
                )
        return cls(
            indices=np.concatenate([p.indices for p in parts], axis=0),
            norms=np.concatenate([p.norms for p in parts]),
            dim=ref.dim,
            bit_width=ref.bit_width,
            metadata=ref.metadata.copy(),
            extra_arrays={
                k: np.concatenate([p.extra_arrays[k] for p in parts if k in p.extra_arrays], axis=0)
                for k in ref.extra_arrays
                if all(k in p.extra_arrays for p in parts)
                and ref.extra_arrays[k].shape[0] == ref.num_vectors
            },
        )

    def save(self, path: str | Path) -> None:
        """Save to a directory on disk.

        Parameters
        ----------
        path : str or Path
            Directory path. Created if it does not exist.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "dim": self.dim,
            "bit_width": self.bit_width,
            "num_vectors": self.num_vectors,
            "extra_array_names": list(self.extra_arrays.keys()),
            **self.metadata,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        np.save(path / "indices.npy", self.indices)
        np.save(path / "norms.npy", self.norms)

        for name, arr in self.extra_arrays.items():
            np.save(path / f"{name}.npy", arr)

        logger.info("Saved %d compressed vectors to %s", self.num_vectors, path)

    @classmethod
    def load(cls, path: str | Path, mmap_mode: str | None = None) -> CompressedVectors:
        """Load from a directory on disk.

        Parameters
        ----------
        path : str or Path
            Directory path containing saved data.
        mmap_mode : str or None
            If set (e.g., 'r'), memory-map the arrays instead of loading into RAM.

        Returns
        -------
        CompressedVectors
            Loaded container.

        Raises
        ------
        StorageError
            If the path does not exist or is missing required files.
        """
        path = Path(path)
        if not path.exists():
            raise StorageError(f"Path does not exist: {path}")

        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise StorageError(f"Missing meta.json in {path}")

        with open(meta_path) as f:
            meta = json.load(f)

        dim = meta.pop("dim")
        bit_width = meta.pop("bit_width")
        num_vectors = meta.pop("num_vectors")
        extra_names = meta.pop("extra_array_names", [])

        indices = np.load(path / "indices.npy", mmap_mode=mmap_mode)
        norms = np.load(path / "norms.npy", mmap_mode=mmap_mode)

        extra_arrays = {}
        for name in extra_names:
            arr_path = path / f"{name}.npy"
            if arr_path.exists():
                extra_arrays[name] = np.load(arr_path, mmap_mode=mmap_mode)

        return cls(
            indices=indices,
            norms=norms,
            dim=dim,
            bit_width=bit_width,
            metadata=meta,
            extra_arrays=extra_arrays,
        )


class CompressedStore:
    """On-disk compressed vector store with memory-mapped search.

    Parameters
    ----------
    path : str or Path
        Path to a saved CompressedVectors directory.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._vectors = CompressedVectors.load(self._path, mmap_mode="r")

    @classmethod
    def load(cls, path: str | Path) -> CompressedStore:
        """Load a compressed store from disk.

        Parameters
        ----------
        path : str or Path
            Path to the store directory.

        Returns
        -------
        CompressedStore
            Loaded store with memory-mapped arrays.
        """
        return cls(path)

    @property
    def dim(self) -> int:
        """Original vector dimensionality."""
        return self._vectors.dim

    @property
    def num_vectors(self) -> int:
        """Number of stored vectors."""
        return self._vectors.num_vectors

    @property
    def bit_width(self) -> int:
        """Quantization bit-width."""
        return self._vectors.bit_width

    @property
    def mode(self) -> str:
        """Quantization mode."""
        return self._vectors.metadata.get("mode", "unknown")

    @property
    def vectors(self) -> CompressedVectors:
        """Access the underlying CompressedVectors."""
        return self._vectors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Update __init__.py exports**

Add to `src/turboquant/__init__.py`:
```python
from turboquant.storage import CompressedStore, CompressedVectors
```

Update `__all__` to include `"CompressedVectors"`, `"CompressedStore"`.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant/storage.py src/turboquant/__init__.py tests/test_storage.py
git commit -m "feat: add CompressedVectors container and CompressedStore persistence"
```

---

### Task 6: QJL Quantizer

**Files:**
- Create: `src/turboquant/qjl.py`
- Create: `tests/test_qjl.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_qjl.py`:
```python
"""Tests for QJL 1-bit quantizer."""

import numpy as np
import pytest

from turboquant.qjl import QJL
from turboquant.exceptions import DimensionMismatchError


class TestQJLConstruction:
    def test_default_projection_dim(self) -> None:
        qjl = QJL(dim=256)
        assert qjl.projection_dim == 256

    def test_custom_projection_dim(self) -> None:
        qjl = QJL(dim=256, projection_dim=128)
        assert qjl.projection_dim == 128

    def test_deterministic_with_seed(self) -> None:
        q1 = QJL(dim=64, seed=42)
        q2 = QJL(dim=64, seed=42)
        np.testing.assert_array_equal(q1._projection, q2._projection)


class TestQJLQuantize:
    def test_output_type(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        assert compressed.num_vectors == 10
        assert compressed.dim == 64
        assert compressed.bit_width == 1

    def test_dimension_mismatch_raises(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 128))
        with pytest.raises(DimensionMismatchError):
            qjl.quantize(vectors)

    def test_signs_are_binary(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        signs = compressed.extra_arrays["signs"]
        assert set(np.unique(signs)).issubset({-1, 1})

    def test_norms_are_positive(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        assert np.all(compressed.norms > 0)


class TestQJLInnerProduct:
    def test_unbiased_estimator(self) -> None:
        """Over many trials, QJL inner product estimate should be unbiased."""
        dim = 256
        n_trials = 200
        rng = np.random.default_rng(42)

        errors = []
        for trial in range(n_trials):
            qjl = QJL(dim=dim, seed=trial)
            x = rng.standard_normal(dim)
            x = x / np.linalg.norm(x)
            y = rng.standard_normal(dim)

            compressed = qjl.quantize(x.reshape(1, -1))
            estimated = qjl.inner_product(y, compressed)[0]
            true_ip = np.dot(x, y)
            errors.append(estimated - true_ip)

        mean_error = np.mean(errors)
        # Mean error should be close to 0 (unbiased)
        assert abs(mean_error) < 0.15, f"Mean error {mean_error} too large for unbiased estimator"

    def test_dimension_mismatch_raises(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = qjl.quantize(vectors)
        bad_query = rng.standard_normal(128)
        with pytest.raises(DimensionMismatchError):
            qjl.inner_product(bad_query, compressed)

    def test_output_shape(self) -> None:
        qjl = QJL(dim=64, seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((20, 64))
        query = rng.standard_normal(64)
        compressed = qjl.quantize(vectors)
        scores = qjl.inner_product(query, compressed)
        assert scores.shape == (20,)

    def test_higher_projection_dim_reduces_variance(self) -> None:
        """More projection dimensions should reduce estimator variance."""
        dim = 128
        rng = np.random.default_rng(42)
        x = rng.standard_normal(dim)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(dim)
        true_ip = np.dot(x, y)

        variances = []
        for proj_dim in [32, 128]:
            errors = []
            for trial in range(100):
                qjl = QJL(dim=dim, projection_dim=proj_dim, seed=trial)
                compressed = qjl.quantize(x.reshape(1, -1))
                estimated = qjl.inner_product(y, compressed)[0]
                errors.append((estimated - true_ip) ** 2)
            variances.append(np.mean(errors))
        assert variances[0] > variances[1], "Higher projection_dim should reduce variance"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qjl.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

`src/turboquant/qjl.py`:
```python
"""QJL: 1-bit Quantized Johnson-Lindenstrauss transform.

Implements the QJL algorithm for 1-bit vector quantization with zero memory
overhead. The key insight is that applying a JL transform followed by sign-bit
quantization yields an unbiased inner product estimator.

Reference: Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache
Quantization with Zero Overhead" (AISTATS 2026).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from turboquant._accel import generate_projection_matrix, matmul, batch_inner_product
from turboquant.exceptions import DimensionMismatchError
from turboquant.storage import CompressedVectors

__all__ = ["QJL"]

logger = logging.getLogger(__name__)


class QJL:
    """1-bit quantizer using the Quantized Johnson-Lindenstrauss transform.

    Parameters
    ----------
    dim : int
        Input vector dimensionality.
    projection_dim : int or None
        Dimension of the random projection (m in the paper). Defaults to dim.
        Higher values give better accuracy at the cost of more storage.
    seed : int or None
        Random seed for deterministic projection matrix generation.
    """

    def __init__(
        self,
        dim: int,
        projection_dim: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._dim = dim
        self._projection_dim = projection_dim if projection_dim is not None else dim
        self._seed = seed
        self._projection = generate_projection_matrix(
            rows=self._projection_dim, cols=dim, seed=seed
        )

    @property
    def dim(self) -> int:
        """Input vector dimensionality."""
        return self._dim

    @property
    def projection_dim(self) -> int:
        """Projection dimensionality."""
        return self._projection_dim

    def quantize(self, vectors: NDArray[np.float64]) -> CompressedVectors:
        """Quantize vectors to 1-bit sign representations.

        Parameters
        ----------
        vectors : NDArray[np.float64]
            Input vectors of shape (n, dim).

        Returns
        -------
        CompressedVectors
            Compressed representation containing sign vectors and norms.

        Raises
        ------
        DimensionMismatchError
            If vectors have wrong dimensionality.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self._dim:
            raise DimensionMismatchError(expected=self._dim, got=vectors.shape[1])

        # Compute norms
        norms = np.linalg.norm(vectors, axis=1)

        # Project and take sign: sign(S · k)
        projected = matmul(self._projection, vectors.T).T  # (n, projection_dim)
        signs = np.sign(projected).astype(np.int8)
        # Replace zeros with +1
        signs[signs == 0] = 1

        return CompressedVectors(
            indices=np.zeros((len(vectors), self._dim), dtype=np.uint8),  # placeholder
            norms=norms,
            dim=self._dim,
            bit_width=1,
            metadata={"mode": "qjl", "projection_dim": self._projection_dim, "seed": self._seed},
            extra_arrays={"signs": signs, "projection": self._projection},
        )

    def inner_product(
        self, query: NDArray[np.float64], compressed: CompressedVectors
    ) -> NDArray[np.float64]:
        """Estimate inner products between a query and compressed vectors.

        Uses the QJL estimator:
            Prod_QJL(q, k) = sqrt(pi/2) / m * ||k||_2 * <S*q, sign(S*k)>

        Parameters
        ----------
        query : NDArray[np.float64]
            Query vector of shape (dim,).
        compressed : CompressedVectors
            Compressed vectors from ``quantize()``.

        Returns
        -------
        NDArray[np.float64]
            Estimated inner products of shape (n,).

        Raises
        ------
        DimensionMismatchError
            If query has wrong dimensionality.
        """
        query = np.asarray(query, dtype=np.float64)
        if query.shape[-1] != self._dim:
            raise DimensionMismatchError(expected=self._dim, got=query.shape[-1])

        signs = compressed.extra_arrays["signs"]  # (n, projection_dim)
        norms = compressed.norms  # (n,)
        m = self._projection_dim

        # Project query: S * q
        projected_query = matmul(self._projection, query)  # (projection_dim,)

        # Inner product: <S*q, sign(S*k)> for each compressed vector
        dot_products = batch_inner_product(projected_query, signs.astype(np.float64))  # (n,)

        # QJL estimator: sqrt(pi/2) / m * ||k|| * <Sq, sign(Sk)>
        scale = np.sqrt(np.pi / 2) / m
        return scale * norms * dot_products
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_qjl.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Update __init__.py exports**

Add to `src/turboquant/__init__.py`:
```python
from turboquant.qjl import QJL
```

Update `__all__` to include `"QJL"`.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant/qjl.py src/turboquant/__init__.py tests/test_qjl.py
git commit -m "feat: add QJL 1-bit quantizer with unbiased inner product estimation"
```

---

### Task 7: TurboQuant Quantizer

**Files:**
- Create: `src/turboquant/turboquant.py`
- Create: `tests/test_turboquant.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_turboquant.py`:
```python
"""Tests for TurboQuant MSE and inner-product quantizers."""

import numpy as np
import pytest

from turboquant.turboquant import TurboQuant
from turboquant.exceptions import DimensionMismatchError, InvalidBitWidthError, InvalidModeError


class TestTurboQuantConstruction:
    def test_valid_construction(self) -> None:
        tq = TurboQuant(dim=256, bit_width=3, mode="mse", seed=42)
        assert tq.dim == 256
        assert tq.bit_width == 3
        assert tq.mode == "mse"

    def test_invalid_bit_width_raises(self) -> None:
        with pytest.raises(InvalidBitWidthError):
            TurboQuant(dim=256, bit_width=0)
        with pytest.raises(InvalidBitWidthError):
            TurboQuant(dim=256, bit_width=5)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(InvalidModeError):
            TurboQuant(dim=256, bit_width=3, mode="bad")

    def test_deterministic_with_seed(self) -> None:
        t1 = TurboQuant(dim=64, bit_width=2, seed=42)
        t2 = TurboQuant(dim=64, bit_width=2, seed=42)
        np.testing.assert_array_equal(t1._rotation, t2._rotation)

    def test_inner_product_mode_needs_bit_width_ge_2(self) -> None:
        """inner_product mode uses (b-1) bits for MSE, so b must be >= 2."""
        with pytest.raises(InvalidBitWidthError):
            TurboQuant(dim=256, bit_width=1, mode="inner_product")


class TestTurboQuantMSE:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_quantize_output_shape(self, bit_width: int) -> None:
        tq = TurboQuant(dim=64, bit_width=bit_width, mode="mse", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = tq.quantize(vectors)
        assert compressed.num_vectors == 10
        assert compressed.dim == 64
        assert compressed.bit_width == bit_width

    def test_dimension_mismatch_raises(self) -> None:
        tq = TurboQuant(dim=64, bit_width=2, mode="mse", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 128))
        with pytest.raises(DimensionMismatchError):
            tq.quantize(vectors)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_dequantize_output_shape(self, bit_width: int) -> None:
        tq = TurboQuant(dim=64, bit_width=bit_width, mode="mse", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = tq.quantize(vectors)
        reconstructed = tq.dequantize(compressed)
        assert reconstructed.shape == (10, 64)

    def test_mse_decreases_with_bit_width(self) -> None:
        """Higher bit widths should give lower reconstruction MSE."""
        dim = 256
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, dim))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms  # unit vectors

        mses = []
        for bw in [1, 2, 3, 4]:
            tq = TurboQuant(dim=dim, bit_width=bw, mode="mse", seed=42)
            compressed = tq.quantize(vectors)
            reconstructed = tq.dequantize(compressed)
            mse = np.mean(np.sum((vectors - reconstructed) ** 2, axis=1))
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], (
                f"MSE did not decrease: bw={i+1} mse={mses[i]:.6f}, bw={i+2} mse={mses[i+1]:.6f}"
            )

    def test_distortion_within_theoretical_bound(self) -> None:
        """MSE should be within the paper's theoretical upper bound.

        Theorem 1: D_mse <= sqrt(3*pi)/2 * 1/4^b for unit vectors.
        """
        dim = 512
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((200, dim))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        for bw in [2, 3, 4]:
            tq = TurboQuant(dim=dim, bit_width=bw, mode="mse", seed=42)
            compressed = tq.quantize(vectors)
            reconstructed = tq.dequantize(compressed)
            mse = np.mean(np.sum((vectors - reconstructed) ** 2, axis=1))

            upper_bound = np.sqrt(3 * np.pi) / 2 * (1 / 4**bw)
            # Allow 2x slack for finite-dimension effects
            assert mse < upper_bound * 2, (
                f"MSE {mse:.6f} exceeds 2x theoretical bound {upper_bound:.6f} at bw={bw}"
            )


class TestTurboQuantInnerProduct:
    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_inner_product_output_shape(self, bit_width: int) -> None:
        tq = TurboQuant(dim=64, bit_width=bit_width, mode="inner_product", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((20, 64))
        query = rng.standard_normal(64)
        compressed = tq.quantize(vectors)
        scores = tq.inner_product(query, compressed)
        assert scores.shape == (20,)

    def test_unbiased_estimator(self) -> None:
        """TurboQuant_prod should provide unbiased inner product estimates."""
        dim = 256
        n_trials = 200
        rng = np.random.default_rng(42)

        x = rng.standard_normal(dim)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(dim)
        true_ip = np.dot(x, y)

        errors = []
        for trial in range(n_trials):
            tq = TurboQuant(dim=dim, bit_width=3, mode="inner_product", seed=trial)
            compressed = tq.quantize(x.reshape(1, -1))
            estimated = tq.inner_product(y, compressed)[0]
            errors.append(estimated - true_ip)

        mean_error = np.mean(errors)
        assert abs(mean_error) < 0.15, f"Mean error {mean_error} too large for unbiased estimator"

    def test_dimension_mismatch_on_query_raises(self) -> None:
        tq = TurboQuant(dim=64, bit_width=2, mode="inner_product", seed=42)
        rng = np.random.default_rng(0)
        vectors = rng.standard_normal((10, 64))
        compressed = tq.quantize(vectors)
        bad_query = rng.standard_normal(128)
        with pytest.raises(DimensionMismatchError):
            tq.inner_product(bad_query, compressed)


class TestTurboQuantDeterminism:
    def test_same_seed_same_result(self) -> None:
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((10, 64))

        tq1 = TurboQuant(dim=64, bit_width=3, mode="mse", seed=99)
        tq2 = TurboQuant(dim=64, bit_width=3, mode="mse", seed=99)

        c1 = tq1.quantize(vectors)
        c2 = tq2.quantize(vectors)

        r1 = tq1.dequantize(c1)
        r2 = tq2.dequantize(c2)

        np.testing.assert_array_equal(r1, r2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_turboquant.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

`src/turboquant/turboquant.py`:
```python
"""TurboQuant: Online vector quantization with near-optimal distortion rate.

Implements TurboQuant_mse (optimized for reconstruction MSE) and
TurboQuant_prod (optimized for unbiased inner product estimation).

Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate" (ICLR 2026).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from turboquant._accel import generate_orthogonal_matrix, generate_projection_matrix, matmul
from turboquant.codebook import get_codebook, quantize_scalar, dequantize_scalar
from turboquant.exceptions import DimensionMismatchError, InvalidBitWidthError, InvalidModeError
from turboquant.storage import CompressedVectors

__all__ = ["TurboQuant"]

logger = logging.getLogger(__name__)

_VALID_BIT_WIDTHS = (1, 2, 3, 4)
_VALID_MODES = ("mse", "inner_product")


class TurboQuant:
    """TurboQuant vector quantizer.

    Parameters
    ----------
    dim : int
        Input vector dimensionality.
    bit_width : int
        Bits per coordinate (1-4). For mode="inner_product", must be >= 2
        since one bit is reserved for the QJL residual correction.
    mode : "mse" or "inner_product"
        Quantization mode. "mse" optimizes reconstruction MSE. "inner_product"
        provides an unbiased inner product estimator by applying QJL to the
        MSE residual.
    seed : int or None
        Random seed for deterministic rotation/projection matrices.
    outlier_channels : int
        Number of outlier channels for mixed-precision quantization (0 = disabled).
    outlier_bit_width : int or None
        Bit-width for outlier channels. Defaults to bit_width + 1.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int,
        mode: Literal["mse", "inner_product"] = "inner_product",
        seed: int | None = None,
        outlier_channels: int = 0,
        outlier_bit_width: int | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise InvalidModeError(mode)
        if mode == "inner_product":
            if bit_width < 2 or bit_width > 4:
                raise InvalidBitWidthError(bit_width, valid_range=(2, 4))
        else:
            if bit_width < 1 or bit_width > 4:
                raise InvalidBitWidthError(bit_width, valid_range=(1, 4))

        self._dim = dim
        self._bit_width = bit_width
        self._mode = mode
        self._seed = seed
        self._outlier_channels = outlier_channels
        self._outlier_bit_width = outlier_bit_width or min(bit_width + 1, 4)

        # Generate rotation matrix
        self._rotation = generate_orthogonal_matrix(dim, seed=seed)

        # For inner_product mode, generate QJL projection matrix with a different seed
        self._qjl_projection: NDArray | None = None
        if mode == "inner_product":
            qjl_seed = seed + 1_000_000 if seed is not None else None
            self._qjl_projection = generate_projection_matrix(
                rows=dim, cols=dim, seed=qjl_seed
            )

    @property
    def dim(self) -> int:
        """Input vector dimensionality."""
        return self._dim

    @property
    def bit_width(self) -> int:
        """Quantization bit-width."""
        return self._bit_width

    @property
    def mode(self) -> str:
        """Quantization mode."""
        return self._mode

    def quantize(self, vectors: NDArray[np.float64]) -> CompressedVectors:
        """Quantize vectors.

        Parameters
        ----------
        vectors : NDArray[np.float64]
            Input vectors of shape (n, dim).

        Returns
        -------
        CompressedVectors
            Compressed representation.

        Raises
        ------
        DimensionMismatchError
            If vectors have wrong dimensionality.
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self._dim:
            raise DimensionMismatchError(expected=self._dim, got=vectors.shape[1])

        n = vectors.shape[0]
        norms = np.linalg.norm(vectors, axis=1)

        # Normalize to unit vectors for quantization
        safe_norms = np.where(norms > 0, norms, 1.0)
        unit_vectors = vectors / safe_norms[:, np.newaxis]

        if self._mode == "mse":
            return self._quantize_mse(unit_vectors, norms, self._bit_width)
        else:
            return self._quantize_prod(unit_vectors, norms)

    def _quantize_mse(
        self, unit_vectors: NDArray, norms: NDArray, bit_width: int
    ) -> CompressedVectors:
        """TurboQuant_mse: MSE-optimized quantization."""
        n = unit_vectors.shape[0]

        # Rotate: y = Pi * x
        rotated = matmul(self._rotation, unit_vectors.T).T  # (n, dim)

        # Get codebook for this dimension and bit-width
        centroids, boundaries = get_codebook(self._dim, bit_width)

        # Quantize each coordinate independently
        indices = np.empty((n, self._dim), dtype=np.uint8)
        for i in range(n):
            indices[i] = quantize_scalar(rotated[i], boundaries)

        return CompressedVectors(
            indices=indices,
            norms=norms,
            dim=self._dim,
            bit_width=bit_width,
            metadata={
                "mode": self._mode,
                "seed": self._seed,
                "actual_bit_width": bit_width,
            },
            extra_arrays={"rotation": self._rotation},
        )

    def _quantize_prod(
        self, unit_vectors: NDArray, norms: NDArray
    ) -> CompressedVectors:
        """TurboQuant_prod: inner-product optimized quantization."""
        n = unit_vectors.shape[0]
        mse_bit_width = self._bit_width - 1

        # Step 1: Apply TurboQuant_mse at (b-1) bits
        rotated = matmul(self._rotation, unit_vectors.T).T
        centroids, boundaries = get_codebook(self._dim, mse_bit_width)

        indices = np.empty((n, self._dim), dtype=np.uint8)
        for i in range(n):
            indices[i] = quantize_scalar(rotated[i], boundaries)

        # Step 2: Compute residual r = x - dequantize_mse(quantize_mse(x))
        dequantized_rotated = np.empty_like(rotated)
        for i in range(n):
            dequantized_rotated[i] = dequantize_scalar(indices[i], centroids)
        dequantized = matmul(self._rotation.T, dequantized_rotated.T).T
        residuals = unit_vectors - dequantized
        residual_norms = np.linalg.norm(residuals, axis=1)

        # Step 3: Apply QJL to residual
        assert self._qjl_projection is not None
        projected_residuals = matmul(self._qjl_projection, residuals.T).T
        qjl_signs = np.sign(projected_residuals).astype(np.int8)
        qjl_signs[qjl_signs == 0] = 1

        return CompressedVectors(
            indices=indices,
            norms=norms,
            dim=self._dim,
            bit_width=self._bit_width,
            metadata={
                "mode": self._mode,
                "seed": self._seed,
                "mse_bit_width": mse_bit_width,
            },
            extra_arrays={
                "rotation": self._rotation,
                "qjl_projection": self._qjl_projection,
                "qjl_signs": qjl_signs,
                "residual_norms": residual_norms,
            },
        )

    def dequantize(self, compressed: CompressedVectors) -> NDArray[np.float64]:
        """Reconstruct approximate original vectors.

        Parameters
        ----------
        compressed : CompressedVectors
            Compressed vectors from ``quantize()``.

        Returns
        -------
        NDArray[np.float64]
            Reconstructed vectors of shape (n, dim).
        """
        actual_bw = compressed.metadata.get("mse_bit_width", compressed.bit_width)
        centroids, _ = get_codebook(self._dim, actual_bw)

        n = compressed.num_vectors
        dequantized_rotated = np.empty((n, self._dim), dtype=np.float64)
        for i in range(n):
            dequantized_rotated[i] = dequantize_scalar(compressed.indices[i], centroids)

        # Rotate back: x_hat = Pi^T * y_hat
        reconstructed = matmul(self._rotation.T, dequantized_rotated.T).T

        # If inner_product mode, add QJL residual reconstruction
        if self._mode == "inner_product" and "qjl_signs" in compressed.extra_arrays:
            qjl_signs = compressed.extra_arrays["qjl_signs"]
            residual_norms = compressed.extra_arrays["residual_norms"]
            qjl_proj = compressed.extra_arrays["qjl_projection"]
            m = qjl_proj.shape[0]
            qjl_reconstructed = np.sqrt(np.pi / 2) / m * matmul(
                qjl_proj.T, (qjl_signs * residual_norms[:, np.newaxis]).T
            ).T
            reconstructed = reconstructed + qjl_reconstructed

        # Scale by original norms
        reconstructed = reconstructed * compressed.norms[:, np.newaxis]
        return reconstructed

    def inner_product(
        self, query: NDArray[np.float64], compressed: CompressedVectors
    ) -> NDArray[np.float64]:
        """Estimate inner products between a query and compressed vectors.

        Parameters
        ----------
        query : NDArray[np.float64]
            Query vector of shape (dim,).
        compressed : CompressedVectors
            Compressed vectors from ``quantize()``.

        Returns
        -------
        NDArray[np.float64]
            Estimated inner products of shape (n,).

        Raises
        ------
        DimensionMismatchError
            If query has wrong dimensionality.
        """
        query = np.asarray(query, dtype=np.float64)
        if query.shape[-1] != self._dim:
            raise DimensionMismatchError(expected=self._dim, got=query.shape[-1])

        if self._mode == "mse":
            # For MSE mode, just use dequantized vectors
            reconstructed = self.dequantize(compressed)
            return reconstructed @ query

        # For inner_product mode, use the unbiased estimator
        # <y, x_mse> + ||r|| * sqrt(pi/2) / d * <S*y, sign(S*r)>
        actual_bw = compressed.metadata.get("mse_bit_width", self._bit_width - 1)
        centroids, _ = get_codebook(self._dim, actual_bw)

        n = compressed.num_vectors
        dequantized_rotated = np.empty((n, self._dim), dtype=np.float64)
        for i in range(n):
            dequantized_rotated[i] = dequantize_scalar(compressed.indices[i], centroids)
        mse_reconstructed = matmul(self._rotation.T, dequantized_rotated.T).T
        # Scale by norms
        mse_reconstructed = mse_reconstructed * compressed.norms[:, np.newaxis]

        # MSE part: <y, x_mse>
        mse_scores = mse_reconstructed @ query

        # QJL residual correction
        qjl_signs = compressed.extra_arrays["qjl_signs"]  # (n, dim)
        residual_norms = compressed.extra_arrays["residual_norms"]  # (n,)
        qjl_proj = compressed.extra_arrays["qjl_projection"]  # (dim, dim)
        m = qjl_proj.shape[0]

        # Project query: S * y
        projected_query = matmul(qjl_proj, query)  # (dim,)

        # <S*y, sign(S*r)> for each vector
        qjl_dots = qjl_signs.astype(np.float64) @ projected_query  # (n,)

        # Scale: ||r|| * sqrt(pi/2) / m * <Sy, sign(Sr)> * ||x||
        qjl_correction = (
            compressed.norms * residual_norms * np.sqrt(np.pi / 2) / m * qjl_dots
        )

        return mse_scores + qjl_correction
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_turboquant.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Update __init__.py exports**

Add to `src/turboquant/__init__.py`:
```python
from turboquant.turboquant import TurboQuant
```

Update `__all__` to include `"TurboQuant"`.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant/turboquant.py src/turboquant/__init__.py tests/test_turboquant.py
git commit -m "feat: add TurboQuant MSE and inner-product quantizers"
```

---

### Task 8: Batch Quantization and CompressedStore Search

**Files:**
- Modify: `src/turboquant/turboquant.py`
- Modify: `src/turboquant/storage.py`
- Modify: `tests/test_storage.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_storage.py`:
```python
class TestCompressedStoreSearch:
    def test_search_returns_top_k(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        n = 50
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((n, dim))
        query = rng.standard_normal(dim)

        tq = TurboQuant(dim=dim, bit_width=3, mode="mse", seed=42)
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")
        assert store.dim == dim
        assert store.num_vectors == n

    def test_store_metadata(self, tmp_path: Path) -> None:
        from turboquant.turboquant import TurboQuant

        dim = 64
        tq = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=42)
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        compressed = tq.quantize(vectors)
        compressed.save(tmp_path / "store")

        store = CompressedStore.load(tmp_path / "store")
        assert store.bit_width == 2
        assert store.mode == "mse"
```

Create `tests/test_integration.py`:
```python
"""End-to-end integration tests."""

import numpy as np
import pytest
from pathlib import Path

from turboquant import QJL, TurboQuant, CompressedVectors


class TestQJLEndToEnd:
    def test_quantize_search_recovers_nearest(self) -> None:
        """QJL should rank the true nearest neighbor highly."""
        dim = 256
        n = 200
        rng = np.random.default_rng(42)

        vectors = rng.standard_normal((n, dim))
        query = rng.standard_normal(dim)

        # True nearest neighbor by inner product
        true_scores = vectors @ query
        true_best = np.argmax(true_scores)

        qjl = QJL(dim=dim, seed=42)
        compressed = qjl.quantize(vectors)
        estimated_scores = qjl.inner_product(query, compressed)
        estimated_best = np.argmax(estimated_scores)

        # The true best should be in the top-5 estimated
        top_5 = np.argsort(estimated_scores)[-5:]
        assert true_best in top_5, f"True best {true_best} not in top-5 estimated {top_5}"


class TestTurboQuantEndToEnd:
    @pytest.mark.parametrize("mode", ["mse", "inner_product"])
    def test_quantize_dequantize_round_trip(self, mode: str) -> None:
        dim = 256
        bw = 3 if mode == "inner_product" else 3
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((50, dim))

        tq = TurboQuant(dim=dim, bit_width=bw, mode=mode, seed=42)
        compressed = tq.quantize(vectors)
        reconstructed = tq.dequantize(compressed)

        # Reconstruction should be correlated with original
        for i in range(50):
            corr = np.corrcoef(vectors[i], reconstructed[i])[0, 1]
            assert corr > 0.5, f"Vector {i} correlation {corr} too low"

    def test_inner_product_search_recovers_nearest(self) -> None:
        """TurboQuant_prod should rank the true nearest neighbor highly."""
        dim = 256
        n = 200
        rng = np.random.default_rng(42)

        vectors = rng.standard_normal((n, dim))
        query = rng.standard_normal(dim)

        true_scores = vectors @ query
        true_best = np.argmax(true_scores)

        tq = TurboQuant(dim=dim, bit_width=3, mode="inner_product", seed=42)
        compressed = tq.quantize(vectors)
        estimated_scores = tq.inner_product(query, compressed)

        top_5 = np.argsort(estimated_scores)[-5:]
        assert true_best in top_5, f"True best {true_best} not in top-5 estimated {top_5}"

    def test_save_load_preserves_search(self, tmp_path: Path) -> None:
        """Save/load should preserve the ability to search."""
        dim = 64
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((30, dim))
        query = rng.standard_normal(dim)

        tq = TurboQuant(dim=dim, bit_width=3, mode="mse", seed=42)
        compressed = tq.quantize(vectors)
        scores_before = tq.inner_product(query, compressed)

        compressed.save(tmp_path / "test_store")
        loaded = CompressedVectors.load(tmp_path / "test_store")
        scores_after = tq.inner_product(query, loaded)

        np.testing.assert_allclose(scores_before, scores_after)


class TestBatchQuantization:
    def test_batched_matches_single(self, tmp_path: Path) -> None:
        """Batch quantization should produce identical results to single-batch."""
        dim = 64
        rng = np.random.default_rng(42)
        all_vectors = rng.standard_normal((100, dim))

        tq = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=42)

        # Single batch
        single_compressed = tq.quantize(all_vectors)
        single_reconstructed = tq.dequantize(single_compressed)

        # Batched
        def vector_iterator():
            for i in range(0, 100, 25):
                yield all_vectors[i:i+25]

        tq.quantize_batched(
            vector_iterator(), batch_size=25, output_path=tmp_path / "batched"
        )
        batched_loaded = CompressedVectors.load(tmp_path / "batched")
        batched_reconstructed = tq.dequantize(batched_loaded)

        np.testing.assert_allclose(single_reconstructed, batched_reconstructed, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_integration.py tests/test_storage.py::TestCompressedStoreSearch -v`
Expected: FAIL — `quantize_batched` method does not exist, store search tests may fail.

- [ ] **Step 3: Add quantize_batched to TurboQuant**

Add this method to the `TurboQuant` class in `src/turboquant/turboquant.py`:

```python
    def quantize_batched(
        self,
        vectors: Iterable[NDArray[np.float64]],
        batch_size: int = 10_000,
        output_path: str | Path = "index.tqz",
    ) -> None:
        """Quantize vectors in batches, writing progressively to disk.

        Parameters
        ----------
        vectors : Iterable[NDArray[np.float64]]
            Iterator yielding batches of vectors, each of shape (batch_n, dim).
        batch_size : int
            Not used when vectors is an iterator of pre-batched arrays.
            Included for API compatibility.
        output_path : str or Path
            Directory path for the output store.
        """
        from pathlib import Path as _Path

        output_path = _Path(output_path)
        parts: list[CompressedVectors] = []

        for batch in vectors:
            compressed = self.quantize(batch)
            parts.append(compressed)

        if not parts:
            raise ValueError("No vectors provided to quantize_batched")

        merged = CompressedVectors.concatenate(parts)
        merged.save(output_path)
        logger.info(
            "Batched quantization complete: %d vectors saved to %s",
            merged.num_vectors,
            output_path,
        )
```

Also add the import at the top of `turboquant.py`:
```python
from typing import Iterable, Literal
from pathlib import Path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_integration.py tests/test_storage.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/turboquant/turboquant.py src/turboquant/storage.py tests/test_storage.py tests/test_integration.py
git commit -m "feat: add batch quantization and end-to-end integration tests"
```

---

### Task 9: Outlier Handling

**Files:**
- Modify: `src/turboquant/turboquant.py`
- Create: `tests/test_outliers.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_outliers.py`:
```python
"""Tests for outlier channel handling in TurboQuant."""

import numpy as np
import pytest

from turboquant.turboquant import TurboQuant


class TestOutlierHandling:
    def test_outlier_channels_produces_valid_output(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        # Inject outliers: make channels 0-3 have 10x magnitude
        vectors[:, :4] *= 10.0

        tq = TurboQuant(
            dim=dim, bit_width=2, mode="mse", seed=42,
            outlier_channels=4, outlier_bit_width=3,
        )
        compressed = tq.quantize(vectors)
        assert compressed.num_vectors == 20
        assert compressed.dim == dim

    def test_outlier_reduces_mse_on_outlier_data(self) -> None:
        """Outlier handling should reduce MSE when outlier channels are present."""
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((100, dim))
        # Inject outliers
        vectors[:, :4] *= 10.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms

        # Without outlier handling
        tq_no_outlier = TurboQuant(dim=dim, bit_width=2, mode="mse", seed=42)
        c1 = tq_no_outlier.quantize(unit_vectors)
        r1 = tq_no_outlier.dequantize(c1)
        mse_no_outlier = np.mean(np.sum((unit_vectors - r1) ** 2, axis=1))

        # With outlier handling
        tq_outlier = TurboQuant(
            dim=dim, bit_width=2, mode="mse", seed=42,
            outlier_channels=4, outlier_bit_width=3,
        )
        c2 = tq_outlier.quantize(unit_vectors)
        r2 = tq_outlier.dequantize(c2)
        mse_outlier = np.mean(np.sum((unit_vectors - r2) ** 2, axis=1))

        assert mse_outlier < mse_no_outlier, (
            f"Outlier handling MSE {mse_outlier:.6f} should be less than "
            f"no-outlier MSE {mse_no_outlier:.6f}"
        )

    def test_dequantize_round_trip_with_outliers(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        vectors[:, :4] *= 10.0

        tq = TurboQuant(
            dim=dim, bit_width=3, mode="mse", seed=42,
            outlier_channels=4, outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        reconstructed = tq.dequantize(compressed)
        assert reconstructed.shape == (20, dim)
        # Check reconstruction is correlated
        for i in range(20):
            corr = np.corrcoef(vectors[i], reconstructed[i])[0, 1]
            assert corr > 0.5

    def test_inner_product_mode_with_outliers(self) -> None:
        dim = 128
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((20, dim))
        vectors[:, :4] *= 10.0
        query = rng.standard_normal(dim)

        tq = TurboQuant(
            dim=dim, bit_width=3, mode="inner_product", seed=42,
            outlier_channels=4, outlier_bit_width=4,
        )
        compressed = tq.quantize(vectors)
        scores = tq.inner_product(query, compressed)
        assert scores.shape == (20,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_outliers.py -v`
Expected: Tests fail because outlier_channels is accepted but not used in quantization logic.

- [ ] **Step 3: Implement outlier handling**

Modify `src/turboquant/turboquant.py`. Update `_quantize_mse` to detect and split outlier channels:

In `__init__`, add after the existing outlier attribute assignments:
```python
        # Identify outlier channel indices if enabled
        self._outlier_indices: NDArray[np.intp] | None = None
        self._inlier_indices: NDArray[np.intp] | None = None
```

Update `_quantize_mse` to handle outliers:
```python
    def _quantize_mse(
        self, unit_vectors: NDArray, norms: NDArray, bit_width: int
    ) -> CompressedVectors:
        """TurboQuant_mse: MSE-optimized quantization."""
        n = unit_vectors.shape[0]

        # Rotate: y = Pi * x
        rotated = matmul(self._rotation, unit_vectors.T).T  # (n, dim)

        if self._outlier_channels > 0:
            # Detect outlier channels by average magnitude across vectors
            channel_magnitudes = np.mean(np.abs(rotated), axis=0)
            outlier_idx = np.argsort(channel_magnitudes)[-self._outlier_channels:]
            inlier_idx = np.setdiff1d(np.arange(self._dim), outlier_idx)

            # Quantize inlier channels at bit_width
            centroids_in, boundaries_in = get_codebook(self._dim, bit_width)
            # Quantize outlier channels at outlier_bit_width
            centroids_out, boundaries_out = get_codebook(self._dim, self._outlier_bit_width)

            indices = np.empty((n, self._dim), dtype=np.uint8)
            for i in range(n):
                indices[i, inlier_idx] = quantize_scalar(rotated[i, inlier_idx], boundaries_in)
                indices[i, outlier_idx] = quantize_scalar(rotated[i, outlier_idx], boundaries_out)

            return CompressedVectors(
                indices=indices,
                norms=norms,
                dim=self._dim,
                bit_width=bit_width,
                metadata={
                    "mode": self._mode,
                    "seed": self._seed,
                    "actual_bit_width": bit_width,
                    "outlier_channels": self._outlier_channels,
                    "outlier_bit_width": self._outlier_bit_width,
                },
                extra_arrays={
                    "rotation": self._rotation,
                    "outlier_indices": outlier_idx,
                    "inlier_indices": inlier_idx,
                },
            )

        # Non-outlier path (existing logic)
        centroids, boundaries = get_codebook(self._dim, bit_width)
        indices = np.empty((n, self._dim), dtype=np.uint8)
        for i in range(n):
            indices[i] = quantize_scalar(rotated[i], boundaries)

        return CompressedVectors(
            indices=indices,
            norms=norms,
            dim=self._dim,
            bit_width=bit_width,
            metadata={
                "mode": self._mode,
                "seed": self._seed,
                "actual_bit_width": bit_width,
            },
            extra_arrays={"rotation": self._rotation},
        )
```

Update `dequantize` to handle outlier channels:
```python
    def dequantize(self, compressed: CompressedVectors) -> NDArray[np.float64]:
        actual_bw = compressed.metadata.get("mse_bit_width", compressed.bit_width)
        outlier_bw = compressed.metadata.get("outlier_bit_width")
        outlier_idx = compressed.extra_arrays.get("outlier_indices")
        inlier_idx = compressed.extra_arrays.get("inlier_indices")

        n = compressed.num_vectors
        dequantized_rotated = np.empty((n, self._dim), dtype=np.float64)

        if outlier_idx is not None and outlier_bw is not None:
            centroids_in, _ = get_codebook(self._dim, actual_bw)
            centroids_out, _ = get_codebook(self._dim, outlier_bw)
            for i in range(n):
                dequantized_rotated[i, inlier_idx] = dequantize_scalar(
                    compressed.indices[i, inlier_idx], centroids_in
                )
                dequantized_rotated[i, outlier_idx] = dequantize_scalar(
                    compressed.indices[i, outlier_idx], centroids_out
                )
        else:
            centroids, _ = get_codebook(self._dim, actual_bw)
            for i in range(n):
                dequantized_rotated[i] = dequantize_scalar(compressed.indices[i], centroids)

        # Rotate back
        reconstructed = matmul(self._rotation.T, dequantized_rotated.T).T

        # Handle QJL residual for inner_product mode (same as before)
        if self._mode == "inner_product" and "qjl_signs" in compressed.extra_arrays:
            qjl_signs = compressed.extra_arrays["qjl_signs"]
            residual_norms = compressed.extra_arrays["residual_norms"]
            qjl_proj = compressed.extra_arrays["qjl_projection"]
            m = qjl_proj.shape[0]
            qjl_reconstructed = np.sqrt(np.pi / 2) / m * matmul(
                qjl_proj.T, (qjl_signs * residual_norms[:, np.newaxis]).T
            ).T
            reconstructed = reconstructed + qjl_reconstructed

        # Scale by original norms
        reconstructed = reconstructed * compressed.norms[:, np.newaxis]
        return reconstructed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_outliers.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run full test suite to ensure nothing broke**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/turboquant/turboquant.py tests/test_outliers.py
git commit -m "feat: add outlier channel handling for mixed-precision quantization"
```

---

### Task 10: Benchmarks

**Files:**
- Create: `benchmarks/bench_all.py`

- [ ] **Step 1: Write the benchmark script**

`benchmarks/bench_all.py`:
```python
"""Performance benchmarks for TurboQuant library.

Run with: python benchmarks/bench_all.py

Measures quantization throughput, search throughput, memory footprint,
and distortion across dimensions, bit-widths, and collection sizes.
Compares NumPy vs PyTorch when available.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from turboquant import QJL, TurboQuant
from turboquant._accel import has_torch


@dataclass
class BenchmarkResult:
    name: str
    dim: int
    n_vectors: int
    bit_width: int
    quantize_time_s: float
    search_time_s: float
    memory_bytes: int
    mse: float | None = None


def measure_quantize_time(
    quantizer: TurboQuant | QJL, vectors: np.ndarray
) -> tuple[float, object]:
    start = time.perf_counter()
    compressed = quantizer.quantize(vectors)
    elapsed = time.perf_counter() - start
    return elapsed, compressed


def measure_search_time(
    quantizer: TurboQuant | QJL,
    query: np.ndarray,
    compressed: object,
    n_queries: int = 100,
) -> float:
    queries = np.random.default_rng(0).standard_normal((n_queries, len(query)))
    start = time.perf_counter()
    for q in queries:
        quantizer.inner_product(q, compressed)
    elapsed = time.perf_counter() - start
    return elapsed / n_queries


def estimate_memory(compressed: object) -> int:
    """Rough estimate of compressed representation size in bytes."""
    mem = compressed.indices.nbytes + compressed.norms.nbytes
    for arr in compressed.extra_arrays.values():
        mem += arr.nbytes
    return mem


def run_turboquant_benchmarks() -> list[BenchmarkResult]:
    results = []
    dims = [384, 768, 1536, 3072]
    n_vectors_list = [1_000, 10_000, 100_000]
    bit_widths = [2, 3, 4]

    for dim in dims:
        for n_vectors in n_vectors_list:
            # Skip very large combos to keep runtime manageable
            if n_vectors >= 100_000 and dim >= 3072:
                continue

            rng = np.random.default_rng(42)
            vectors = rng.standard_normal((n_vectors, dim))
            query = rng.standard_normal(dim)

            for bw in bit_widths:
                tq = TurboQuant(dim=dim, bit_width=bw, mode="mse", seed=42)

                qt, compressed = measure_quantize_time(tq, vectors)
                st = measure_search_time(tq, query, compressed, n_queries=10)
                mem = estimate_memory(compressed)

                # MSE on first 100 vectors
                subset = vectors[:100]
                norms = np.linalg.norm(subset, axis=1, keepdims=True)
                unit_subset = subset / norms
                c = tq.quantize(unit_subset)
                r = tq.dequantize(c)
                mse = float(np.mean(np.sum((unit_subset - r) ** 2, axis=1)))

                result = BenchmarkResult(
                    name=f"TurboQuant(mse,bw={bw})",
                    dim=dim,
                    n_vectors=n_vectors,
                    bit_width=bw,
                    quantize_time_s=qt,
                    search_time_s=st,
                    memory_bytes=mem,
                    mse=mse,
                )
                results.append(result)
                print(
                    f"  {result.name:30s} dim={dim:5d} n={n_vectors:7d} "
                    f"quant={qt:8.3f}s search={st:8.5f}s/q "
                    f"mem={mem/1024/1024:7.2f}MB mse={mse:.6f}"
                )

    return results


def run_qjl_benchmarks() -> list[BenchmarkResult]:
    results = []
    dims = [384, 768, 1536]
    n_vectors_list = [1_000, 10_000]

    for dim in dims:
        for n_vectors in n_vectors_list:
            rng = np.random.default_rng(42)
            vectors = rng.standard_normal((n_vectors, dim))
            query = rng.standard_normal(dim)

            qjl = QJL(dim=dim, seed=42)
            qt, compressed = measure_quantize_time(qjl, vectors)
            st = measure_search_time(qjl, query, compressed, n_queries=10)
            mem = estimate_memory(compressed)

            result = BenchmarkResult(
                name="QJL(1-bit)",
                dim=dim,
                n_vectors=n_vectors,
                bit_width=1,
                quantize_time_s=qt,
                search_time_s=st,
                memory_bytes=mem,
            )
            results.append(result)
            print(
                f"  {result.name:30s} dim={dim:5d} n={n_vectors:7d} "
                f"quant={qt:8.3f}s search={st:8.5f}s/q "
                f"mem={mem/1024/1024:7.2f}MB"
            )

    return results


def main() -> None:
    print("=" * 80)
    print("TurboQuant Benchmarks")
    print("=" * 80)
    print(f"PyTorch available: {has_torch()}")
    print()

    print("--- QJL Benchmarks ---")
    qjl_results = run_qjl_benchmarks()
    print()

    print("--- TurboQuant Benchmarks ---")
    tq_results = run_turboquant_benchmarks()
    print()

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmarks**

Run: `cd /Users/msilverblatt/Projects/piedpiper && python benchmarks/bench_all.py`
Expected: Benchmark results printed to stdout. No crashes.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_all.py
git commit -m "feat: add performance benchmark suite"
```

---

### Task 11: Final Cleanup and Full Test Suite

**Files:**
- Modify: `src/turboquant/__init__.py` (ensure all exports are complete)
- All test files

- [ ] **Step 1: Verify __init__.py has all exports**

Read `src/turboquant/__init__.py` and verify it exports:
- `QJL`
- `TurboQuant`
- `CompressedVectors`
- `CompressedStore`
- `compute_codebook`
- `get_codebook`
- All exceptions
- `__version__`

- [ ] **Step 2: Run full test suite**

Run:
```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS.

- [ ] **Step 3: Run ruff on entire codebase**

Run:
```bash
ruff check src/ tests/ benchmarks/
ruff format --check src/ tests/ benchmarks/
```
Expected: No errors. If there are formatting issues, run `ruff format src/ tests/ benchmarks/` to fix.

- [ ] **Step 4: Run ruff format if needed**

Run: `ruff format src/ tests/ benchmarks/`

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup, lint, and formatting pass"
```
