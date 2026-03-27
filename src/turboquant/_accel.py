"""Acceleration dispatch layer for NumPy/PyTorch.

All matrix operations go through this module. If PyTorch is installed,
operations dispatch to torch tensors for potential speedup. Otherwise,
falls back to NumPy.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "batch_inner_product",
    "generate_orthogonal_matrix",
    "generate_projection_matrix",
    "has_torch",
    "matmul",
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
