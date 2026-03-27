"""Lloyd-Max codebook computation for the Beta distribution.

Computes optimal scalar quantizers for coordinates of randomly rotated
unit vectors. In high dimensions, each coordinate follows a Beta distribution
that converges to Gaussian N(0, 1/d).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "beta_pdf",
    "compute_codebook",
    "dequantize_scalar",
    "get_codebook",
    "quantize_scalar",
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
    n_points = 10000

    centroids = np.linspace(-1 + 1 / n_centroids, 1 - 1 / n_centroids, n_centroids)

    x_grid = np.linspace(-0.9999, 0.9999, n_points)
    pdf_vals = beta_pdf(x_grid, dim)

    for iteration in range(max_iter):
        boundaries = np.empty(n_centroids + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(n_centroids - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2

        new_centroids = np.empty(n_centroids)
        for i in range(n_centroids):
            mask = (x_grid >= boundaries[i]) & (x_grid < boundaries[i + 1])
            if i == n_centroids - 1:
                mask = (x_grid >= boundaries[i]) & (x_grid <= boundaries[i + 1])
            if np.any(mask):
                weights = pdf_vals[mask]
                total_weight = np.trapezoid(weights, x_grid[mask])
                if total_weight > 0:
                    numerator = np.trapezoid(x_grid[mask] * weights, x_grid[mask])
                    new_centroids[i] = numerator / total_weight
                else:
                    new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = centroids[i]

        max_shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if max_shift < tol:
            logger.debug(
                "Lloyd-Max converged after %d iterations (max_shift=%.2e)",
                iteration + 1,
                max_shift,
            )
            break

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
