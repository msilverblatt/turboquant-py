"""QJL: 1-bit Quantized Johnson-Lindenstrauss transform.

Implements the QJL algorithm for 1-bit vector quantization with zero memory
overhead. The key insight is that applying a JL transform followed by sign-bit
quantization yields an unbiased inner product estimator.

Reference: Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache
Quantization with Zero Overhead" (AISTATS 2026).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from turboquant._accel import batch_inner_product, generate_projection_matrix, matmul

if TYPE_CHECKING:
    from numpy.typing import NDArray
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
        if self._projection_dim > dim:
            raise ValueError(f"projection_dim ({self._projection_dim}) cannot exceed dim ({dim})")
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

        norms = np.linalg.norm(vectors, axis=1)

        # Project and take sign: sign(S · k)
        projected = matmul(self._projection, vectors.T).T  # (n, projection_dim)
        signs = np.sign(projected).astype(np.int8)
        signs[signs == 0] = 1

        return CompressedVectors(
            indices=np.zeros((len(vectors), self._dim), dtype=np.uint8),
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
        dot_products = batch_inner_product(projected_query, signs.astype(np.float64))

        # QJL estimator: sqrt(pi/2) / m * ||k|| * <Sq, sign(Sk)>
        scale = np.sqrt(np.pi / 2) / m
        return scale * norms * dot_products
