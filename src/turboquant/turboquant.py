"""TurboQuant MSE and inner-product quantizers.

Implements two quantization modes:
- MSE mode: Minimizes mean squared error via random rotation + Lloyd-Max scalar quantization.
- Inner-product mode: Uses MSE quantization at (b-1) bits plus QJL on the residual
  to produce an unbiased inner-product estimator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from turboquant._accel import (
    batch_inner_product,
    generate_orthogonal_matrix,
    generate_projection_matrix,
    matmul,
)
from turboquant.codebook import dequantize_scalar, get_codebook, quantize_scalar
from turboquant.exceptions import DimensionMismatchError, InvalidBitWidthError, InvalidModeError
from turboquant.storage import CompressedVectors

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

__all__ = ["TurboQuant"]

logger = logging.getLogger(__name__)

_VALID_MODES = {"mse", "inner_product"}
_MIN_BIT_WIDTH = 1
_MAX_BIT_WIDTH = 4
_QJL_SEED_OFFSET = 1_000_000


class TurboQuant:
    """Vector quantizer with MSE and inner-product modes.

    Parameters
    ----------
    dim : int
        Input vector dimensionality.
    bit_width : int
        Bits per quantized coordinate (1-4).
    mode : str
        Quantization mode: ``"mse"`` or ``"inner_product"``.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        bit_width: int,
        mode: str = "mse",
        seed: int | None = None,
        outlier_channels: int = 0,
        outlier_bit_width: int | None = None,
    ) -> None:
        if bit_width < _MIN_BIT_WIDTH or bit_width > _MAX_BIT_WIDTH:
            raise InvalidBitWidthError(bit_width, (_MIN_BIT_WIDTH, _MAX_BIT_WIDTH))
        if mode not in _VALID_MODES:
            raise InvalidModeError(mode)
        if mode == "inner_product" and bit_width < 2:
            raise InvalidBitWidthError(bit_width, (2, _MAX_BIT_WIDTH))

        self._dim = dim
        self._bit_width = bit_width
        self._mode = mode
        self._seed = seed
        self._outlier_channels = outlier_channels
        self._outlier_bit_width = outlier_bit_width

        # Generate random rotation matrix
        self._rotation = generate_orthogonal_matrix(dim, seed=seed)

        # For inner_product mode, generate QJL projection matrix
        if mode == "inner_product":
            qjl_seed = seed + _QJL_SEED_OFFSET if seed is not None else None
            self._projection = generate_projection_matrix(rows=dim, cols=dim, seed=qjl_seed)

    @property
    def dim(self) -> int:
        """Input vector dimensionality."""
        return self._dim

    @property
    def bit_width(self) -> int:
        """Bits per quantized coordinate."""
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
            Input vectors of shape ``(n, dim)``.

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

        # Compute norms and normalize to unit length
        norms = np.linalg.norm(vectors, axis=1)
        safe_norms = np.where(norms > 0, norms, 1.0)
        unit_vectors = vectors / safe_norms[:, np.newaxis]

        if self._mode == "mse":
            return self._quantize_mse(unit_vectors, norms, self._bit_width)
        else:
            return self._quantize_inner_product(unit_vectors, norms)

    def _quantize_mse(
        self,
        unit_vectors: NDArray[np.float64],
        norms: NDArray[np.float64],
        bit_width: int,
    ) -> CompressedVectors:
        """MSE quantization: rotate, scalar quantize each coordinate."""
        # Rotate: y = Pi^T @ x^T => (dim, n), transpose to (n, dim)
        rotated = matmul(self._rotation.T, unit_vectors.T).T

        n = rotated.shape[0]
        extra_arrays: dict[str, NDArray] = {}
        metadata: dict[str, object] = {
            "mode": self._mode,
            "seed": self._seed,
            "outlier_channels": self._outlier_channels,
            "outlier_bit_width": self._outlier_bit_width,
        }

        if self._outlier_channels > 0 and self._outlier_bit_width is not None:
            # Detect outlier channels by average magnitude across the batch
            avg_magnitude = np.mean(np.abs(rotated), axis=0)
            outlier_indices = np.argsort(avg_magnitude)[-self._outlier_channels :]
            all_indices = np.arange(self._dim)
            inlier_mask = np.ones(self._dim, dtype=bool)
            inlier_mask[outlier_indices] = False
            inlier_indices = all_indices[inlier_mask]

            # Codebooks for inlier and outlier channels
            _, boundaries_in = get_codebook(self._dim, bit_width)
            _, boundaries_out = get_codebook(self._dim, self._outlier_bit_width)

            # Quantize all channels; outlier channels use higher bit-width codebook
            indices = np.empty((n, self._dim), dtype=np.uint8)
            indices[:, inlier_indices] = quantize_scalar(rotated[:, inlier_indices], boundaries_in)
            indices[:, outlier_indices] = quantize_scalar(
                rotated[:, outlier_indices], boundaries_out
            )

            extra_arrays["outlier_indices"] = outlier_indices.astype(np.int64)
            extra_arrays["inlier_indices"] = inlier_indices.astype(np.int64)
        else:
            # Standard path: all channels use the same codebook
            _, boundaries = get_codebook(self._dim, bit_width)
            indices = quantize_scalar(rotated, boundaries)

        return CompressedVectors(
            indices=indices,
            norms=norms,
            dim=self._dim,
            bit_width=bit_width,
            metadata=metadata,
            extra_arrays=extra_arrays,
        )

    def _quantize_inner_product(
        self,
        unit_vectors: NDArray[np.float64],
        norms: NDArray[np.float64],
    ) -> CompressedVectors:
        """Inner-product quantization: MSE at (b-1) bits + QJL on residual."""
        mse_bit_width = self._bit_width - 1

        # MSE quantize at (b-1) bits
        mse_compressed = self._quantize_mse(unit_vectors, norms, mse_bit_width)

        # Dequantize to get MSE reconstruction (unit-length domain)
        mse_reconstructed = self._dequantize_mse(mse_compressed, return_unit=True)

        # Compute residual in unit-vector space
        residual = unit_vectors - mse_reconstructed

        # Compute residual norms
        residual_norms = np.linalg.norm(residual, axis=1)

        # QJL on residual: project and take sign
        projected_residual = matmul(self._projection, residual.T).T  # (n, dim)
        residual_signs = np.sign(projected_residual).astype(np.int8)
        residual_signs[residual_signs == 0] = 1

        # Merge outlier arrays from MSE stage with QJL arrays
        merged_extra: dict[str, NDArray] = {}
        merged_extra.update(mse_compressed.extra_arrays)
        merged_extra["residual_signs"] = residual_signs
        merged_extra["residual_norms"] = residual_norms

        merged_metadata: dict[str, object] = {
            "mode": self._mode,
            "seed": self._seed,
        }
        merged_metadata.update(
            {k: v for k, v in mse_compressed.metadata.items() if k not in merged_metadata}
        )

        return CompressedVectors(
            indices=mse_compressed.indices,
            norms=norms,
            dim=self._dim,
            bit_width=self._bit_width,
            metadata=merged_metadata,
            extra_arrays=merged_extra,
        )

    def dequantize(self, compressed: CompressedVectors) -> NDArray[np.float64]:
        """Dequantize compressed vectors back to approximate originals.

        Parameters
        ----------
        compressed : CompressedVectors
            Compressed representation from ``quantize()``.

        Returns
        -------
        NDArray[np.float64]
            Reconstructed vectors of shape ``(n, dim)``.
        """
        if self._mode == "mse":
            return self._dequantize_mse(compressed, return_unit=False)
        else:
            # For inner_product mode, dequantize the MSE part
            return self._dequantize_mse(compressed, return_unit=False)

    def _dequantize_mse(
        self,
        compressed: CompressedVectors,
        return_unit: bool = False,
    ) -> NDArray[np.float64]:
        """Dequantize MSE-compressed vectors."""
        # Determine bit_width for codebook lookup
        bw = self._bit_width - 1 if self._mode == "inner_product" else self._bit_width

        if "outlier_indices" in compressed.extra_arrays:
            outlier_indices = compressed.extra_arrays["outlier_indices"]
            inlier_indices = compressed.extra_arrays["inlier_indices"]
            outlier_bw = compressed.metadata["outlier_bit_width"]

            centroids_in, _ = get_codebook(self._dim, bw)
            centroids_out, _ = get_codebook(self._dim, outlier_bw)

            reconstructed_rotated = np.empty(compressed.indices.shape, dtype=np.float64)
            reconstructed_rotated[:, inlier_indices] = dequantize_scalar(
                compressed.indices[:, inlier_indices], centroids_in
            )
            reconstructed_rotated[:, outlier_indices] = dequantize_scalar(
                compressed.indices[:, outlier_indices], centroids_out
            )
        else:
            centroids, _ = get_codebook(self._dim, bw)
            reconstructed_rotated = dequantize_scalar(compressed.indices, centroids)

        # Rotate back: x_hat = Pi @ y
        reconstructed = matmul(self._rotation, reconstructed_rotated.T).T

        if return_unit:
            return reconstructed

        # Scale by original norms
        return reconstructed * compressed.norms[:, np.newaxis]

    def inner_product(
        self,
        query: NDArray[np.float64],
        compressed: CompressedVectors,
    ) -> NDArray[np.float64]:
        """Estimate inner products between a query and compressed vectors.

        Uses the TurboQuant inner-product estimator:
            <y, x_mse> + ||r|| * sqrt(pi/2) / d * <S*y, sign(S*r)>

        Parameters
        ----------
        query : NDArray[np.float64]
            Query vector of shape ``(dim,)``.
        compressed : CompressedVectors
            Compressed vectors from ``quantize()``.

        Returns
        -------
        NDArray[np.float64]
            Estimated inner products of shape ``(n,)``.

        Raises
        ------
        DimensionMismatchError
            If query has wrong dimensionality.
        """
        query = np.asarray(query, dtype=np.float64)
        if query.shape[-1] != self._dim:
            raise DimensionMismatchError(expected=self._dim, got=query.shape[-1])

        if self._mode == "mse":
            # For MSE mode, dequantize and compute inner products directly
            reconstructed = self.dequantize(compressed)
            return batch_inner_product(query, reconstructed)

        # Inner product mode: MSE part + QJL residual correction
        mse_reconstructed = self._dequantize_mse(compressed, return_unit=True)
        mse_full = mse_reconstructed * compressed.norms[:, np.newaxis]
        mse_scores = batch_inner_product(query, mse_full)

        # QJL correction on residual
        residual_signs = compressed.extra_arrays["residual_signs"]  # (n, dim)
        residual_norms = compressed.extra_arrays["residual_norms"]  # (n,)

        projected_query = matmul(self._projection, query)  # (dim,)
        dot_products = batch_inner_product(projected_query, residual_signs.astype(np.float64))

        d = self._dim
        scale = np.sqrt(np.pi / 2) / d
        qjl_scores = compressed.norms * residual_norms * scale * dot_products

        return mse_scores + qjl_scores

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
        output_path = Path(output_path)
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
