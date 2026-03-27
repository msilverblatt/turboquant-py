"""Compressed vector storage: in-memory container and on-disk persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from turboquant.qjl import QJL
    from turboquant.turboquant import TurboQuant

from turboquant._bitpack import pack_indices, unpack_indices
from turboquant._entropy import (
    build_huffman_table,
    compute_symbol_probabilities,
    huffman_decode,
    huffman_encode,
    table_from_serializable,
    table_to_serializable,
)
from turboquant.exceptions import StorageError

__all__ = ["CompressedStore", "CompressedVectors"]

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
    metadata : dict[str, Any] or None
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
        sliced_extras = {
            k: v[key] if v.shape[0] == self.num_vectors else v
            for k, v in self.extra_arrays.items()
        }
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
            extra_arrays=cls._merge_extra_arrays(parts),
        )

    @classmethod
    def _merge_extra_arrays(cls, parts: list[CompressedVectors]) -> dict[str, NDArray]:
        """Merge extra_arrays from multiple CompressedVectors.

        Per-vector arrays (shape[0] == num_vectors) are concatenated.
        Shared arrays (e.g. outlier_indices) are taken from the first part.
        """
        if not parts:
            return {}
        ref = parts[0]
        merged: dict[str, NDArray] = {}
        for k in ref.extra_arrays:
            if not all(k in p.extra_arrays for p in parts):
                continue
            if ref.extra_arrays[k].shape[0] == ref.num_vectors:
                merged[k] = np.concatenate([p.extra_arrays[k] for p in parts], axis=0)
            else:
                # Shared array (same across all parts) — validate consistency
                for i, p in enumerate(parts[1:], 1):
                    if not np.array_equal(p.extra_arrays[k], ref.extra_arrays[k]):
                        raise ValueError(
                            f"Shared extra array '{k}' differs between parts 0 and {i}"
                        )
                merged[k] = ref.extra_arrays[k].copy()
        return merged

    def save(self, path: str | Path, entropy_encode: bool = False) -> None:
        """Save to a directory on disk.

        Parameters
        ----------
        path : str or Path
            Directory path. Created if it does not exist.
        entropy_encode : bool
            If True, apply Huffman entropy encoding to the indices for
            additional lossless compression.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        indices_shape = self.indices.shape
        # When outlier channels use a higher bit-width, pack at that width
        # so outlier indices are not truncated.
        outlier_bw = self.metadata.get("outlier_bit_width")
        pack_bw = max(self.bit_width, outlier_bw) if outlier_bw else self.bit_width

        if entropy_encode:
            # Use pack_bw to cover all possible index values (including outlier channels)
            probs = compute_symbol_probabilities(self.dim, pack_bw)
            huffman_table = build_huffman_table(probs)

            # Encode indices
            encoded_data = huffman_encode(self.indices.ravel(), huffman_table)

            # Write encoded indices as raw bytes
            with open(path / "indices.huffman", "wb") as f:
                f.write(encoded_data)

            # Write Huffman table
            with open(path / "huffman_table.json", "w") as f:
                json.dump(table_to_serializable(huffman_table), f, indent=2)
        else:
            if pack_bw < 8:
                flat = self.indices.ravel()
                packed = pack_indices(flat, pack_bw)
            else:
                packed = self.indices
            np.save(path / "indices.npy", packed)

        # User metadata goes first so reserved keys always win
        meta = {
            **self.metadata,
            "dim": self.dim,
            "bit_width": self.bit_width,
            "num_vectors": self.num_vectors,
            "indices_packed": pack_bw < 8 and not entropy_encode,
            "indices_pack_width": pack_bw,
            "indices_shape": list(indices_shape),
            "extra_array_names": list(self.extra_arrays.keys()),
            "entropy_encoded": entropy_encode,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

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
        meta.pop("num_vectors", None)
        is_packed = meta.pop("indices_packed", False)
        pack_bw = meta.pop("indices_pack_width", bit_width)
        indices_shape = meta.pop("indices_shape", None)
        extra_names = meta.pop("extra_array_names", [])
        is_entropy_encoded = meta.pop("entropy_encoded", False)

        if is_entropy_encoded and indices_shape is not None:
            # Load Huffman-encoded indices
            huffman_path = path / "huffman_table.json"
            if not huffman_path.exists():
                raise StorageError(f"Missing huffman_table.json in {path}")
            with open(huffman_path) as f:
                huffman_table = table_from_serializable(json.load(f))

            with open(path / "indices.huffman", "rb") as f:
                encoded_data = f.read()

            n_values = int(np.prod(indices_shape))
            indices = huffman_decode(encoded_data, huffman_table, n_values).reshape(indices_shape)
        else:
            raw = np.load(path / "indices.npy", mmap_mode=mmap_mode)
            if is_packed and indices_shape is not None:
                n_values = int(np.prod(indices_shape))
                # mmap arrays are read-only; make a writable copy for unpack
                raw_arr = np.array(raw)
                indices = unpack_indices(raw_arr, pack_bw, n_values).reshape(indices_shape)
            else:
                indices = np.array(raw) if mmap_mode else raw
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

    def _build_quantizer(self) -> TurboQuant | QJL:
        """Reconstruct the quantizer from saved metadata."""
        meta = self._vectors.metadata
        mode = meta.get("mode", "unknown")
        seed = meta.get("seed")

        if mode == "qjl":
            from turboquant.qjl import QJL

            projection_dim = meta.get("projection_dim", self._vectors.dim)
            return QJL(dim=self._vectors.dim, projection_dim=projection_dim, seed=seed)

        if mode in {"mse", "inner_product"}:
            from turboquant.turboquant import TurboQuant

            return TurboQuant(
                dim=self._vectors.dim,
                bit_width=self._vectors.bit_width,
                mode=mode,
                seed=seed,
                outlier_channels=meta.get("outlier_channels", 0),
                outlier_bit_width=meta.get("outlier_bit_width"),
            )

        raise StorageError(f"Cannot reconstruct quantizer for unknown mode: {mode!r}")

    def search(
        self,
        query: NDArray[np.float64],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search for the top-k most similar vectors.

        Parameters
        ----------
        query : NDArray[np.float64]
            Query vector of shape ``(dim,)``.
        k : int
            Number of results to return. If k exceeds the number of stored
            vectors, all vectors are returned.

        Returns
        -------
        list[tuple[int, float]]
            Top-k ``(index, score)`` pairs sorted by descending score.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        query = np.asarray(query, dtype=np.float64)
        quantizer = self._build_quantizer()
        scores = quantizer.inner_product(query, self._vectors)

        effective_k = min(k, self._vectors.num_vectors)
        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]
