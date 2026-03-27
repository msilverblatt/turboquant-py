# TurboQuant Library Design Spec

## Overview

A standalone Python library implementing the TurboQuant and QJL vector quantization algorithms from the Google Research papers (ICLR 2026 / AISTATS 2026). The library provides general-purpose vector quantization primitives — quantize, dequantize, and inner product estimation — suitable for any downstream use case including RAG, agent memory, nearest-neighbor search, and KV cache compression.

## Decisions

- **Language:** Python (NumPy-first, optional PyTorch acceleration)
- **Algorithms:** TurboQuant (MSE + inner-product modes) and standalone QJL (1-bit)
- **Bit-widths:** 1, 2, 3, 4 bits per coordinate
- **Dimensions:** Arbitrary, validated against 384, 512, 768, 1024, 1536, 2048, 3072
- **Scale:** Small to large collections (millions+) via batched quantization and memory-mapped storage
- **No index structures** in v1 — brute-force scan over compressed vectors

## Public API

### QJL (1-bit quantization)

```python
from turboquant import QJL

qjl = QJL(dim: int, projection_dim: int | None = None, seed: int | None = None)
# projection_dim defaults to dim (same dimensionality as input).
# Higher projection_dim = better accuracy but more storage.
# Paper guarantees: m >= (4/3) · (1+ε)/ε² · log(2/δ) for (1±3ε) relative distortion on attention scores.

# Quantize a collection of vectors
compressed: CompressedVectors = qjl.quantize(vectors: np.ndarray)  # (n, dim) → CompressedVectors

# Estimate inner products between a query and compressed vectors
scores: np.ndarray = qjl.inner_product(query: np.ndarray, compressed: CompressedVectors)  # (dim,) → (n,)
```

### TurboQuant (multi-bit quantization)

```python
from turboquant import TurboQuant

tq = TurboQuant(
    dim: int,
    bit_width: int,
    mode: Literal["mse", "inner_product"] = "inner_product",
    seed: int | None = None,
    outlier_channels: int = 0,          # number of outlier channels (0 = no outlier handling)
    outlier_bit_width: int | None = None,  # bit-width for outlier channels (defaults to bit_width + 1)
)
# mode="mse" — TurboQuant_mse, optimized for reconstruction quality
# mode="inner_product" — TurboQuant_prod, unbiased inner product estimation
# outlier_channels > 0 enables mixed-precision quantization per the paper's outlier strategy

# Quantize
compressed: CompressedVectors = tq.quantize(vectors: np.ndarray)  # (n, dim) → CompressedVectors

# Dequantize (reconstruct approximate original vectors)
reconstructed: np.ndarray = tq.dequantize(compressed: CompressedVectors)  # → (n, dim)

# Estimate inner products
scores: np.ndarray = tq.inner_product(query: np.ndarray, compressed: CompressedVectors)  # (dim,) → (n,)

# Batch/streaming quantization for large collections
tq.quantize_batched(
    vectors: Iterable[np.ndarray],
    batch_size: int = 10_000,
    output_path: str | Path = "index.tqz",
)
```

### Storage

```python
from turboquant import CompressedStore

# Load a memory-mapped compressed store
store = CompressedStore.load(path: str | Path)

# Search: top-k by estimated inner product
results: list[tuple[int, float]] = store.search(query: np.ndarray, k: int = 10)

# Metadata access
store.dim -> int
store.num_vectors -> int
store.bit_width -> int
store.mode -> str
```

### CompressedVectors (in-memory container)

```python
from turboquant import CompressedVectors

compressed.indices      # bit-packed quantization indices
compressed.norms        # per-vector norms
compressed.num_vectors  # number of vectors
compressed.dim          # original dimensionality
compressed.bit_width    # quantization bit-width
compressed.metadata     # dict of additional metadata

# Slicing and concatenation
subset = compressed[10:20]
merged = CompressedVectors.concatenate([compressed_a, compressed_b])

# Persistence
compressed.save(path)
loaded = CompressedVectors.load(path)
```

## Core Algorithm Implementation

### QJL

1. Generate random projection matrix `S ∈ R^{m×d}` with i.i.d. N(0,1) entries, orthogonalized via QR decomposition.
2. **Quantize:** `sign(S · k) → {-1, +1}^m`, store alongside `‖k‖₂`.
3. **Inner product estimator:** `√(π/2) / m · ‖k‖₂ · ⟨S·q, sign(S·k)⟩`.
4. The projection matrix `S` is shared across all vectors in a collection. It is generated deterministically from a seed and stored with the compressed data.

### TurboQuant_mse

1. Generate random rotation matrix `Π ∈ R^{d×d}` via QR decomposition of a Gaussian matrix.
2. Rotate input: `y = Π · x`.
3. Each coordinate of the rotated vector follows a Beta distribution (converging to Gaussian N(0, 1/d) in high dimensions). Quantize each coordinate independently using a precomputed Lloyd-Max codebook for this distribution.
4. Store b-bit codebook indices per coordinate.
5. **Dequantize:** look up centroids from codebook, rotate back with `Π⊤`.

### TurboQuant_prod

1. Apply TurboQuant_mse at bit-width `(b - 1)`.
2. Compute residual: `r = x - dequantize(quantize_mse(x))`.
3. Apply QJL to residual: store `sign(S · r)` and `‖r‖₂`.
4. **Inner product estimator:** `⟨y, x̃_mse⟩ + ‖r‖₂ · √(π/2) / d · ⟨S·y, sign(S·r)⟩`.
5. This yields an unbiased estimator (the QJL on the residual corrects the bias inherent in MSE-optimal quantizers).

### Codebook Precomputation

- Solve the 1D continuous k-means problem (Lloyd-Max algorithm) for the Beta distribution `f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)` at each bit-width (1-4) and a range of representative dimensions.
- For high dimensions, the distribution converges to Gaussian, so codebooks stabilize.
- Precomputed codebooks are stored as NumPy `.npy` arrays shipped with the package.

### Outlier Handling

- Detect outlier channels by magnitude across a calibration sample (or the vectors being quantized).
- Quantize outlier and non-outlier channels with separate quantizer instances at different bit-widths.
- This enables non-integer effective bit-widths (e.g., 2.5-bit: 32 outlier channels at 3 bits, 96 channels at 2 bits for d=128).

## Storage Format

### On-disk layout (`.tqz` directory)

```
index.tqz/
├── meta.json           # dimensions, bit_width, mode, num_vectors, outlier config, seed
├── indices.npy         # bit-packed quantization indices (memory-mappable)
├── norms.npy           # per-vector L2 norms
├── rotation.npy        # rotation matrix Π
├── projection.npy      # QJL projection matrix S (if mode="inner_product" or QJL)
├── qjl_signs.npy       # QJL sign bits for residuals (if mode="inner_product")
└── qjl_norms.npy       # residual norms (if mode="inner_product")
```

All `.npy` files are memory-mappable via `np.load(..., mmap_mode='r')`.

### Batch quantization

- `quantize_batched()` processes vectors in chunks, appending to the on-disk store progressively.
- Rotation/projection matrices are generated once upfront from the seed and reused across all batches.
- Search scans in batches to keep memory bounded.

## Acceleration

- All core operations are implemented in NumPy.
- At runtime, if PyTorch is importable, the following operations dispatch to PyTorch tensors for acceleration:
  - Random rotation / projection matrix generation
  - Matrix-vector and matrix-matrix multiplications (rotation, projection)
  - Batch inner product computation during search
- Detection is via a simple `try: import torch` check, not a plugin/backend interface.
- A single internal module (`_accel.py`) encapsulates all dispatch logic.

## Package Structure

```
turboquant/
├── pyproject.toml
├── src/
│   └── turboquant/
│       ├── __init__.py         # public API re-exports, __all__
│       ├── py.typed            # PEP 561 marker
│       ├── qjl.py              # QJL quantizer
│       ├── turboquant.py       # TurboQuant MSE and inner-product quantizers
│       ├── codebook.py         # Lloyd-Max codebook computation and loading
│       ├── storage.py          # CompressedVectors, CompressedStore, serialization
│       ├── _accel.py           # NumPy/PyTorch dispatch logic
│       ├── exceptions.py       # TurboQuantError, DimensionMismatchError, InvalidBitWidthError, etc.
│       └── codebooks/          # precomputed codebook .npy arrays
│           └── ...
├── tests/
│   ├── test_qjl.py
│   ├── test_turboquant.py
│   ├── test_codebook.py
│   ├── test_storage.py
│   └── test_integration.py
└── benchmarks/
    └── bench_all.py
```

## Code Quality Standards

- Type annotations on all functions and class attributes
- Docstrings on all public API (NumPy-style)
- `py.typed` marker for PEP 561 compliance
- `__all__` exports in every module
- Clean public API re-exported from `turboquant.__init__`
- Custom exception hierarchy rooted at `TurboQuantError`
- Logging via stdlib `logging` (no print statements)
- Linting: ruff
- Formatting: ruff format
- 100% of public API covered by tests

## Dependencies

### Required

- Python >= 3.10
- NumPy
- SciPy

### Optional

- PyTorch (detected at runtime for acceleration)

### Dev

- pytest
- pytest-benchmark

## Testing

### Correctness

- **Codebook validation:** verify precomputed centroids match the Lloyd-Max solution for the Beta distribution at each bit-width across representative dimensions.
- **Unbiasedness:** for TurboQuant_prod and QJL, verify the inner product estimator is unbiased over many random trials (mean error converges to 0).
- **Distortion bounds:** verify MSE and inner product distortion fall within the theoretical bounds from the papers (Theorems 1-3 of the TurboQuant paper).
- **Round-trip:** quantize → dequantize, verify reconstruction error decreases monotonically with increasing bit-width.
- **Determinism:** same seed produces identical results.

### Dimension coverage

Test across: 384, 512, 768, 1024, 1536, 2048, 3072.

### Storage

- Write → load → search round-trip.
- Memory-mapped search on collections exceeding a configured memory limit.
- Batch quantization produces identical results to single-batch quantization.

## Benchmarks

- NumPy vs PyTorch for quantization and inner product estimation.
- Sweep across dimensions (384 → 3072) and collection sizes (1K, 10K, 100K, 1M).
- Metrics: quantization throughput (vectors/sec), search throughput (queries/sec), peak memory footprint, distortion at each bit-width.
- Output as formatted tables and optionally as matplotlib plots.

## Out of Scope for v1

- Index structures (HNSW, IVF, etc.) — brute-force scan only
- Formal backend/plugin interface for compute backends
- CUDA kernels
- Go bindings or gRPC wrapper
- PolarQuant algorithm
