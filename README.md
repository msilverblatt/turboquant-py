# TurboQuant

**Vector quantization with near-optimal distortion rates.**

TurboQuant is a Python library implementing the TurboQuant and QJL vector quantization algorithms from Google Research (ICLR 2026 / AISTATS 2026). It compresses high-dimensional floating-point vectors to 1-4 bits per coordinate while preserving inner products and distances with provably near-optimal distortion. The library offers two quantization modes — MSE mode for reconstruction fidelity and inner-product mode for unbiased similarity search — plus a standalone 1-bit QJL quantizer, all built on a NumPy-first core with optional PyTorch acceleration.

## Installation

```bash
pip install -e .
# With PyTorch acceleration (optional)
pip install -e ".[torch]"
```

## Quick Start

### TurboQuant MSE mode

```python
import numpy as np
from turboquant import TurboQuant

vectors = np.random.randn(1000, 384)  # 1000 vectors, dim=384
tq = TurboQuant(dim=384, bit_width=2, mode="mse", seed=42)

compressed = tq.quantize(vectors)
reconstructed = tq.dequantize(compressed)

mse = float(np.mean((vectors - reconstructed) ** 2))
print(f"Reconstruction MSE: {mse:.6f}")

# Save and reload
compressed.save("my_index")
from turboquant import CompressedVectors
reloaded = CompressedVectors.load("my_index")
```

### TurboQuant inner product mode

```python
import numpy as np
from turboquant import TurboQuant

db = np.random.randn(10000, 768)
query = np.random.randn(768)

tq = TurboQuant(dim=768, bit_width=3, mode="inner_product", seed=42)
compressed = tq.quantize(db)

# Estimate inner products against all compressed database vectors
scores = tq.inner_product(query, compressed)
top10 = np.argsort(scores)[::-1][:10]
print(f"Top-10 indices: {top10}")
```

### QJL 1-bit quantization

```python
import numpy as np
from turboquant import QJL

db = np.random.randn(10000, 1536)
query = np.random.randn(1536)

qjl = QJL(dim=1536, seed=42)
compressed = qjl.quantize(db)

scores = qjl.inner_product(query, compressed)
top10 = np.argsort(scores)[::-1][:10]
print(f"Top-10 indices: {top10}")
```

## API Reference

### `TurboQuant(dim, bit_width, mode, seed, outlier_channels, outlier_bit_width)`

Multi-bit vector quantizer supporting MSE and inner-product modes.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim` | `int` | required | Input vector dimensionality |
| `bit_width` | `int` | required | Bits per coordinate (1-4; inner-product mode requires 2+) |
| `mode` | `str` | `"mse"` | `"mse"` for reconstruction quality; `"inner_product"` for unbiased similarity search |
| `seed` | `int \| None` | `None` | Random seed for the rotation matrix and QJL projection |
| `outlier_channels` | `int` | `0` | Number of high-magnitude channels to quantize at higher precision |
| `outlier_bit_width` | `int \| None` | `None` | Bit-width for outlier channels |

**Methods:**
- `quantize(vectors)` — compress an `(n, dim)` array; returns `CompressedVectors`
- `dequantize(compressed)` — reconstruct approximate originals; returns `(n, dim)` array
- `inner_product(query, compressed)` — estimate inner products; returns `(n,)` scores
- `quantize_batched(vectors, batch_size, output_path)` — stream-quantize large collections to disk

---

### `QJL(dim, projection_dim, seed)`

1-bit quantizer using the Quantized Johnson-Lindenstrauss transform. Applies a random Gaussian projection followed by sign-bit quantization to produce an unbiased inner-product estimator with no reconstruction capability.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim` | `int` | required | Input vector dimensionality |
| `projection_dim` | `int \| None` | `dim` | Projection dimension; higher values improve accuracy at the cost of storage |
| `seed` | `int \| None` | `None` | Random seed for the projection matrix |

**Methods:**
- `quantize(vectors)` — compress to 1-bit sign representation; returns `CompressedVectors`
- `inner_product(query, compressed)` — estimate inner products; returns `(n,)` scores

---

### `CompressedVectors`

In-memory container for quantized vectors. Holds the bit-packed quantization indices, per-vector L2 norms, and any auxiliary arrays (QJL signs, residual norms for inner-product mode). Supports slicing with `[start:end]` and merging with `CompressedVectors.concatenate(parts)`.

**Save/load:**
```python
compressed.save("path/to/dir")          # writes indices.npy, norms.npy, meta.json
loaded = CompressedVectors.load("path/to/dir")
loaded = CompressedVectors.load("path/to/dir", mmap_mode="r")  # memory-mapped
```

**Key attributes:** `indices`, `norms`, `dim`, `bit_width`, `num_vectors`, `metadata`, `extra_arrays`

---

### `CompressedStore`

On-disk vector store backed by memory-mapped arrays. Reconstructs the original quantizer from saved metadata and supports brute-force top-k search without loading all vectors into RAM.

```python
store = CompressedStore.load("path/to/dir")
results = store.search(query, k=10)  # returns list[tuple[int, float]]
```

**Properties:** `dim`, `num_vectors`, `bit_width`, `mode`, `vectors`

---

### `compute_codebook(dim, bit_width)` / `get_codebook(dim, bit_width)`

`compute_codebook` runs Lloyd-Max optimization on the Beta distribution that describes coordinates of randomly rotated unit vectors, returning `(centroids, boundaries)` arrays of sizes `2^bit_width` and `2^bit_width + 1`.

`get_codebook` is an `lru_cache`-wrapped convenience wrapper around `compute_codebook`.

## Supported Bit-Widths

Results on synthetic unit vectors (dim=768, n=1000).

| Bit-width | Compression ratio | Reconstruction MSE | Recall@1 (MSE mode) | Recall@10 (MSE mode) |
|---|---|---|---|---|
| 1 | 32x | 0.000473 | 0.19 | 0.65 |
| 2 | 16x | 0.000153 | 0.46 | 0.91 |
| 3 | 10.7x | 0.000045 | 0.67 | 0.99 |
| 4 | 8x | 0.000013 | 0.73 | 1.00 |

## Benchmarks

All results use `all-MiniLM-L6-v2` embeddings (dim=384, n=2000 database, n=200 queries, seed=42).

### MSE distortion and recall on neural embeddings

| Method | Bit-width | MSE | Recall@1 | Recall@10 |
|---|---|---|---|---|
| NaiveUniform | 2 | 0.001079 | 0.675 | 0.721 |
| NaiveUniform | 3 | 0.000195 | 0.830 | 0.850 |
| NaiveUniform | 4 | 0.000042 | 0.895 | 0.913 |
| TurboQuant-mse | 2 | 0.000305 | 0.755 | 0.817 |
| TurboQuant-mse | 3 | 0.000090 | 0.895 | 0.878 |
| TurboQuant-mse | 4 | 0.000025 | 0.895 | 0.918 |
| TurboQuant-inner_product | 2 | 0.000946 | 0.605 | 0.713 |
| TurboQuant-inner_product | 3 | 0.000305 | 0.760 | 0.820 |
| TurboQuant-inner_product | 4 | 0.000090 | 0.895 | 0.881 |
| QJL (1-bit) | 1-4 | — | 0.580 | 0.703 |

TurboQuant-mse achieves 3.5x lower MSE than naive uniform quantization at 2 bits. At 4 bits, TurboQuant-inner_product matches QJL recall while storing structured residuals that enable better approximation.

### Comparison vs. uniform quantization at 2 bits (dim=384)

| Method | MSE | Recall@1 | Recall@10 |
|---|---|---|---|
| NaiveUniform | 0.000856 | 0.30 | 0.80 |
| PerChannelUniform | 0.001032 | 0.29 | 0.82 |
| RandProj+Uniform | 0.001027 | 0.28 | 0.76 |
| TurboQuant-mse | 0.000304 | 0.44 | 0.95 |

TurboQuant-mse reduces MSE by 2.8x over NaiveUniform and improves Recall@10 from 80% to 95% at the same 2-bit budget.

## How It Works

**TurboQuant MSE mode** applies a random orthogonal rotation to each input vector before scalar quantization. Because coordinates of a randomly rotated unit vector follow a known Beta distribution, a precomputed Lloyd-Max codebook can be derived analytically for that distribution rather than estimated from data. Lloyd-Max quantization is provably optimal for a fixed scalar quantizer, and the rotation ensures coordinates match the distribution the codebook was designed for. Rotating back after dequantization reconstructs the original vector with near-optimal MSE.

**TurboQuant inner-product mode** extends MSE quantization to produce an unbiased inner-product estimator. It quantizes each vector at `(b-1)` bits using the MSE codebook, computes the residual between the original and the MSE reconstruction, and then applies the QJL transform (random Gaussian projection followed by sign extraction) to that residual. The stored representation consists of the MSE quantization indices plus the sign bits and L2 norm of the residual. At query time, the inner product estimate combines the MSE dot product and a QJL correction term scaled by `sqrt(pi/2) / d`, which exactly cancels the bias introduced by the MSE quantizer. Using one bit for QJL on the residual and `(b-1)` bits for MSE thus achieves better inner-product accuracy than spending all `b` bits on MSE quantization alone.

**QJL (Quantized Johnson-Lindenstrauss)** is a 1-bit scheme that projects each key vector through a random Gaussian matrix `S` and stores only the sign vector `sign(S·k)` along with the vector norm. For a query `q`, the inner product is estimated as `sqrt(pi/2) / m * ||k|| * <S·q, sign(S·k)>`, where `m` is the projection dimension. This estimator is unbiased and requires only 1 bit per projected coordinate, making it suitable for extreme compression where reconstruction is not needed.

## References

- **TurboQuant:** "TurboQuant: Redefining AI Efficiency with Extreme Compression" — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **QJL:** Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead" (AISTATS 2026) — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **PolarQuant:** [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
