# Changelog

## 0.1.0 (2026-03-27)

Initial release.

- TurboQuant MSE mode: random rotation + Lloyd-Max scalar quantization (1-4 bits)
- TurboQuant inner-product mode: MSE quantization + QJL residual correction
- QJL 1-bit quantizer: Quantized Johnson-Lindenstrauss transform
- CompressedVectors: in-memory container with save/load, slicing, concatenation
- CompressedStore: on-disk memory-mapped vector store with brute-force search
- Precomputed Lloyd-Max codebooks for dimensions 64-4096
- Optional Huffman entropy encoding for additional storage savings
- Outlier channel detection with per-channel higher-precision quantization
- Streaming batched quantization for large collections
- NumPy-first core with optional PyTorch acceleration (CUDA, Apple Silicon MPS)
