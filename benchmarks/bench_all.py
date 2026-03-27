"""Performance benchmarks for TurboQuant library.

Run with: python benchmarks/bench_all.py

Measures quantization throughput, search throughput, memory footprint,
and distortion across dimensions, bit-widths, and collection sizes.
Compares NumPy vs PyTorch when available.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass

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
    mem = compressed.indices.nbytes + compressed.norms.nbytes
    for arr in compressed.extra_arrays.values():
        mem += arr.nbytes
    return mem


def run_turboquant_benchmarks() -> list[BenchmarkResult]:
    results = []
    dims = [384, 768, 1536, 3072]
    n_vectors_list = [1_000, 10_000]
    bit_widths = [2, 3, 4]

    for dim in dims:
        for n_vectors in n_vectors_list:
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
                    f"mem={mem / 1024 / 1024:7.2f}MB mse={mse:.6f}"
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
                f"mem={mem / 1024 / 1024:7.2f}MB"
            )

    return results


def save_results(
    results: list[BenchmarkResult],
    path: str | pathlib.Path = "examples/results/benchmark_results.json",
) -> None:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {out}")


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

    all_results = qjl_results + tq_results
    save_results(all_results)


if __name__ == "__main__":
    main()
