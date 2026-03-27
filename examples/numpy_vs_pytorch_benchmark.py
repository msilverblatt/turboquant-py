"""NumPy vs PyTorch acceleration benchmark for TurboQuant.

Compares wall-clock time for TurboQuant (MSE, 3-bit) and QJL operations
when dispatching through NumPy vs PyTorch via the _accel layer.

Usage:
    uv run python examples/numpy_vs_pytorch_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import turboquant._accel as _accel
from turboquant import QJL, TurboQuant

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
DIMENSIONS = [384, 768, 1536]
N_VECTORS_LIST = [1000, 10000]
N_QUERIES = 100
TQ_BIT_WIDTH = 3
N_REPEATS = 3

RESULTS_PATH = _REPO_ROOT / "examples" / "results" / "numpy_vs_pytorch_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    v = rng.standard_normal((n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def time_fn(fn, n_repeats: int) -> float:
    """Run fn() n_repeats times and return the minimum elapsed seconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def force_numpy():
    """Monkeypatch _accel.has_torch to always return False."""
    _accel.has_torch.cache_clear()
    _accel.has_torch = lru_cache(maxsize=1)(lambda: False)


def restore_pytorch():
    """Restore _accel.has_torch to the real implementation (returns True)."""
    _accel.has_torch = lru_cache(maxsize=1)(lambda: True)
    _accel.has_torch.cache_clear()


def print_table(headers: list[str], rows: list[list]) -> None:
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmarks() -> list[dict]:
    rng = np.random.default_rng(SEED)
    all_results: list[dict] = []

    for dim in DIMENSIONS:
        for n_vectors in N_VECTORS_LIST:
            print(f"\n  dim={dim}, n_vectors={n_vectors}")

            db = make_vectors(rng, n_vectors, dim)
            queries = make_vectors(rng, N_QUERIES, dim)

            # Pre-build quantizers (not timed — same objects used in both modes)
            tq = TurboQuant(dim=dim, bit_width=TQ_BIT_WIDTH, mode="mse", seed=SEED)
            qjl = QJL(dim=dim, seed=SEED)

            for backend_label, setup_fn in [("numpy", force_numpy), ("pytorch", restore_pytorch)]:
                setup_fn()

                # ── TurboQuant quantize ──────────────────────────────────────
                t_tq_quantize = time_fn(lambda: tq.quantize(db), N_REPEATS)
                compressed_tq = tq.quantize(db)

                # ── TurboQuant dequantize ────────────────────────────────────
                t_tq_dequantize = time_fn(lambda: tq.dequantize(compressed_tq), N_REPEATS)

                # ── TurboQuant inner_product (100 queries) ───────────────────
                def _tq_ip():
                    for q in queries:
                        tq.inner_product(q, compressed_tq)

                t_tq_ip = time_fn(_tq_ip, N_REPEATS)

                # ── QJL quantize ─────────────────────────────────────────────
                t_qjl_quantize = time_fn(lambda: qjl.quantize(db), N_REPEATS)
                compressed_qjl = qjl.quantize(db)

                # ── QJL inner_product (100 queries) ──────────────────────────
                def _qjl_ip():
                    for q in queries:
                        qjl.inner_product(q, compressed_qjl)

                t_qjl_ip = time_fn(_qjl_ip, N_REPEATS)

                for op_label, elapsed in [
                    ("tq_quantize", t_tq_quantize),
                    ("tq_dequantize", t_tq_dequantize),
                    ("tq_inner_product", t_tq_ip),
                    ("qjl_quantize", t_qjl_quantize),
                    ("qjl_inner_product", t_qjl_ip),
                ]:
                    all_results.append({
                        "dim": dim,
                        "n_vectors": n_vectors,
                        "operation": op_label,
                        "backend": backend_label,
                        "time_s": elapsed,
                    })

                print(f"    [{backend_label:8s}] tq_quantize={t_tq_quantize:.4f}s  "
                      f"tq_dequantize={t_tq_dequantize:.4f}s  tq_ip={t_tq_ip:.4f}s  "
                      f"qjl_quantize={t_qjl_quantize:.4f}s  qjl_ip={t_qjl_ip:.4f}s")

    return all_results


def build_comparison_rows(results: list[dict]) -> list[dict]:
    """Pair numpy and pytorch timings and compute speedup."""
    from collections import defaultdict

    # Key: (dim, n_vectors, operation) -> {backend: time}
    lookup: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in results:
        key = (r["dim"], r["n_vectors"], r["operation"])
        lookup[key][r["backend"]] = r["time_s"]

    rows = []
    for (dim, n, op), times in sorted(lookup.items()):
        numpy_t = times.get("numpy")
        pytorch_t = times.get("pytorch")
        speedup = (numpy_t / pytorch_t) if (numpy_t and pytorch_t and pytorch_t > 0) else None
        rows.append({
            "dim": dim,
            "n_vectors": n,
            "operation": op,
            "numpy_time_s": round(numpy_t, 6) if numpy_t is not None else None,
            "pytorch_time_s": round(pytorch_t, 6) if pytorch_t is not None else None,
            "speedup": round(speedup, 3) if speedup is not None else None,
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version:   {np.__version__}")

    print("\nRunning benchmarks (this may take a minute)...")
    raw_results = run_benchmarks()

    # Restore PyTorch for safety before building comparison
    restore_pytorch()

    comparison = build_comparison_rows(raw_results)

    # ── Print formatted table ──────────────────────────────────────────────────
    print("\n\nComparison Table")
    headers = ["dim", "n", "operation", "numpy_time_s", "pytorch_time_s", "speedup"]
    table_rows = [
        [
            r["dim"],
            r["n_vectors"],
            r["operation"],
            f"{r['numpy_time_s']:.6f}" if r["numpy_time_s"] is not None else "N/A",
            f"{r['pytorch_time_s']:.6f}" if r["pytorch_time_s"] is not None else "N/A",
            f"{r['speedup']:.3f}x" if r["speedup"] is not None else "N/A",
        ]
        for r in comparison
    ]
    print_table(headers, table_rows)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "seed": SEED,
        "n_queries": N_QUERIES,
        "tq_bit_width": TQ_BIT_WIDTH,
        "n_repeats": N_REPEATS,
        "raw": raw_results,
        "comparison": comparison,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
