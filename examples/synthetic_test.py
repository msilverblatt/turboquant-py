"""Synthetic benchmark for TurboQuant and QJL quantizers.

Tests reconstruction MSE, inner product accuracy, and nearest-neighbor recall
at multiple dimensions and bit-widths. Saves results to examples/results/synthetic_results.json.

Usage:
    uv run python examples/synthetic_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure the src directory is on the path when run directly
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from turboquant import QJL, TurboQuant  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_DB = 1000
N_QUERIES = 100
DIMENSIONS = [384, 768, 1536]
TURBOQUANT_BIT_WIDTHS = [1, 2, 3, 4]         # MSE supports all; inner_product needs ≥2
TURBOQUANT_IP_BIT_WIDTHS = [2, 3, 4]         # inner_product mode minimum is 2
RECALL_KS = [1, 5, 10]
FLOAT32_BITS = 32

RESULTS_PATH = _REPO_ROOT / "examples" / "results" / "synthetic_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Return n unit-normalized Gaussian random vectors of shape (n, dim)."""
    v = rng.standard_normal((n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def true_inner_products(db: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Exact inner products: shape (n_queries, n_db)."""
    return queries @ db.T


def true_nearest_neighbors(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k for each query row (descending)."""
    return np.argsort(scores, axis=1)[:, ::-1][:, :k]


def recall_at_k(true_nn: np.ndarray, approx_scores: np.ndarray, k: int) -> float:
    """Fraction of queries whose true top-1 neighbour appears in the approx top-k."""
    true_top1 = true_nn[:, 0]                        # (n_queries,)
    approx_topk = np.argsort(approx_scores, axis=1)[:, ::-1][:, :k]  # (n_queries, k)
    hits = np.any(approx_topk == true_top1[:, None], axis=1)
    return float(hits.mean())


def compression_ratio(bit_width: int) -> float:
    """Bits per coordinate / 32-bit float baseline."""
    return bit_width / FLOAT32_BITS


def print_table(title: str, headers: list[str], rows: list[list]) -> None:
    """Print a simple fixed-width ASCII table."""
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print(f"\n{title}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


# ── Benchmark routines ────────────────────────────────────────────────────────

def benchmark_qjl(db: np.ndarray, queries: np.ndarray, dim: int, rng_seed: int) -> dict:
    """Run QJL benchmark for one (dim,) config. Returns metrics dict."""
    qjl = QJL(dim=dim, seed=rng_seed)
    compressed = qjl.quantize(db)

    # Dequantize is not defined for QJL (1-bit sign only), so no MSE on reconstruction.
    # We report reconstruction MSE as N/A.

    exact_scores = true_inner_products(db, queries)          # (n_q, n_db)
    true_nn = true_nearest_neighbors(exact_scores, k=max(RECALL_KS))

    est_scores = np.array([qjl.inner_product(q, compressed) for q in queries])  # (n_q, n_db)
    errors = est_scores - exact_scores
    ip_bias = float(np.mean(errors))
    ip_var = float(np.var(errors))

    recalls = {k: recall_at_k(true_nn, est_scores, k) for k in RECALL_KS}
    comp = compression_ratio(1)

    return {
        "algorithm": "QJL",
        "dim": dim,
        "bit_width": 1,
        "mode": "sign",
        "reconstruction_mse": None,
        "ip_bias": round(ip_bias, 6),
        "ip_variance": round(ip_var, 6),
        "recall": {f"@{k}": round(v, 4) for k, v in recalls.items()},
        "compression_ratio": round(comp, 6),
        "bits_per_coord": 1,
    }


def benchmark_turboquant(
    db: np.ndarray,
    queries: np.ndarray,
    dim: int,
    bit_width: int,
    mode: str,
    rng_seed: int,
) -> dict:
    """Run TurboQuant benchmark for one config. Returns metrics dict."""
    tq = TurboQuant(dim=dim, bit_width=bit_width, mode=mode, seed=rng_seed)
    compressed = tq.quantize(db)

    # Reconstruction MSE (on unit vectors — db is already unit-normalised)
    reconstructed = tq.dequantize(compressed)
    mse = float(np.mean((db - reconstructed) ** 2))

    # Inner product accuracy
    exact_scores = true_inner_products(db, queries)          # (n_q, n_db)
    true_nn = true_nearest_neighbors(exact_scores, k=max(RECALL_KS))

    est_scores = np.array([tq.inner_product(q, compressed) for q in queries])  # (n_q, n_db)
    errors = est_scores - exact_scores
    ip_bias = float(np.mean(errors))
    ip_var = float(np.var(errors))

    recalls = {k: recall_at_k(true_nn, est_scores, k) for k in RECALL_KS}
    comp = compression_ratio(bit_width)

    return {
        "algorithm": "TurboQuant",
        "dim": dim,
        "bit_width": bit_width,
        "mode": mode,
        "reconstruction_mse": round(mse, 6),
        "ip_bias": round(ip_bias, 6),
        "ip_variance": round(ip_var, 6),
        "recall": {f"@{k}": round(v, 4) for k, v in recalls.items()},
        "compression_ratio": round(comp, 6),
        "bits_per_coord": bit_width,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(SEED)

    all_results: list[dict] = []

    for dim in DIMENSIONS:
        print(f"\n{'='*72}")
        print(f"  Dimension: {dim}   (DB={N_DB} vectors, Queries={N_QUERIES})")
        print(f"{'='*72}")

        db = make_unit_vectors(rng, N_DB, dim)
        queries = make_unit_vectors(rng, N_QUERIES, dim)

        # ── QJL ──────────────────────────────────────────────────────────────
        print("  Running QJL (1-bit)...")
        r = benchmark_qjl(db, queries, dim, rng_seed=SEED)
        all_results.append(r)

        # ── TurboQuant MSE ────────────────────────────────────────────────────
        mse_rows = []
        for bw in TURBOQUANT_BIT_WIDTHS:
            print(f"  Running TurboQuant MSE  bit_width={bw}...")
            r = benchmark_turboquant(db, queries, dim, bw, "mse", rng_seed=SEED)
            all_results.append(r)
            mse_rows.append(r)

        # ── TurboQuant inner_product ──────────────────────────────────────────
        ip_rows = []
        for bw in TURBOQUANT_IP_BIT_WIDTHS:
            print(f"  Running TurboQuant IP   bit_width={bw}...")
            r = benchmark_turboquant(db, queries, dim, bw, "inner_product", rng_seed=SEED)
            all_results.append(r)
            ip_rows.append(r)

        # ── Print tables for this dimension ───────────────────────────────────
        recall_headers = ["Algorithm", "Mode", "Bits", "MSE", "IP Bias", "IP Var",
                          "R@1", "R@5", "R@10", "Compression"]

        def result_to_row(res: dict) -> list:
            mse_str = f"{res['reconstruction_mse']:.5f}" if res["reconstruction_mse"] is not None else "N/A"
            return [
                res["algorithm"],
                res["mode"],
                res["bit_width"],
                mse_str,
                f"{res['ip_bias']:.5f}",
                f"{res['ip_variance']:.5f}",
                f"{res['recall']['@1']:.3f}",
                f"{res['recall']['@5']:.3f}",
                f"{res['recall']['@10']:.3f}",
                f"{res['compression_ratio']:.4f}",
            ]

        # Gather QJL result for this dim
        qjl_result = next(
            r for r in all_results
            if r["algorithm"] == "QJL" and r["dim"] == dim
        )
        rows = (
            [result_to_row(qjl_result)]
            + [result_to_row(r) for r in mse_rows]
            + [result_to_row(r) for r in ip_rows]
        )
        print_table(f"  Results — dim={dim}", recall_headers, rows)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"seed": SEED, "n_db": N_DB, "n_queries": N_QUERIES, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Summary table across all dims ─────────────────────────────────────────
    print_table(
        "Summary — Recall@1 across all configurations",
        ["Algorithm", "Mode", "Bits", "dim=384", "dim=768", "dim=1536"],
        _build_summary(all_results),
    )


def _build_summary(results: list[dict]) -> list[list]:
    """Build summary rows grouped by (algorithm, mode, bit_width)."""
    from collections import defaultdict

    groups: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in results:
        key = (r["algorithm"], r["mode"], r["bit_width"])
        groups[key][r["dim"]] = r["recall"]["@1"]

    rows = []
    for key, dim_map in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        algo, mode, bw = key
        rows.append([
            algo,
            mode,
            bw,
            f"{dim_map.get(384, float('nan')):.3f}",
            f"{dim_map.get(768, float('nan')):.3f}",
            f"{dim_map.get(1536, float('nan')):.3f}",
        ])
    return rows


if __name__ == "__main__":
    main()
