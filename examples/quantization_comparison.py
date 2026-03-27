"""Comparison benchmark: TurboQuant vs baseline quantization methods.

Compares five quantization methods across dimensions [384, 768, 1536] and
bit-widths [2, 3, 4] on 1000 unit vectors (seed=42).

Methods
-------
1. Naive Uniform      – per-vector min/max linear mapping
2. Per-channel Uniform – per-channel min/max across all vectors
3. Random Proj + Uniform – random rotation then uniform quantize
4. TurboQuant MSE     – library MSE mode
5. TurboQuant IP      – library inner_product mode

Metrics: reconstruction MSE, IP bias, IP variance, Recall@1/5/10.

Usage
-----
    uv run python examples/quantization_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from turboquant import TurboQuant  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_DB = 1000
N_QUERIES = 100
DIMENSIONS = [384, 768, 1536]
BIT_WIDTHS = [2, 3, 4]
RECALL_KS = [1, 5, 10]

RESULTS_PATH = _REPO_ROOT / "examples" / "results" / "quantization_comparison_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Return n unit-normalised Gaussian random vectors of shape (n, dim)."""
    v = rng.standard_normal((n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def true_inner_products(db: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Exact inner products, shape (n_queries, n_db)."""
    return queries @ db.T


def true_nearest_neighbors(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of top-k for each query row (descending)."""
    return np.argsort(scores, axis=1)[:, ::-1][:, :k]


def recall_at_k(true_nn: np.ndarray, approx_scores: np.ndarray, k: int) -> float:
    """Fraction of queries whose true top-1 neighbour appears in the approx top-k."""
    true_top1 = true_nn[:, 0]
    approx_topk = np.argsort(approx_scores, axis=1)[:, ::-1][:, :k]
    hits = np.any(approx_topk == true_top1[:, None], axis=1)
    return float(hits.mean())


def compute_ip_metrics(
    est_scores: np.ndarray,
    exact_scores: np.ndarray,
    true_nn: np.ndarray,
) -> tuple[float, float, dict[int, float]]:
    """Return (ip_bias, ip_variance, {k: recall}) from score matrices."""
    errors = est_scores - exact_scores
    ip_bias = float(np.mean(errors))
    ip_var = float(np.var(errors))
    recalls = {k: recall_at_k(true_nn, est_scores, k) for k in RECALL_KS}
    return ip_bias, ip_var, recalls


def build_result(
    method: str,
    dim: int,
    bit_width: int,
    mse: float,
    ip_bias: float,
    ip_var: float,
    recalls: dict[int, float],
) -> dict:
    return {
        "method": method,
        "dim": dim,
        "bit_width": bit_width,
        "reconstruction_mse": round(mse, 6),
        "ip_bias": round(ip_bias, 6),
        "ip_variance": round(ip_var, 6),
        "recall": {f"@{k}": round(v, 4) for k, v in recalls.items()},
    }


# ── Baseline: Naive Uniform Quantization ─────────────────────────────────────


def naive_uniform_quantize(vectors: np.ndarray, bit_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize each vector independently using its own [min, max].

    Returns
    -------
    indices : (n, dim) uint8
    mins    : (n,) per-vector minimum
    maxs    : (n,) per-vector maximum
    """
    n_levels = (1 << bit_width) - 1  # 2^b - 1
    mins = vectors.min(axis=1)        # (n,)
    maxs = vectors.max(axis=1)        # (n,)
    ranges = maxs - mins
    # Avoid division by zero for constant vectors
    safe_ranges = np.where(ranges > 0, ranges, 1.0)
    # Map [min, max] -> [0, n_levels]
    normalised = (vectors - mins[:, None]) / safe_ranges[:, None]
    indices = np.clip(np.round(normalised * n_levels), 0, n_levels).astype(np.uint8)
    return indices, mins, maxs


def naive_uniform_dequantize(indices: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bit_width: int) -> np.ndarray:
    n_levels = (1 << bit_width) - 1
    ranges = maxs - mins
    return indices.astype(np.float64) / n_levels * ranges[:, None] + mins[:, None]


def benchmark_naive_uniform(
    db: np.ndarray, queries: np.ndarray, bit_width: int
) -> tuple[float, float, float, dict[int, float]]:
    indices, mins, maxs = naive_uniform_quantize(db, bit_width)
    reconstructed = naive_uniform_dequantize(indices, mins, maxs, bit_width)
    mse = float(np.mean((db - reconstructed) ** 2))

    exact_scores = true_inner_products(db, queries)
    true_nn = true_nearest_neighbors(exact_scores, k=max(RECALL_KS))
    est_scores = queries @ reconstructed.T
    ip_bias, ip_var, recalls = compute_ip_metrics(est_scores, exact_scores, true_nn)
    return mse, ip_bias, ip_var, recalls


# ── Baseline: Per-channel Uniform Quantization ────────────────────────────────


def per_channel_uniform_quantize(vectors: np.ndarray, bit_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize using per-channel (per-dimension) min/max across all vectors.

    Returns
    -------
    indices    : (n, dim) uint8
    chan_mins  : (dim,)
    chan_maxs  : (dim,)
    """
    n_levels = (1 << bit_width) - 1
    chan_mins = vectors.min(axis=0)   # (dim,)
    chan_maxs = vectors.max(axis=0)   # (dim,)
    ranges = chan_maxs - chan_mins
    safe_ranges = np.where(ranges > 0, ranges, 1.0)
    normalised = (vectors - chan_mins[None, :]) / safe_ranges[None, :]
    indices = np.clip(np.round(normalised * n_levels), 0, n_levels).astype(np.uint8)
    return indices, chan_mins, chan_maxs


def per_channel_uniform_dequantize(
    indices: np.ndarray, chan_mins: np.ndarray, chan_maxs: np.ndarray, bit_width: int
) -> np.ndarray:
    n_levels = (1 << bit_width) - 1
    ranges = chan_maxs - chan_mins
    return indices.astype(np.float64) / n_levels * ranges[None, :] + chan_mins[None, :]


def benchmark_per_channel_uniform(
    db: np.ndarray, queries: np.ndarray, bit_width: int
) -> tuple[float, float, float, dict[int, float]]:
    indices, chan_mins, chan_maxs = per_channel_uniform_quantize(db, bit_width)
    reconstructed = per_channel_uniform_dequantize(indices, chan_mins, chan_maxs, bit_width)
    mse = float(np.mean((db - reconstructed) ** 2))

    exact_scores = true_inner_products(db, queries)
    true_nn = true_nearest_neighbors(exact_scores, k=max(RECALL_KS))
    est_scores = queries @ reconstructed.T
    ip_bias, ip_var, recalls = compute_ip_metrics(est_scores, exact_scores, true_nn)
    return mse, ip_bias, ip_var, recalls


# ── Baseline: Random Projection + Uniform Quantization ────────────────────────


def make_random_rotation(dim: int, seed: int) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(A)
    return Q


def benchmark_random_proj_uniform(
    db: np.ndarray, queries: np.ndarray, dim: int, bit_width: int, seed: int
) -> tuple[float, float, float, dict[int, float]]:
    rotation = make_random_rotation(dim, seed=seed)

    # Rotate both db and queries
    rotated_db = db @ rotation.T          # (n, dim)
    rotated_queries = queries @ rotation.T  # (n_q, dim)

    # Uniform quantize the rotated db using per-channel stats
    indices, chan_mins, chan_maxs = per_channel_uniform_quantize(rotated_db, bit_width)
    reconstructed_rotated = per_channel_uniform_dequantize(indices, chan_mins, chan_maxs, bit_width)

    # Rotate back for MSE
    reconstructed = reconstructed_rotated @ rotation  # (n, dim)
    mse = float(np.mean((db - reconstructed) ** 2))

    exact_scores = true_inner_products(db, queries)
    true_nn = true_nearest_neighbors(exact_scores, k=max(RECALL_KS))

    # Inner product estimated in rotated space (rotation is orthogonal, so <Rx, Ry> = <x, y>)
    est_scores = rotated_queries @ reconstructed_rotated.T
    ip_bias, ip_var, recalls = compute_ip_metrics(est_scores, exact_scores, true_nn)
    return mse, ip_bias, ip_var, recalls


# ── TurboQuant benchmarks ─────────────────────────────────────────────────────


def benchmark_turboquant_mse(
    db: np.ndarray, queries: np.ndarray, dim: int, bit_width: int, seed: int
) -> tuple[float, float, float, dict[int, float]]:
    tq = TurboQuant(dim=dim, bit_width=bit_width, mode="mse", seed=seed)
    compressed = tq.quantize(db)
    reconstructed = tq.dequantize(compressed)
    mse = float(np.mean((db - reconstructed) ** 2))

    exact_scores = true_inner_products(db, queries)
    true_nn = true_nearest_neighbors(exact_scores, k=max(RECALL_KS))
    est_scores = np.array([tq.inner_product(q, compressed) for q in queries])
    ip_bias, ip_var, recalls = compute_ip_metrics(est_scores, exact_scores, true_nn)
    return mse, ip_bias, ip_var, recalls


def benchmark_turboquant_ip(
    db: np.ndarray, queries: np.ndarray, dim: int, bit_width: int, seed: int
) -> tuple[float, float, float, dict[int, float]]:
    tq = TurboQuant(dim=dim, bit_width=bit_width, mode="inner_product", seed=seed)
    compressed = tq.quantize(db)
    reconstructed = tq.dequantize(compressed)
    mse = float(np.mean((db - reconstructed) ** 2))

    exact_scores = true_inner_products(db, queries)
    true_nn = true_nearest_neighbors(exact_scores, k=max(RECALL_KS))
    est_scores = np.array([tq.inner_product(q, compressed) for q in queries])
    ip_bias, ip_var, recalls = compute_ip_metrics(est_scores, exact_scores, true_nn)
    return mse, ip_bias, ip_var, recalls


# ── Table printing ─────────────────────────────────────────────────────────────


def print_table(title: str, headers: list[str], rows: list[list]) -> None:
    """Print a fixed-width ASCII table."""
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print(f"\n{title}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)


def result_to_row(res: dict) -> list:
    return [
        res["method"],
        res["dim"],
        f"{res['reconstruction_mse']:.6f}",
        f"{res['ip_bias']:.6f}",
        f"{res['ip_variance']:.6f}",
        f"{res['recall']['@1']:.3f}",
        f"{res['recall']['@5']:.3f}",
        f"{res['recall']['@10']:.3f}",
    ]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Pre-generate all data so methods are compared on identical vectors
    data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for dim in DIMENSIONS:
        db = make_unit_vectors(rng, N_DB, dim)
        queries = make_unit_vectors(rng, N_QUERIES, dim)
        data[dim] = (db, queries)

    all_results: list[dict] = []

    headers = ["method", "dim", "MSE", "IP_bias", "IP_var", "R@1", "R@5", "R@10"]

    for bit_width in BIT_WIDTHS:
        print(f"\n{'='*80}")
        print(f"  Bit-width: {bit_width}")
        print(f"{'='*80}")

        bw_rows: list[list] = []

        for dim in DIMENSIONS:
            db, queries = data[dim]

            # 1. Naive Uniform
            mse, ip_bias, ip_var, recalls = benchmark_naive_uniform(db, queries, bit_width)
            r = build_result("NaiveUniform", dim, bit_width, mse, ip_bias, ip_var, recalls)
            all_results.append(r)
            bw_rows.append(result_to_row(r))

            # 2. Per-channel Uniform
            mse, ip_bias, ip_var, recalls = benchmark_per_channel_uniform(db, queries, bit_width)
            r = build_result("PerChannelUniform", dim, bit_width, mse, ip_bias, ip_var, recalls)
            all_results.append(r)
            bw_rows.append(result_to_row(r))

            # 3. Random Projection + Uniform
            mse, ip_bias, ip_var, recalls = benchmark_random_proj_uniform(
                db, queries, dim, bit_width, seed=SEED
            )
            r = build_result("RandProj+Uniform", dim, bit_width, mse, ip_bias, ip_var, recalls)
            all_results.append(r)
            bw_rows.append(result_to_row(r))

            # 4. TurboQuant MSE
            mse, ip_bias, ip_var, recalls = benchmark_turboquant_mse(
                db, queries, dim, bit_width, seed=SEED
            )
            r = build_result("TurboQuant_MSE", dim, bit_width, mse, ip_bias, ip_var, recalls)
            all_results.append(r)
            bw_rows.append(result_to_row(r))

            # 5. TurboQuant Inner Product
            mse, ip_bias, ip_var, recalls = benchmark_turboquant_ip(
                db, queries, dim, bit_width, seed=SEED
            )
            r = build_result("TurboQuant_IP", dim, bit_width, mse, ip_bias, ip_var, recalls)
            all_results.append(r)
            bw_rows.append(result_to_row(r))

        print_table(f"  bit_width={bit_width}", headers, bw_rows)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {"seed": SEED, "n_db": N_DB, "n_queries": N_QUERIES, "results": all_results},
            f,
            indent=2,
        )
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
