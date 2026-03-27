"""Neural embeddings test for TurboQuant using real sentence-transformer embeddings.

Uses the 'all-MiniLM-L6-v2' model to generate 384-dim neural embeddings from
the 20 Newsgroups dataset, then benchmarks QJL, TurboQuant (MSE + inner_product),
and naive uniform quantization at bit-widths 2, 3, 4.

Run with:
    uv run python examples/neural_embeddings_test.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import normalize

from turboquant import QJL, TurboQuant

# ── Configuration ────────────────────────────────────────────────────────────
N_DB = 2000
N_QUERIES = 200
EMBED_DIM = 384
BIT_WIDTHS = [2, 3, 4]
TOP_KS = [1, 5, 10]
MODEL_NAME = "all-MiniLM-L6-v2"
RESULTS_PATH = Path(__file__).parent / "results" / "neural_embeddings_results.json"
RANDOM_SEED = 42


# ── Data preparation ─────────────────────────────────────────────────────────

def build_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Return (db_vectors, query_vectors) as float64 unit-length arrays."""
    print("Fetching 20 Newsgroups dataset ...")
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = data.data

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(texts))[: N_DB + N_QUERIES]
    selected = [texts[i] for i in indices]

    print(f"  Loading sentence-transformer model '{MODEL_NAME}' ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=False)

    print(f"  Encoding {len(selected)} documents (this may take a moment) ...")
    embeddings = model.encode(selected, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype(np.float64)

    # Unit-normalise
    embeddings = normalize(embeddings, norm="l2")

    db_vecs = embeddings[:N_DB]
    query_vecs = embeddings[N_DB: N_DB + N_QUERIES]
    print(f"  Database: {db_vecs.shape}  Queries: {query_vecs.shape}")
    return db_vecs, query_vecs


# ── Exact search ─────────────────────────────────────────────────────────────

def exact_topk(db: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Return (n_queries, k) indices of top-k inner products."""
    scores = queries @ db.T          # (n_queries, n_db)
    return np.argsort(-scores, axis=1)[:, :k]


# ── Recall ───────────────────────────────────────────────────────────────────

def recall_at_k(
    approx_topk: np.ndarray,   # (n_queries, k_approx)
    exact_top: np.ndarray,     # (n_queries, k_exact)
    k: int,
) -> float:
    hits = 0
    for approx_row, exact_row in zip(approx_topk[:, :k], exact_top[:, :k]):
        hits += len(set(approx_row) & set(exact_row))
    return hits / (len(exact_top) * k)


# ── MSE distortion ───────────────────────────────────────────────────────────

def mse_distortion(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original - reconstructed
    return float(np.mean(diff ** 2))


# ── Throughput helpers ────────────────────────────────────────────────────────

def _time_quantize(quantizer, vectors: np.ndarray) -> tuple[float, object]:
    t0 = time.perf_counter()
    compressed = quantizer.quantize(vectors)
    return time.perf_counter() - t0, compressed


def _time_search(quantizer, queries: np.ndarray, compressed) -> float:
    t0 = time.perf_counter()
    for q in queries:
        quantizer.inner_product(q, compressed)
    return time.perf_counter() - t0


# ── Naive uniform quantization baseline ──────────────────────────────────────

class NaiveUniformQuantizer:
    """Per-dimension uniform scalar quantizer as a baseline."""

    def __init__(self, bit_width: int) -> None:
        self.bit_width = bit_width
        self.n_levels = 2 ** bit_width
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        self.min_vals = vectors.min(axis=0)
        self.max_vals = vectors.max(axis=0)
        scale = self.max_vals - self.min_vals
        # Avoid division by zero for constant dimensions
        scale = np.where(scale == 0, 1.0, scale)
        normalised = (vectors - self.min_vals) / scale
        codes = np.round(normalised * (self.n_levels - 1)).astype(np.int32)
        return np.clip(codes, 0, self.n_levels - 1)

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        scale = self.max_vals - self.min_vals
        scale = np.where(scale == 0, 1.0, scale)
        return codes.astype(np.float64) / (self.n_levels - 1) * scale + self.min_vals

    def inner_product(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        reconstructed = self.dequantize(codes)
        return reconstructed @ query


# ── Per-method run functions ──────────────────────────────────────────────────

def run_qjl(
    db: np.ndarray,
    queries: np.ndarray,
    exact_results: dict[int, np.ndarray],
) -> list[dict]:
    rows = []
    for bw in BIT_WIDTHS:
        projection_dim = db.shape[1] * bw
        qjl = QJL(dim=db.shape[1], projection_dim=projection_dim, seed=RANDOM_SEED)

        quant_time, compressed = _time_quantize(qjl, db)
        search_time = _time_search(qjl, queries, compressed)

        scores_all = np.stack([qjl.inner_product(q, compressed) for q in queries])
        approx_topk_max = np.argsort(-scores_all, axis=1)[:, : max(TOP_KS)]

        recalls = {k: recall_at_k(approx_topk_max, exact_results[max(TOP_KS)], k) for k in TOP_KS}

        row = {
            "method": "QJL",
            "bit_width": bw,
            "quant_throughput_vecs_per_sec": N_DB / quant_time,
            "search_throughput_queries_per_sec": N_QUERIES / search_time,
            "mse": None,
        }
        for k in TOP_KS:
            row[f"recall@{k}"] = recalls[k]
        rows.append(row)
        print(f"  QJL  b={bw}  recall@1={recalls[1]:.3f}  recall@10={recalls[10]:.3f}")
    return rows


def run_turboquant(
    db: np.ndarray,
    queries: np.ndarray,
    exact_results: dict[int, np.ndarray],
    mode: str,
) -> list[dict]:
    rows = []
    for bw in BIT_WIDTHS:
        tq = TurboQuant(dim=db.shape[1], bit_width=bw, mode=mode, seed=RANDOM_SEED)

        quant_time, compressed = _time_quantize(tq, db)
        search_time = _time_search(tq, queries, compressed)

        scores_all = np.stack([tq.inner_product(q, compressed) for q in queries])
        approx_topk_max = np.argsort(-scores_all, axis=1)[:, : max(TOP_KS)]

        recalls = {k: recall_at_k(approx_topk_max, exact_results[max(TOP_KS)], k) for k in TOP_KS}

        reconstructed = tq.dequantize(compressed)
        mse = mse_distortion(db, reconstructed)

        label = f"TurboQuant-{mode}"
        row = {
            "method": label,
            "bit_width": bw,
            "quant_throughput_vecs_per_sec": N_DB / quant_time,
            "search_throughput_queries_per_sec": N_QUERIES / search_time,
            "mse": mse,
        }
        for k in TOP_KS:
            row[f"recall@{k}"] = recalls[k]
        rows.append(row)
        print(f"  {label}  b={bw}  recall@1={recalls[1]:.3f}  recall@10={recalls[10]:.3f}  mse={mse:.6f}")
    return rows


def run_naive_uniform(
    db: np.ndarray,
    queries: np.ndarray,
    exact_results: dict[int, np.ndarray],
) -> list[dict]:
    rows = []
    for bw in BIT_WIDTHS:
        quant = NaiveUniformQuantizer(bit_width=bw)

        quant_time, codes = _time_quantize(quant, db)
        search_time = _time_search(quant, queries, codes)

        scores_all = np.stack([quant.inner_product(q, codes) for q in queries])
        approx_topk_max = np.argsort(-scores_all, axis=1)[:, : max(TOP_KS)]

        recalls = {k: recall_at_k(approx_topk_max, exact_results[max(TOP_KS)], k) for k in TOP_KS}

        reconstructed = quant.dequantize(codes)
        mse = mse_distortion(db, reconstructed)

        row = {
            "method": "NaiveUniform",
            "bit_width": bw,
            "quant_throughput_vecs_per_sec": N_DB / quant_time,
            "search_throughput_queries_per_sec": N_QUERIES / search_time,
            "mse": mse,
        }
        for k in TOP_KS:
            row[f"recall@{k}"] = recalls[k]
        rows.append(row)
        print(f"  NaiveUniform  b={bw}  recall@1={recalls[1]:.3f}  recall@10={recalls[10]:.3f}  mse={mse:.6f}")
    return rows


# ── Table printing ────────────────────────────────────────────────────────────

def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'─' * 88}")
    print(f"  {title}")
    print(f"{'─' * 88}")
    header = (
        f"{'Method':<26} {'Bits':>4}  {'R@1':>6}  {'R@5':>6}  {'R@10':>7}  "
        f"{'MSE':>10}  {'Q-thpt':>10}  {'S-thpt':>10}"
    )
    print(header)
    print("─" * 88)
    for r in rows:
        mse_str = f"{r['mse']:.6f}" if r["mse"] is not None else "     n/a"
        print(
            f"{r['method']:<26} {r['bit_width']:>4}  "
            f"{r['recall@1']:>6.3f}  {r['recall@5']:>6.3f}  {r['recall@10']:>7.3f}  "
            f"{mse_str:>10}  "
            f"{r['quant_throughput_vecs_per_sec']:>10.0f}  "
            f"{r['search_throughput_queries_per_sec']:>10.0f}"
        )
    print(f"{'─' * 88}")
    print("  Q-thpt = quantization vecs/sec  |  S-thpt = search queries/sec")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    db, queries = build_vectors()

    print("\nComputing exact inner-product results ...")
    exact_topk_results: dict[int, np.ndarray] = {}
    for k in TOP_KS:
        exact_topk_results[k] = exact_topk(db, queries, k)
    exact_topk_results[max(TOP_KS)] = exact_topk(db, queries, max(TOP_KS))

    all_rows: list[dict] = []

    print("\nRunning QJL ...")
    qjl_rows = run_qjl(db, queries, exact_topk_results)
    all_rows.extend(qjl_rows)

    print("\nRunning TurboQuant (mse mode) ...")
    tq_mse_rows = run_turboquant(db, queries, exact_topk_results, mode="mse")
    all_rows.extend(tq_mse_rows)

    print("\nRunning TurboQuant (inner_product mode) ...")
    tq_ip_rows = run_turboquant(db, queries, exact_topk_results, mode="inner_product")
    all_rows.extend(tq_ip_rows)

    print("\nRunning Naive Uniform quantization (baseline) ...")
    naive_rows = run_naive_uniform(db, queries, exact_topk_results)
    all_rows.extend(naive_rows)

    print_table(
        f"Recall & Distortion  (20 Newsgroups, {MODEL_NAME}, unit-norm)",
        all_rows,
    )

    # ── Save results ─────────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "model": MODEL_NAME,
            "n_db": N_DB,
            "n_queries": N_QUERIES,
            "embed_dim": EMBED_DIM,
            "bit_widths": BIT_WIDTHS,
            "top_ks": TOP_KS,
            "random_seed": RANDOM_SEED,
        },
        "rows": all_rows,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
