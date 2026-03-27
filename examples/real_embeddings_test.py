"""Real embeddings test for TurboQuant using sklearn 20 Newsgroups + TF-IDF.

Uses scikit-learn's 20 Newsgroups dataset to produce realistic high-dimensional
text vectors, reduces to 384-d with TruncatedSVD (mimicking sentence embedding
dimensions), then benchmarks QJL and TurboQuant at bit-widths 2, 3, 4.

Run with:
    uv run python examples/real_embeddings_test.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from turboquant import QJL, TurboQuant

# ── Configuration ────────────────────────────────────────────────────────────
N_DB = 2000
N_QUERIES = 200
EMBED_DIM = 384
BIT_WIDTHS = [2, 3, 4]
TOP_KS = [1, 5, 10]
RESULTS_PATH = Path(__file__).parent / "results" / "real_embeddings_results.json"
RANDOM_SEED = 42


# ── Data preparation ─────────────────────────────────────────────────────────

def build_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Return (db_vectors, query_vectors) as float64 unit-length arrays."""
    print("Fetching 20 Newsgroups dataset …")
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = data.data

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(texts))[: N_DB + N_QUERIES]
    selected = [texts[i] for i in indices]

    print(f"  Vectorising {len(selected)} documents with TF-IDF …")
    vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(selected)  # sparse

    print(f"  TF-IDF shape: {tfidf_matrix.shape}  →  reducing to {EMBED_DIM}-d with TruncatedSVD …")
    svd = TruncatedSVD(n_components=EMBED_DIM, random_state=RANDOM_SEED)
    dense = svd.fit_transform(tfidf_matrix)          # (N, EMBED_DIM) float64

    # Unit-normalise
    dense = normalize(dense, norm="l2").astype(np.float64)

    db_vecs = dense[:N_DB]
    query_vecs = dense[N_DB: N_DB + N_QUERIES]
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
    exact_topk: np.ndarray,    # (n_queries, k_exact)
    k: int,
) -> float:
    hits = 0
    for approx_row, exact_row in zip(approx_topk[:, :k], exact_topk[:, :k]):
        hits += len(set(approx_row) & set(exact_row))
    return hits / (len(exact_topk) * k)


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


# ── QJL wrapper (inner_product method lives on QJL directly) ─────────────────

def run_qjl(
    db: np.ndarray,
    queries: np.ndarray,
    exact_results: dict[int, np.ndarray],
) -> list[dict]:
    rows = []
    for bw in BIT_WIDTHS:
        projection_dim = db.shape[1] * bw   # bw bits per coord → bw×dim projection rows
        qjl = QJL(dim=db.shape[1], projection_dim=projection_dim, seed=RANDOM_SEED)

        quant_time, compressed = _time_quantize(qjl, db)
        search_time = _time_search(qjl, queries, compressed)

        # Approximate top-k via QJL scores
        scores_all = np.stack([qjl.inner_product(q, compressed) for q in queries])  # (nq, ndb)
        approx_topk_max = np.argsort(-scores_all, axis=1)[:, : max(TOP_KS)]

        recalls = {}
        for k in TOP_KS:
            recalls[k] = recall_at_k(approx_topk_max, exact_results[max(TOP_KS)], k)

        # QJL has no explicit dequantize; skip MSE
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

        recalls = {}
        for k in TOP_KS:
            recalls[k] = recall_at_k(approx_topk_max, exact_results[max(TOP_KS)], k)

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


# ── Table printing ────────────────────────────────────────────────────────────

def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")
    header = f"{'Method':<22} {'Bits':>4}  {'R@1':>6}  {'R@5':>6}  {'R@10':>7}  {'MSE':>10}  {'Q-thpt':>10}  {'S-thpt':>10}"
    print(header)
    print("─" * 80)
    for r in rows:
        mse_str = f"{r['mse']:.6f}" if r["mse"] is not None else "     n/a"
        print(
            f"{r['method']:<22} {r['bit_width']:>4}  "
            f"{r['recall@1']:>6.3f}  {r['recall@5']:>6.3f}  {r['recall@10']:>7.3f}  "
            f"{mse_str:>10}  "
            f"{r['quant_throughput_vecs_per_sec']:>10.0f}  "
            f"{r['search_throughput_queries_per_sec']:>10.0f}"
        )
    print(f"{'─' * 80}")
    print("  Q-thpt = quantization vecs/sec  |  S-thpt = search queries/sec")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    db, queries = build_vectors()

    print("\nComputing exact inner-product results …")
    exact_topk_results: dict[int, np.ndarray] = {}
    for k in TOP_KS:
        exact_topk_results[k] = exact_topk(db, queries, k)
    # Store the largest k so recall helpers can slice
    exact_topk_results[max(TOP_KS)] = exact_topk(db, queries, max(TOP_KS))

    all_rows: list[dict] = []

    print("\nRunning QJL …")
    qjl_rows = run_qjl(db, queries, exact_topk_results)
    all_rows.extend(qjl_rows)

    print("\nRunning TurboQuant (mse mode) …")
    tq_mse_rows = run_turboquant(db, queries, exact_topk_results, mode="mse")
    all_rows.extend(tq_mse_rows)

    print("\nRunning TurboQuant (inner_product mode) …")
    tq_ip_rows = run_turboquant(db, queries, exact_topk_results, mode="inner_product")
    all_rows.extend(tq_ip_rows)

    print_table("Recall & Distortion  (20 Newsgroups TF-IDF → SVD-384, unit-norm)", all_rows)

    # ── Save results ─────────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
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
