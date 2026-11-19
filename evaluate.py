"""Benchmark TF-IDF vs MiniLM + FAISS, and measure latency and caching.

Evaluation is at the DOCUMENT level: each model's chunk scores are collapsed
to one score per file (its best chunk) before taking the top 5, so a file can
only count once. Both models use exactly the same chunks, queries and labels.

Outputs (in results/):
    per_query_results.csv   every query x model: P@5, R@5, reciprocal rank, top files
    retrieval_metrics.csv   mean metrics per model and query type
    model_comparison.csv    MiniLM vs TF-IDF: differences, relative improvement, bootstrap CI
    latency.csv             timing of every stage, cold and warm, median of repeats
    cache_effectiveness.csv what the disk cache saves, and that it invalidates correctly
    summary.json            environment details and correctness checks

Run:  python evaluate.py            (about 3-4 minutes on a laptop CPU)
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from search import (CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, MODEL_NAME, SearchEngine, TfidfSearch,
                    embed_texts, index_folder, load_model, numpy_cosine_search)

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
TOP_K = 5
MODES = ["tfidf", "minilm", "minilm_bruteforce"]
MODEL_LABELS = {"tfidf": "TF-IDF + cosine", "minilm": "MiniLM + FAISS",
                "minilm_bruteforce": "MiniLM + NumPy cosine (diagnostic)"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(ranked, relevant, k=TOP_K):
    """Fraction of the top-k results that are relevant. Missing slots count as wrong."""
    return sum(f in relevant for f in ranked[:k]) / k


def recall_at_k(ranked, relevant, k=TOP_K):
    """Fraction of all relevant files that appear in the top k."""
    return sum(f in relevant for f in ranked[:k]) / len(relevant) if relevant else float("nan")


def reciprocal_rank(ranked, relevant):
    """1 / rank of the first relevant file (0 if none is retrieved).

    `ranked` holds only the top 5 files, so the average is MRR@5: a first relevant
    file below rank 5 counts as 0. The CSV columns keep the short name "mrr".
    """
    for position, file_name in enumerate(ranked, start=1):
        if file_name in relevant:
            return 1 / position
    return 0.0


def paired_bootstrap_ci(differences, n_resamples=10_000, seed=42):
    """95% confidence interval of the mean per-query difference, by resampling queries."""
    rng = np.random.default_rng(seed)
    differences = np.asarray(differences)
    samples = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    means = differences[samples].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def evaluate_retrieval(engine, queries, labels):
    relevant_by_query = labels.groupby("query_id")["file_name"].apply(set).to_dict()
    rows = []
    for q in queries.itertuples():
        relevant = relevant_by_query.get(q.query_id, set())
        for mode in MODES:
            results = engine.search(q.query, mode, TOP_K)
            ranked = [r["file_name"] for r in results]
            rows.append({
                "query_id": q.query_id, "query_type": q.query_type, "query": q.query, "model": mode,
                "n_relevant": len(relevant),
                "precision_at_5": precision_at_k(ranked, relevant) if relevant else np.nan,
                "recall_at_5": recall_at_k(ranked, relevant),
                "reciprocal_rank": reciprocal_rank(ranked, relevant) if relevant else np.nan,
                "relevant_in_top5": sum(f in relevant for f in ranked),
                "results_returned": len(ranked),
                "top1_file": ranked[0] if ranked else "",
                "top1_score": results[0]["score"] if results else 0.0,
                "top5_files": "; ".join(ranked),
                "relevant_files": "; ".join(sorted(relevant)),
            })
    return pd.DataFrame(rows)


def summarise(per_query):
    rows = []
    answerable = per_query[per_query["n_relevant"] > 0]
    groups = [("all answerable", answerable)] + \
             [(t, per_query[per_query["query_type"] == t]) for t in ["keyword", "paraphrase", "ambiguous", "no_answer"]]
    for group_name, group in groups:
        for mode in MODES:
            g = group[group["model"] == mode]
            rows.append({
                "query_group": group_name, "model": MODEL_LABELS[mode], "n_queries": len(g),
                "precision_at_5": round(g["precision_at_5"].mean(), 4) if group_name != "no_answer" else np.nan,
                "recall_at_5": round(g["recall_at_5"].mean(), 4) if group_name != "no_answer" else np.nan,
                "mrr": round(g["reciprocal_rank"].mean(), 4) if group_name != "no_answer" else np.nan,
                # best P@5 any model could reach: most queries have only 1-2 relevant files
                "max_possible_precision_at_5": round((g["n_relevant"].clip(upper=TOP_K) / TOP_K).mean(), 4)
                if group_name != "no_answer" else np.nan,
                "mean_top1_score": round(g["top1_score"].mean(), 4),
                "mean_results_returned": round(g["results_returned"].mean(), 2),
            })
    return pd.DataFrame(rows)


def compare_models(per_query):
    answerable = per_query[per_query["n_relevant"] > 0]
    tfidf = answerable[answerable["model"] == "tfidf"].set_index("query_id")
    minilm = answerable[answerable["model"] == "minilm"].set_index("query_id")
    rows = []
    for group_name in ["all answerable", "keyword", "paraphrase", "ambiguous"]:
        ids = tfidf.index if group_name == "all answerable" else tfidf.index[tfidf["query_type"] == group_name]
        for metric in ["precision_at_5", "recall_at_5", "reciprocal_rank"]:
            a, b = tfidf.loc[ids, metric], minilm.loc[ids, metric]
            diff = b - a
            low, high = paired_bootstrap_ci(diff.values)
            rows.append({
                "query_group": group_name, "metric": "mrr" if metric == "reciprocal_rank" else metric,
                "n_queries": len(ids), "tfidf": round(a.mean(), 4), "minilm_faiss": round(b.mean(), 4),
                "difference_points": round(diff.mean() * 100, 2),  # percentage points
                "relative_improvement_pct": round((b.mean() - a.mean()) / a.mean() * 100, 1) if a.mean() > 0 else np.nan,
                "diff_95ci_low_points": round(low * 100, 2), "diff_95ci_high_points": round(high * 100, 2),
                "queries_minilm_better": int((diff > 0).sum()), "queries_tfidf_better": int((diff < 0).sum()),
                "queries_tied": int((diff == 0).sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def time_calls(function, repeats):
    """Run `function` several times and return each duration in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return times


def latency_row(scenario, timed, times, note=""):
    return {"scenario": scenario, "what_is_timed": timed, "measurements": len(times),
            "median_ms": round(float(np.median(times)), 3), "p95_ms": round(float(np.percentile(times, 95)), 3),
            "min_ms": round(float(np.min(times)), 3), "max_ms": round(float(np.max(times)), 3), "notes": note}


def measure_cold_start(repeats=3):
    """Import + model load in a brand-new Python process (what a user waits for at start-up)."""
    code = ("import time; t=time.perf_counter(); import search; m=search.load_model(); "
            "t2=time.perf_counter(); m.encode(['warm up']); "
            "print(t2-t, time.perf_counter()-t2)")
    load_ms, first_encode_ms = [], []
    for _ in range(repeats):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent)
        load_s, encode_s = map(float, out.stdout.strip().split()[-2:])
        load_ms.append(load_s * 1000)
        first_encode_ms.append(encode_s * 1000)
    return load_ms, first_encode_ms


def measure_latency(engine, model, queries, repeats=20):
    rows = []
    texts = [q for q in queries["query"]]

    load_ms, first_encode_ms = measure_cold_start()
    rows.append(latency_row("cold_start_model_load", "new Python process: import libraries + load MiniLM from disk",
                            load_ms, "excludes the one-time ~90 MB download; first run after boot is slowest"))
    rows.append(latency_row("cold_start_first_encode", "first MiniLM forward pass in a new process", first_encode_ms,
                            "one-time PyTorch warm-up, not repeated on later queries"))

    with tempfile.TemporaryDirectory() as tmp:
        uncached = time_calls(lambda: index_folder(DOCS_DIR, model, cache_dir=Path(tmp) / "c", use_cache=False), 3)
        rows.append(latency_row("index_build_no_cache", "extract + chunk + embed all files + build FAISS index",
                                uncached, f"{len(engine.chunks)} chunks from 32 files"))
        index_folder(DOCS_DIR, model, cache_dir=Path(tmp) / "warm")  # fill the cache
        cached = time_calls(lambda: index_folder(DOCS_DIR, model, cache_dir=Path(tmp) / "warm"), 10)
        rows.append(latency_row("index_load_from_cache", "hash files + load cached chunks/embeddings + load FAISS index",
                                cached, "files unchanged since the cache was written"))
    rows.append(latency_row("tfidf_fit", "fit TfidfVectorizer (NLTK preprocessing) on all chunks",
                            time_calls(lambda: TfidfSearch(engine.chunks), 5)))

    # warm query latency, model already loaded; repeated and summarised by the median
    for query in texts[:3]:  # warm-up so one-off first-call costs are not counted
        engine.search(query, "minilm")
    rows.append(latency_row(
        "query_tfidf", "end-to-end query: preprocess + sparse cosine over all chunks + rank files",
        [t for q in texts for t in time_calls(lambda: engine.search(q, "tfidf"), repeats)],
        f"{len(texts)} queries x {repeats} repeats"))
    rows.append(latency_row(
        "query_minilm_faiss", "end-to-end query: MiniLM query embedding + FAISS search + rank files",
        [t for q in texts for t in time_calls(lambda: engine.search(q, "minilm", use_query_cache=False), repeats)],
        f"{len(texts)} queries x {repeats} repeats; query cache disabled"))
    for q in texts:
        engine.search(q, "minilm")  # fill the query cache
    rows.append(latency_row(
        "query_minilm_faiss_cached_query", "same, but the query embedding comes from the in-memory query cache",
        [t for q in texts for t in time_calls(lambda: engine.search(q, "minilm"), repeats)],
        "only helps when the exact same query text is repeated"))

    # search step alone: FAISS vs NumPy brute force on the same precomputed vectors
    query_vectors = embed_texts(model, texts)
    embeddings, index = engine.semantic.embeddings, engine.semantic.index
    n = len(embeddings)
    rows.append(latency_row("search_step_faiss_flat", "FAISS IndexFlatIP search only (vector already computed)",
                            [t for v in query_vectors for t in time_calls(lambda: index.search(v.reshape(1, -1), n), 100)],
                            f"{n} vectors; retrieves all chunks, as used for ranking"))
    rows.append(latency_row("search_step_numpy_cosine", "NumPy matrix-vector cosine + argsort only",
                            [t for v in query_vectors for t in time_calls(lambda: numpy_cosine_search(embeddings, v, n), 100)],
                            f"{n} vectors"))

    # Reference point only: a design that stores no document embeddings would
    # re-embed every chunk for every query. Real vector search never does this.
    # What the disk cache itself saves is indexing time (index_build_no_cache vs
    # index_load_from_cache above).
    chunk_texts = [c["text"] for c in engine.chunks]

    def naive_query(q):
        vectors = embed_texts(model, chunk_texts)
        numpy_cosine_search(vectors, embed_texts(model, [q])[0], n)
    rows.append(latency_row("query_without_embedding_cache", "re-embed all chunks + embed query + NumPy cosine",
                            [t for q in texts[:10] for t in time_calls(lambda: naive_query(q), 1)],
                            "naive design with no stored document embeddings; text extraction excluded"))

    # Synthetic scale test: where does FAISS start to matter for the search step?
    rng = np.random.default_rng(0)
    for size in [10_000, 100_000]:
        vectors = rng.standard_normal((size, 384)).astype("float32")
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        probe = vectors[:20] + 0.01
        flat = faiss.IndexFlatIP(384)
        flat.add(vectors)

        def numpy_topk(v):
            scores = vectors @ v
            top = np.argpartition(-scores, TOP_K)[:TOP_K]
            return top[np.argsort(-scores[top])]
        rows.append(latency_row(f"synthetic_{size}_faiss_flat", "FAISS IndexFlatIP top-5 search",
                                [t for v in probe for t in time_calls(lambda: flat.search(v.reshape(1, -1), TOP_K), 10)],
                                "random unit vectors, NOT the document corpus"))
        rows.append(latency_row(f"synthetic_{size}_numpy_cosine", "NumPy matrix-vector + argpartition top-5",
                                [t for v in probe for t in time_calls(lambda: numpy_topk(v), 10)],
                                "random unit vectors, NOT the document corpus"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cache effectiveness and invalidation
# ---------------------------------------------------------------------------

def measure_cache(model):
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        docs, cache = Path(tmp) / "docs", Path(tmp) / "cache"
        shutil.copytree(DOCS_DIR, docs)

        def run(step, **settings):
            result = index_folder(docs, model, cache_dir=cache, **settings)
            s = result["stats"]
            rows.append({"step": step, "files_seen": s["files_seen"], "files_from_cache": s["files_from_cache"],
                         "files_processed": s["files_processed"], "index_from_cache": s["index_from_cache"],
                         "chunks": s["chunks"], "seconds": round(s["seconds"], 3)})
            return result

        first = run("1. empty cache (first indexing)")
        second = run("2. same files again (warm cache)")
        # a cache hit must give exactly the same vectors as recomputing them
        identical = bool(np.array_equal(first["embeddings"], second["embeddings"]))
        with open(docs / "support_tickets.csv", "a", encoding="utf-8") as f:
            f.write("T-1013,Delivery,Parcel left in the rain,Apology and refund of shipping fee\n")
        run("3. one file edited (support_tickets.csv)")
        run("4. chunk size changed 120 -> 80 words", chunk_size=80, overlap=20)
        run("5. back to original settings (old entries reused)")
        (docs / "notes.txt").write_text("not a supported format", encoding="utf-8")
        result = run("6. unsupported .txt file added")
        skipped_reported = any("notes.txt" in e for e in result["errors"])

    fresh = index_folder(DOCS_DIR, model, use_cache=False)["embeddings"]
    return pd.DataFrame(rows), {"cached_equals_first_run": identical,
                                "unsupported_file_reported": skipped_reported,
                                "fresh_embedding_shape": list(fresh.shape)}


# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    queries = pd.read_csv(DATA_DIR / "queries.csv")
    labels = pd.read_csv(DATA_DIR / "relevance_labels.csv")

    print("Loading model and indexing documents...")
    model = load_model()
    engine = SearchEngine(DOCS_DIR, model=model)
    print(f"  {engine.stats['chunks']} chunks, embedding dim {engine.semantic.embeddings.shape[1]}, "
          f"FAISS vectors {engine.semantic.index.ntotal}")

    print("Evaluating retrieval quality...")
    per_query = evaluate_retrieval(engine, queries, labels)
    per_query.to_csv(RESULTS_DIR / "per_query_results.csv", index=False)
    metrics = summarise(per_query)
    metrics.to_csv(RESULTS_DIR / "retrieval_metrics.csv", index=False)
    comparison = compare_models(per_query)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # FAISS (exact) and NumPy brute force must return the same files in the same order
    pivot = per_query.pivot(index="query_id", columns="model", values="top5_files")
    faiss_matches = float((pivot["minilm"] == pivot["minilm_bruteforce"]).mean())

    print("Measuring cache behaviour...")
    cache_table, cache_checks = measure_cache(model)
    cache_table.to_csv(RESULTS_DIR / "cache_effectiveness.csv", index=False)

    print("Measuring latency (includes 3 fresh-process cold starts)...")
    latency = measure_latency(engine, model, queries)
    latency.to_csv(RESULTS_DIR / "latency.csv", index=False)

    import torch
    summary = {
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "cpu_count": os.cpu_count(),
                        "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
                        "faiss": faiss.__version__, "model": MODEL_NAME, "device": "cpu"},
        "corpus": {"files": engine.stats["files_seen"], "chunks": engine.stats["chunks"],
                   "chunk_size_words": CHUNK_SIZE, "chunk_overlap_words": CHUNK_OVERLAP,
                   "queries": len(queries), "answerable_queries": int(labels["query_id"].nunique()),
                   "relevance_labels": len(labels)},
        "checks": {"embedding_dim": int(engine.semantic.embeddings.shape[1]),
                   "faiss_index_dim": int(engine.semantic.index.d),
                   "faiss_index_size_equals_chunks": engine.semantic.index.ntotal == len(engine.chunks),
                   "embeddings_unit_norm": bool(np.allclose(np.linalg.norm(engine.semantic.embeddings, axis=1), 1, atol=1e-5)),
                   "faiss_top5_equals_numpy_bruteforce_fraction": faiss_matches, **cache_checks},
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Retrieval metrics (document level, top 5) ===")
    print(metrics.to_string(index=False))
    print("\n=== MiniLM + FAISS vs TF-IDF ===")
    print(comparison.to_string(index=False))
    print("\n=== Cache ===")
    print(cache_table.to_string(index=False))
    print("\n=== Latency ===")
    print(latency[["scenario", "measurements", "median_ms", "p95_ms"]].to_string(index=False))
    print("\nChecks:", json.dumps(summary["checks"], indent=2))


if __name__ == "__main__":
    main()
