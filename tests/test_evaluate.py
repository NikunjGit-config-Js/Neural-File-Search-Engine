import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate import paired_bootstrap_ci, precision_at_k, recall_at_k, reciprocal_rank

RESULTS = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "data"


def test_precision_recall_mrr_by_hand():
    ranked = ["a", "x", "b", "y", "z"]
    relevant = {"a", "b", "c"}
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, relevant) == 1.0
    assert reciprocal_rank(["x", "y", "b"], relevant) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], relevant) == 0.0


def test_precision_counts_missing_results_as_misses():
    # TF-IDF can return fewer than 5 files; empty slots are not free hits
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)


def test_bootstrap_ci_contains_mean():
    diffs = np.array([0.0, 0.2, 0.2, 0.0, 0.4, 0.0, 0.2])
    low, high = paired_bootstrap_ci(diffs)
    assert low <= diffs.mean() <= high


def test_labels_reference_real_files_and_queries():
    queries = pd.read_csv(DATA / "queries.csv")
    labels = pd.read_csv(DATA / "relevance_labels.csv")
    files = {p.name for p in (DATA / "sample_documents").iterdir()}
    assert set(labels["file_name"]) <= files
    assert set(labels["query_id"]) <= set(queries["query_id"])
    no_answer = set(queries.loc[queries["query_type"] == "no_answer", "query_id"])
    assert not no_answer & set(labels["query_id"])


@pytest.mark.skipif(not (RESULTS / "retrieval_metrics.csv").exists(), reason="run evaluate.py first")
def test_result_files_contain_computed_values():
    metrics = pd.read_csv(RESULTS / "retrieval_metrics.csv")
    answerable = metrics[metrics["query_group"] != "no_answer"]
    for column in ["precision_at_5", "recall_at_5", "mrr"]:
        assert answerable[column].notna().all()
        assert answerable[column].between(0, 1).all()

    # recompute the headline numbers from the per-query file
    per_query = pd.read_csv(RESULTS / "per_query_results.csv")
    for model, label in [("tfidf", "TF-IDF + cosine"), ("minilm", "MiniLM + FAISS")]:
        rows = per_query[(per_query["model"] == model) & (per_query["n_relevant"] > 0)]
        row = metrics[(metrics["query_group"] == "all answerable") & (metrics["model"] == label)].iloc[0]
        assert rows["precision_at_5"].mean() == pytest.approx(row["precision_at_5"], abs=1e-4)
        assert rows["reciprocal_rank"].mean() == pytest.approx(row["mrr"], abs=1e-4)

    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    p5 = comparison[(comparison["query_group"] == "all answerable") & (comparison["metric"] == "precision_at_5")].iloc[0]
    expected = (p5["minilm_faiss"] - p5["tfidf"]) / p5["tfidf"] * 100
    assert p5["relative_improvement_pct"] == pytest.approx(expected, abs=0.1)

    latency = pd.read_csv(RESULTS / "latency.csv")
    assert (latency["median_ms"] > 0).all()
    assert len(latency) >= 10

    checks = json.loads((RESULTS / "summary.json").read_text())["checks"]
    assert checks["embedding_dim"] == checks["faiss_index_dim"] == 384
    assert checks["faiss_top5_equals_numpy_bruteforce_fraction"] == 1.0
    assert checks["cached_equals_first_run"]
