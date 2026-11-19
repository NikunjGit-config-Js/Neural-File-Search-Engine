import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from documents import DocumentError
from search import (SearchEngine, TfidfSearch, chunk_cache_key, embed_texts, embedding_cache_key, index_folder,
                    load_chunks, model_fingerprint, numpy_cosine_search, preprocess_for_tfidf, rank_documents)
from tests.conftest import make_small_corpus


@pytest.fixture(scope="module")
def engine(small_corpus, model, tmp_path_factory):
    return SearchEngine(small_corpus, model=model, cache_dir=tmp_path_factory.mktemp("cache"))


def test_tfidf_preprocessing():
    tokens = preprocess_for_tfidf("The policies are NOT about the vehicles, obviously!")
    assert "the" not in tokens and "!" not in tokens
    assert "policy" in tokens and "vehicle" in tokens  # lemmatised plurals


def test_tfidf_finds_exact_keyword(engine):
    results = engine.search("paternity leave", mode="tfidf")
    assert results[0]["file_name"] == "leave_policy.pdf"
    assert results[0]["location"] == "page 2"


def test_tfidf_returns_nothing_without_shared_words(engine):
    assert engine.search("zebra quantum xylophone", mode="tfidf") == []


def test_minilm_finds_paraphrase(engine):
    # no word in common with "paternity leave" except stopwords
    results = engine.search("how much time off do new dads get", mode="minilm")
    assert results[0]["file_name"] == "leave_policy.pdf"


def test_embedding_shape_norm_and_index(engine):
    vectors = engine.semantic.embeddings
    assert vectors.shape == (len(engine.chunks), 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert engine.semantic.index.d == 384
    assert engine.semantic.index.ntotal == len(engine.chunks)


def test_faiss_matches_numpy_bruteforce(engine, model):
    vectors = engine.semantic.embeddings
    for query in ["electric vans", "rate limit", "solar panels", "annual leave"]:
        q = embed_texts(model, [query])[0]
        scores, ids = engine.semantic.index.search(q.reshape(1, -1), len(vectors))
        brute = numpy_cosine_search(vectors, q, len(vectors))
        assert np.allclose(scores[0], [s for _, s in brute], atol=1e-5)
        assert ids[0][0] == brute[0][0]


def test_results_have_source_metadata_and_one_entry_per_file(engine):
    for mode in ["tfidf", "minilm", "minilm_bruteforce"]:
        results = engine.search("leave", mode=mode, top_k=10)
        names = [r["file_name"] for r in results]
        assert len(names) == len(set(names))
        for r in results:
            assert {"rank", "score", "file_name", "file_type", "location", "text", "chunk_id"} <= set(r)
        assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


def test_top_k_and_empty_query(engine):
    assert len(engine.search("vehicles", "minilm", top_k=2)) == 2
    assert engine.search("   ", "minilm") == []
    with pytest.raises(ValueError):
        engine.search("x", mode="bm25")


def test_query_cache_returns_same_vector(engine):
    engine.semantic.query_cache.clear()
    first = engine.semantic.embed_query("solar panels")
    assert "solar panels" in engine.semantic.query_cache
    second = engine.semantic.embed_query("solar panels")
    assert second is first
    fresh = engine.semantic.embed_query("solar panels", use_cache=False)
    assert np.allclose(fresh, first, atol=1e-6)


def test_rank_documents_keeps_best_chunk_per_file():
    chunks = [{"file_name": "a", "x": 0}, {"file_name": "a", "x": 1}, {"file_name": "b", "x": 2}]
    ranked = rank_documents([(1, 0.9), (0, 0.8), (2, 0.5)], chunks, top_k=5)
    assert [(r["file_name"], r["score"], r["x"]) for r in ranked] == [("a", 0.9, 1), ("b", 0.5, 2)]


def test_cached_and_uncached_results_are_identical(small_corpus, model, tmp_path):
    cached_engine = SearchEngine(small_corpus, model=model, cache_dir=tmp_path / "c")
    cached_again = SearchEngine(small_corpus, model=model, cache_dir=tmp_path / "c")
    uncached = SearchEngine(small_corpus, model=model, use_cache=False)
    assert cached_again.stats["files_from_cache"] == 4 and cached_again.stats["index_from_cache"]
    assert np.array_equal(cached_engine.semantic.embeddings, cached_again.semantic.embeddings)
    assert np.allclose(cached_again.semantic.embeddings, uncached.semantic.embeddings, atol=1e-6)
    for query in ["electric", "429 error", "holiday allowance"]:
        for mode in ["tfidf", "minilm"]:
            a = [(r["file_name"], r["location"]) for r in cached_again.search(query, mode)]
            b = [(r["file_name"], r["location"]) for r in uncached.search(query, mode)]
            assert a == b


def test_cache_invalidates_when_a_file_changes(model, tmp_path):
    docs = make_small_corpus(tmp_path / "docs")
    cache = tmp_path / "cache"
    first = index_folder(docs, model, cache_dir=cache)
    assert first["stats"]["files_processed"] == 4
    with open(docs / "fleet.csv", "a", encoding="utf-8") as f:
        f.write("V-3,Electric,Rotterdam\n")
    second = index_folder(docs, model, cache_dir=cache)
    assert second["stats"]["files_processed"] == 1
    assert second["stats"]["files_from_cache"] == 3
    assert second["stats"]["index_from_cache"] is False
    assert any("Rotterdam" in c["text"] for c in second["chunks"])


def test_cache_keys_depend_on_settings_name_and_model(small_corpus, model, tmp_path):
    # chunk keys never involve the model; embedding keys add the model's fingerprint
    path = small_corpus / "fleet.csv"
    base = chunk_cache_key(path, 120, 30)
    assert base == chunk_cache_key(path, 120, 30)
    assert base != chunk_cache_key(path, 80, 30)
    assert base != chunk_cache_key(path, 120, 10)
    copy = tmp_path / "fleet_copy.csv"
    shutil.copy(path, copy)
    assert base != chunk_cache_key(copy, 120, 30)  # same bytes, different name
    model_id = model_fingerprint(model)
    assert model_id == model_fingerprint(model)
    assert embedding_cache_key(base, model_id) != embedding_cache_key(base, "another-model")


def test_tfidf_is_built_from_chunks_without_the_model(small_corpus, tmp_path):
    loaded = load_chunks(small_corpus, cache_dir=tmp_path)  # step 1 only: MiniLM is not involved
    hits = TfidfSearch(loaded["chunks"]).search_chunks("paternity leave", n=len(loaded["chunks"]))
    assert loaded["chunks"][hits[0][0]]["file_name"] == "leave_policy.pdf"


def test_cli_tfidf_mode_does_not_load_minilm(small_corpus):
    # a fresh process, so modules imported by other tests cannot hide a dependency on PyTorch
    code = ("import runpy, sys\n"
            "sys.argv = ['search.py', 'paternity leave', '--mode', 'tfidf', '--no-cache', '--folder', sys.argv[1]]\n"
            "runpy.run_path('search.py', run_name='__main__')\n"
            "assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules\n")
    out = subprocess.run([sys.executable, "-c", code, str(small_corpus)], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    assert "MiniLM not loaded" in out.stdout and "1. leave_policy.pdf" in out.stdout


def test_bad_files_are_reported_not_fatal(model, tmp_path):
    docs = make_small_corpus(tmp_path / "docs")
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    (docs / "notes.txt").write_text("unsupported")
    engine = SearchEngine(docs, model=model, use_cache=False)
    assert engine.stats["files_failed"] == 1
    assert any("broken.pdf" in e for e in engine.errors)
    assert any("notes.txt" in e for e in engine.errors)
    assert {c["file_name"] for c in engine.chunks} == {"leave_policy.pdf", "api_guide.docx", "roadmap.pptx", "fleet.csv"}


def test_folder_without_text_raises(model, tmp_path):
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(DocumentError, match="No searchable text"):
        SearchEngine(tmp_path, model=model, use_cache=False)
