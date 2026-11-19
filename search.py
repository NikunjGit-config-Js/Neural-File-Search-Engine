"""Search engine: TF-IDF keyword baseline and MiniLM + FAISS semantic search.

Pipeline:
    files -> text extraction -> chunking -> (A) NLTK preprocessing + TF-IDF
                                         -> (B) MiniLM embeddings -> FAISS index
    query -> score every chunk -> keep the best chunk per file -> ranked files

Two different caches are used:
  * Document cache (on disk, in ./cache), in two independent levels:
      - the chunks of each file, keyed by its content, name, the chunk settings
        and PIPELINE_VERSION. No model is involved, so TF-IDF never depends on MiniLM.
      - the embeddings of each file's chunks, keyed by the chunk key and a
        fingerprint of the loaded model's weights, plus the FAISS index.
  * Query cache (in memory): embeddings of recent query strings, so repeating a
    query skips the MiniLM forward pass.

Command line:
    python search.py "how do I claim travel expenses" --mode minilm --top-k 5
"""

import argparse
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from documents import SUPPORTED_EXTENSIONS, DocumentError, list_files, load_document

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 120     # words per chunk
CHUNK_OVERLAP = 30   # words shared by neighbouring chunks
CACHE_DIR = Path("cache")
DOCS_DIR = Path("data/sample_documents")
# Bump this whenever extraction or chunking code changes, so old caches are ignored.
PIPELINE_VERSION = "1"
SEARCH_MODES = {
    "tfidf": "TF-IDF + cosine (keyword baseline)",
    "minilm": "MiniLM + FAISS (semantic)",
    "minilm_bruteforce": "MiniLM + NumPy brute-force cosine (diagnostic)",
}


# ---------------------------------------------------------------------------
# Model A: TF-IDF keyword baseline with NLTK preprocessing
# ---------------------------------------------------------------------------

def ensure_nltk_data():
    for resource, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                           ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_data()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_for_tfidf(text):
    """Tokenise, lowercase, drop stopwords/punctuation and lemmatise.

    Used ONLY for TF-IDF. MiniLM receives the original text because removing
    words like "not" or "before" would change the meaning of a sentence.
    """
    tokens = nltk.word_tokenize(text.lower())
    words = [t for t in tokens if any(ch.isalnum() for ch in t) and t not in STOP_WORDS]
    return [LEMMATIZER.lemmatize(w) for w in words]


class TfidfSearch:
    def __init__(self, chunks):
        self.vectorizer = TfidfVectorizer(tokenizer=preprocess_for_tfidf, lowercase=False,
                                          token_pattern=None, sublinear_tf=True)
        # rows are L2-normalised, so a dot product equals cosine similarity
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search_chunks(self, query, n):
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:n]
        # a chunk that shares no term with the query is not a keyword match
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


# ---------------------------------------------------------------------------
# Model B: MiniLM sentence embeddings + FAISS
# ---------------------------------------------------------------------------

def load_model(model_name=MODEL_NAME):
    from sentence_transformers import SentenceTransformer  # slow import (PyTorch), so done lazily
    try:
        # use the locally cached weights without contacting the Hugging Face Hub
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name, device="cpu")  # first run: download (~90 MB)


def embed_texts(model, texts, batch_size=32):
    """Return L2-normalised float32 embeddings (shape: len(texts) x 384)."""
    vectors = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    return np.ascontiguousarray(vectors, dtype="float32")


def build_faiss_index(embeddings):
    # IndexFlatIP = exact inner-product search over all vectors. Because the
    # vectors are unit length, inner product == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def numpy_cosine_search(embeddings, query_vector, n):
    """Brute-force cosine with NumPy. Used to check FAISS and to time it."""
    scores = embeddings @ query_vector
    order = np.argsort(-scores, kind="stable")[:n]
    return [(int(i), float(scores[i])) for i in order]


class SemanticSearch:
    def __init__(self, model, embeddings, index=None, query_cache_size=1000):
        self.model = model
        self.embeddings = embeddings
        self.index = index if index is not None else build_faiss_index(embeddings)
        if self.index.d != embeddings.shape[1] or self.index.ntotal != len(embeddings):
            raise ValueError("FAISS index does not match the embeddings")
        self.query_cache = OrderedDict()  # small LRU cache: query text -> vector
        self.query_cache_size = query_cache_size

    def embed_query(self, query, use_cache=True):
        if use_cache and query in self.query_cache:
            self.query_cache.move_to_end(query)
            return self.query_cache[query]
        vector = embed_texts(self.model, [query])[0]
        if use_cache:
            self.query_cache[query] = vector
            if len(self.query_cache) > self.query_cache_size:
                self.query_cache.popitem(last=False)
        return vector

    def search_chunks(self, query, n, use_cache=True, use_faiss=True):
        vector = self.embed_query(query, use_cache)
        n = min(n, len(self.embeddings))
        if not use_faiss:
            return numpy_cosine_search(self.embeddings, vector, n)
        scores, ids = self.index.search(vector.reshape(1, -1), n)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


# ---------------------------------------------------------------------------
# Indexing a folder, with an on-disk cache, in two independent steps:
#   1. files -> chunks (no model). TF-IDF is built from these chunks.
#   2. chunks -> MiniLM embeddings + FAISS index, cached per model.
# Both models therefore search exactly the same chunks.
# ---------------------------------------------------------------------------

def chunk_cache_key(path, chunk_size, overlap):
    """Changes whenever the file bytes, file name, chunk settings or PIPELINE_VERSION change."""
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    digest.update(f"|{Path(path).name}|{chunk_size}|{overlap}|v{PIPELINE_VERSION}".encode())
    return digest.hexdigest()[:32]


def model_fingerprint(model):
    """Identify the model that computes the embeddings: a hash of its weights plus its input limit.

    It is computed from the loaded model, so no network access is needed, and embeddings
    cached for one model (or for older weights) are never reused for another. The weight
    hash is stored on the model object, so it is computed once per loaded model.
    """
    weights_hash = getattr(model, "_weights_sha256", None)
    if weights_hash is None:
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()))
        weights_hash = model._weights_sha256 = digest.hexdigest()[:32]
    return f"{weights_hash}|max_seq_length={model.max_seq_length}"


def embedding_cache_key(chunk_key, model_id):
    """Changes whenever a file's chunks or the embedding model change."""
    return hashlib.sha256(f"{chunk_key}|{model_id}".encode()).hexdigest()[:32]


def load_chunks(folder, cache_dir=CACHE_DIR, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, use_cache=True):
    """Step 1, no model needed: extract and chunk every supported file in `folder`.

    Returns a dict with: chunks (all files), files (one (chunk_key, chunks) pair per
    readable file), errors, stats.
    """
    files_dir = Path(cache_dir) / "files"
    if use_cache:
        files_dir.mkdir(parents=True, exist_ok=True)

    files, errors = [], []
    stats = {"files_seen": 0, "files_failed": 0}
    for path in list_files(folder):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(f"{path.name}: skipped, unsupported file type '{path.suffix}'")
            continue
        stats["files_seen"] += 1
        key = chunk_cache_key(path, chunk_size, overlap)
        chunk_file = files_dir / f"{key}.json"

        if use_cache and chunk_file.exists():
            chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        else:
            try:
                chunks = load_document(path, chunk_size, overlap)
            except DocumentError as error:
                errors.append(str(error))
                stats["files_failed"] += 1
                continue
            if use_cache:
                chunk_file.write_text(json.dumps(chunks), encoding="utf-8")
        files.append((key, chunks))

    all_chunks = [chunk for _, file_chunks in files for chunk in file_chunks]
    if not all_chunks:
        raise DocumentError("No searchable text found in the selected files. " + " ".join(errors))
    return {"chunks": all_chunks, "files": files, "errors": errors, "stats": stats}


def embed_chunks(files, model, cache_dir=CACHE_DIR, use_cache=True):
    """Step 2: embed each file's chunks with `model` and build the FAISS index.

    `files` is the list of (chunk_key, chunks) pairs from load_chunks.
    Returns a dict with: embeddings, index, stats.
    """
    files_dir = Path(cache_dir) / "files"
    if use_cache:
        files_dir.mkdir(parents=True, exist_ok=True)

    model_id = model_fingerprint(model)
    all_vectors, keys = [], []
    stats = {"files_from_cache": 0, "files_processed": 0}
    for chunk_key, chunks in files:
        key = embedding_cache_key(chunk_key, model_id)
        vector_file = files_dir / f"{key}.npy"

        if use_cache and vector_file.exists():
            vectors = np.load(vector_file)
            stats["files_from_cache"] += 1
        else:
            vectors = embed_texts(model, [c["text"] for c in chunks])
            stats["files_processed"] += 1
            if use_cache:
                np.save(vector_file, vectors)
        all_vectors.append(vectors)
        keys.append(key)
    embeddings = np.vstack(all_vectors).astype("float32")

    # Cache the FAISS index for this exact set of files. For an exact flat index
    # the saved index is just the stacked vectors; for approximate indexes
    # (IVF, HNSW) the saved file also avoids expensive re-training.
    index = None
    if use_cache:
        corpus_key = hashlib.sha256("".join(keys).encode()).hexdigest()[:32]
        index_file = Path(cache_dir) / f"faiss_{corpus_key}.index"
        if index_file.exists():
            index = faiss.read_index(str(index_file))
            stats["index_from_cache"] = True
        else:
            index = build_faiss_index(embeddings)
            faiss.write_index(index, str(index_file))
            stats["index_from_cache"] = False
    else:
        index = build_faiss_index(embeddings)
        stats["index_from_cache"] = False
    return {"embeddings": embeddings, "index": index, "stats": stats}


def index_folder(folder, model, cache_dir=CACHE_DIR, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                 use_cache=True):
    """Steps 1 and 2 together: extract, chunk and embed every supported file in `folder`.

    Returns a dict with: chunks, embeddings, index, errors, stats.
    """
    start = time.perf_counter()
    loaded = load_chunks(folder, cache_dir, chunk_size, overlap, use_cache)
    embedded = embed_chunks(loaded["files"], model, cache_dir, use_cache)
    stats = {**loaded["stats"], **embedded["stats"], "chunks": len(loaded["chunks"]),
             "seconds": time.perf_counter() - start}
    return {"chunks": loaded["chunks"], "embeddings": embedded["embeddings"], "index": embedded["index"],
            "errors": loaded["errors"], "stats": stats}


# ---------------------------------------------------------------------------
# Search engine combining both models
# ---------------------------------------------------------------------------

def rank_documents(chunk_hits, chunks, top_k):
    """Collapse chunk hits into files: each file is scored by its best chunk.

    `chunk_hits` must be sorted by score (highest first). Ranking files rather
    than chunks stops one file from filling the top 5 with its own chunks.
    """
    results = {}
    for chunk_index, score in chunk_hits:
        chunk = chunks[chunk_index]
        if chunk["file_name"] not in results:
            results[chunk["file_name"]] = {"rank": len(results) + 1, "score": round(score, 4), **chunk}
            if len(results) == top_k:
                break
    return list(results.values())


class SearchEngine:
    def __init__(self, folder=DOCS_DIR, model=None, cache_dir=CACHE_DIR, chunk_size=CHUNK_SIZE,
                 overlap=CHUNK_OVERLAP, use_cache=True):
        self.model = model if model is not None else load_model()
        start = time.perf_counter()
        # Step 1 needs no model: the TF-IDF baseline is built from the chunks alone.
        loaded = load_chunks(folder, cache_dir, chunk_size, overlap, use_cache)
        self.chunks = loaded["chunks"]
        self.errors = loaded["errors"]
        self.tfidf = TfidfSearch(self.chunks)
        # Step 2: MiniLM embeds the same chunks.
        embedded = embed_chunks(loaded["files"], self.model, cache_dir, use_cache)
        self.semantic = SemanticSearch(self.model, embedded["embeddings"], embedded["index"])
        self.stats = {**loaded["stats"], **embedded["stats"], "chunks": len(self.chunks),
                      "seconds": time.perf_counter() - start}

    def search_chunks(self, query, mode="minilm", n=None, use_query_cache=True):
        # Score every chunk (exact search). For much larger corpora you would
        # fetch e.g. top_k * 20 chunks instead.
        n = n or len(self.chunks)
        if mode == "tfidf":
            return self.tfidf.search_chunks(query, n)
        if mode == "minilm":
            return self.semantic.search_chunks(query, n, use_query_cache, use_faiss=True)
        if mode == "minilm_bruteforce":
            return self.semantic.search_chunks(query, n, use_query_cache, use_faiss=False)
        raise ValueError(f"unknown mode '{mode}', choose from {list(SEARCH_MODES)}")

    def search(self, query, mode="minilm", top_k=5, use_query_cache=True):
        query = query.strip()
        if not query:
            return []
        hits = self.search_chunks(query, mode, use_query_cache=use_query_cache)
        return rank_documents(hits, self.chunks, top_k)


def main():
    parser = argparse.ArgumentParser(description="Search PDF, DOCX, PPTX and CSV files.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=list(SEARCH_MODES), default="minilm")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--folder", default=str(DOCS_DIR))
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the disk cache")
    args = parser.parse_args()

    if args.mode == "tfidf":
        # The keyword baseline needs only the chunks, so MiniLM and PyTorch are never loaded.
        start = time.perf_counter()
        loaded = load_chunks(args.folder, use_cache=not args.no_cache)
        chunks, errors, s = loaded["chunks"], loaded["errors"], loaded["stats"]
        tfidf = TfidfSearch(chunks)
        print(f"Indexed {len(chunks)} chunks from {s['files_seen']} files in {time.perf_counter() - start:.2f}s "
              f"({s['files_failed']} failed; MiniLM not loaded)")

        def search(query):
            query = query.strip()
            return rank_documents(tfidf.search_chunks(query, len(chunks)), chunks, args.top_k) if query else []
    else:
        engine = SearchEngine(args.folder, use_cache=not args.no_cache)
        s, errors = engine.stats, engine.errors
        print(f"Indexed {s['chunks']} chunks from {s['files_seen']} files in {s['seconds']:.2f}s "
              f"({s['files_from_cache']} from cache, {s['files_processed']} processed, {s['files_failed']} failed)")

        def search(query):
            return engine.search(query, args.mode, args.top_k)
    for message in errors:
        print("  warning:", message)

    start = time.perf_counter()
    results = search(args.query)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"\n{SEARCH_MODES[args.mode]} | query: {args.query!r} | {elapsed_ms:.1f} ms\n")
    if not results:
        print("No matching documents.")
    for r in results:
        print(f"{r['rank']}. {r['file_name']} ({r['location']})  score={r['score']:.3f}")
        print(f"   {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}\n")


if __name__ == "__main__":
    main()
