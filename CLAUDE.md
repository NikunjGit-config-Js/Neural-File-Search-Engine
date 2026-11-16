# Neural File Search — notes for Claude Code

Semantic search over PDF/DOCX/PPTX/CSV files. The baseline is TF-IDF (NLTK preprocessing) and the
main model is `sentence-transformers/all-MiniLM-L6-v2` with a FAISS `IndexFlatIP`. Windows,
Python 3.12, virtual environment in `.venv`.

## Commands
- Setup: `python -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt`
- Data: `.venv\Scripts\python prepare_data.py`
- Search CLI: `.venv\Scripts\python search.py "query" --mode minilm|tfidf|minilm_bruteforce`
- Benchmark: `.venv\Scripts\python evaluate.py` (about 3–4 min, writes `results/`)
- Tests: `.venv\Scripts\python -m pytest` (needs the MiniLM weights downloaded once)
- App: `.venv\Scripts\python -m streamlit run app.py`
- Notebook: `.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_search_experiments.ipynb`

## Layout
- `documents.py`: extraction per format and word-window chunking (chunks never cross a page/slide/row)
- `search.py`: TF-IDF, MiniLM + FAISS, disk cache (per-file chunks/embeddings + FAISS index), query LRU cache, CLI
- `evaluate.py`: metrics (document level, top 5), latency, cache checks → `results/*.csv`, `summary.json`
- `app.py`: Streamlit UI · `prepare_data.py`: synthetic corpus, queries, labels · `tests/`: pytest

## Rules for this project
- Never describe TF-IDF as embeddings. MiniLM must be the real sentence-transformers model.
- Never invent or hand-edit metrics. Every number in the README comes from `results/` produced by `evaluate.py`.
- Do not tune models, chunk settings or labels on the evaluation queries. If labels change, say so and explain why.
- Bump `PIPELINE_VERSION` in `search.py` when extraction or chunking changes, so caches are invalidated.
- Keep the code simple: plain functions, a few small classes, no frameworks.
