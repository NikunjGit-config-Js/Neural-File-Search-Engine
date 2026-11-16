"""Streamlit interface for the search engine.

Run:  streamlit run app.py
"""

import tempfile
import time
from pathlib import Path

import streamlit as st

from documents import SUPPORTED_EXTENSIONS, DocumentError
from search import DOCS_DIR, SEARCH_MODES, SearchEngine, load_model

st.set_page_config(page_title="Neural File Search", layout="wide")


@st.cache_resource(show_spinner="Loading the MiniLM model...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Indexing documents...", max_entries=5)
def get_engine(folder, fingerprint):
    # `fingerprint` is only part of the cache key: when files are added, removed
    # or changed, Streamlit builds a new engine. Unchanged files are still read
    # from the on-disk embedding cache, so this is fast.
    return SearchEngine(folder, model=get_model())


def folder_fingerprint(folder):
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(Path(folder).iterdir()) if p.is_file())


def sync_uploads(uploaded_files):
    """Mirror the uploaded files into a temporary folder for this browser session."""
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = tempfile.mkdtemp(prefix="neural_file_search_")
    folder = Path(st.session_state.upload_dir)
    wanted = {}
    for f in uploaded_files:
        wanted[Path(f.name).name] = f.getvalue()  # Path(...).name drops any directory part
    for existing in folder.iterdir():
        if existing.name not in wanted:
            existing.unlink()
    for name, content in wanted.items():
        target = folder / name
        if not target.exists() or target.read_bytes() != content:  # keep mtime stable if unchanged
            target.write_bytes(content)
    return folder


st.title("Neural File Search")
st.caption("Search inside PDF, DOCX, PPTX and CSV files using TF-IDF keyword matching "
           "or MiniLM sentence embeddings with a FAISS index.")

with st.sidebar:
    st.header("Settings")
    source = st.radio("Documents", ["Sample documents", "Upload my own files"])
    mode_options = {"MiniLM + FAISS (semantic)": ["minilm"],
                    "TF-IDF (keyword baseline)": ["tfidf"],
                    "Compare both side by side": ["tfidf", "minilm"]}
    mode_choice = st.radio("Search mode", list(mode_options))
    top_k = st.slider("Number of results (top-k)", min_value=1, max_value=10, value=5)

if source == "Upload my own files":
    uploaded = st.sidebar.file_uploader("Upload files", type=sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS),
                                        accept_multiple_files=True)
    if not uploaded:
        st.info("Upload one or more .pdf, .docx, .pptx or .csv files in the sidebar to start searching.")
        st.stop()
    folder = sync_uploads(uploaded)
else:
    folder = DOCS_DIR
    if not folder.exists() or not any(folder.iterdir()):
        st.error(f"No sample documents found in {folder}. Run `python prepare_data.py` first.")
        st.stop()

try:
    engine = get_engine(str(folder), folder_fingerprint(folder))
except DocumentError as error:
    st.error(f"Could not index the documents: {error}")
    st.stop()

files = sorted({c["file_name"] for c in engine.chunks})
st.sidebar.markdown(f"**Indexed:** {len(files)} files, {len(engine.chunks)} chunks")
with st.sidebar.expander("Indexed files"):
    st.write("\n".join(f"- {name}" for name in files))
for message in engine.errors:
    st.warning(message)

query = st.text_input("Search query", placeholder="e.g. how much time off do new parents get?")

if query.strip():
    modes = mode_options[mode_choice]
    columns = st.columns(len(modes))
    for column, mode in zip(columns, modes):
        with column:
            start = time.perf_counter()
            results = engine.search(query, mode, top_k)
            elapsed_ms = (time.perf_counter() - start) * 1000
            st.subheader(SEARCH_MODES[mode])
            st.caption(f"{len(results)} result(s) in {elapsed_ms:.1f} ms")
            if not results:
                st.info("No matching documents. TF-IDF only finds files that share at least one word "
                        "with the query - try MiniLM for questions phrased differently.")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r['rank']}. {r['file_name']}**  \n"
                                f"{r['file_type'].upper()} - {r['location']} - similarity `{r['score']:.3f}`")
                    snippet = r["text"] if len(r["text"]) <= 500 else r["text"][:500] + " ..."
                    st.write(snippet)
    st.caption("Scores are cosine similarities. TF-IDF and MiniLM scores are on different scales, so compare "
               "rankings rather than raw numbers between the two modes. Each file appears once, represented "
               "by its best-matching passage.")
