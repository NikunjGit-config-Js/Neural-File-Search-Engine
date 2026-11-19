import pytest

from documents import DocumentError, extract_sections, list_files, load_document, split_into_chunks
from prepare_data import write_csv


def test_pdf_keeps_page_numbers(small_corpus):
    sections = extract_sections(small_corpus / "leave_policy.pdf")
    assert [s["location"] for s in sections] == ["page 1", "page 2"]
    assert "paternity leave" in sections[1]["text"]


def test_docx_splits_on_headings_and_merges_title(small_corpus):
    sections = extract_sections(small_corpus / "api_guide.docx")
    assert [s["location"] for s in sections] == ["section 'Authentication'", "section 'Rate Limits'"]
    # the document title has no body, so it is carried into the first section
    assert sections[0]["text"].startswith("API Guide Authentication")
    assert "HTTP 429" in sections[1]["text"]


def test_pptx_keeps_slide_numbers(small_corpus):
    sections = extract_sections(small_corpus / "roadmap.pptx")
    assert [s["location"] for s in sections] == ["slide 1", "slide 2"]
    assert "Solar panels" in sections[1]["text"]


def test_csv_rows_include_column_names(small_corpus):
    sections = extract_sections(small_corpus / "fleet.csv")
    assert [s["location"] for s in sections] == ["row 1", "row 2"]
    assert sections[1]["text"] == "vehicle id: V-2; fuel type: Electric; depot: Utrecht"


def test_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(DocumentError, match="unsupported file type"):
        extract_sections(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(DocumentError, match="empty"):
        extract_sections(path)


@pytest.mark.parametrize("name", ["broken.pdf", "broken.docx", "broken.pptx"])
def test_corrupted_files_raise_document_error(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"this is not a real office or pdf file \x00\x01\x02")
    with pytest.raises(DocumentError, match="could not be read|no extractable text"):
        extract_sections(path)


def test_csv_with_header_only(tmp_path):
    path = tmp_path / "header_only.csv"
    path.write_text("a,b,c\n")
    with pytest.raises(DocumentError, match="no extractable text"):
        extract_sections(path)


def test_csv_latin1_encoding(tmp_path):
    path = tmp_path / "latin.csv"
    path.write_bytes("city,note\nZürich,café\n".encode("latin-1"))
    assert "café" in extract_sections(path)[0]["text"]


def test_chunking_sizes_and_overlap():
    words = [f"w{i}" for i in range(250)]
    chunks = split_into_chunks(" ".join(words), chunk_size=100, overlap=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    # consecutive chunks share exactly `overlap` words
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    # every word is covered
    assert set(" ".join(chunks).split()) == set(words)


def test_chunking_short_text_and_bad_settings():
    assert split_into_chunks("just a few words", 100, 20) == ["just a few words"]
    assert split_into_chunks("   ", 100, 20) == []
    with pytest.raises(ValueError):
        split_into_chunks("text", chunk_size=50, overlap=50)


def test_load_document_metadata(small_corpus):
    chunks = load_document(small_corpus / "fleet.csv")
    assert len(chunks) == 2
    assert chunks[0]["file_name"] == "fleet.csv"
    assert chunks[0]["file_type"] == "csv"
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_long_section_is_chunked_within_its_page(tmp_path):
    path = tmp_path / "long.csv"
    # 299 words + the "text:" column prefix = 300 words -> windows start at 0, 90, 180
    write_csv(path, [{"text": " ".join(f"word{i}" for i in range(299))}])
    chunks = load_document(path, chunk_size=120, overlap=30)
    assert len(chunks) == 3
    assert all(c["location"] == "row 1" for c in chunks)


def test_all_sample_documents_load():
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / "data" / "sample_documents"
    files = list_files(folder)
    assert len(files) == 32
    types = {}
    for path in files:
        chunks = load_document(path)
        assert chunks, path.name
        types[path.suffix] = types.get(path.suffix, 0) + 1
    assert types == {".pdf": 8, ".docx": 8, ".pptx": 8, ".csv": 8}


def test_sample_chunks_fit_the_model_token_limit(model):
    # Chunks are cut by words, but MiniLM silently truncates input longer than
    # max_seq_length word pieces (special tokens included). No sample chunk may be cut off.
    from pathlib import Path
    from search import CHUNK_OVERLAP, CHUNK_SIZE
    folder = Path(__file__).resolve().parents[1] / "data" / "sample_documents"
    texts = [c["text"] for path in list_files(folder) for c in load_document(path, CHUNK_SIZE, CHUNK_OVERLAP)]
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    print(f"{len(texts)} chunks, longest {max(lengths)} of {model.max_seq_length} tokens")
    assert max(lengths) <= model.max_seq_length
