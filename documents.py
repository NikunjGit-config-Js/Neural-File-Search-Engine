"""Text extraction and chunking for PDF, DOCX, PPTX and CSV files.

Every file is turned into a list of *sections* (a PDF page, a DOCX heading
section, a PPTX slide or a CSV row). Sections are then split into overlapping
word-window *chunks*. Each chunk keeps its file name and location so search
results can point back to the original passage.
"""

import re
from pathlib import Path

import pandas as pd
from docx import Document
from pptx import Presentation
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv"}


class DocumentError(Exception):
    """Raised when a file is unsupported, unreadable or has no text."""


def clean_text(text):
    """Light cleanup only: MiniLM works best on natural sentences, so we do not
    remove stopwords or punctuation here (that happens only for TF-IDF)."""
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Extractors: each returns a list of {"location": str, "text": str}
# ---------------------------------------------------------------------------

def extract_pdf(path):
    reader = PdfReader(path)
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            sections.append({"location": f"page {page_number}", "text": text})
    return sections


def extract_docx(path):
    # python-docx has no notion of pages, so we split on headings instead.
    doc = Document(path)
    sections = []
    heading, lines = None, []

    def flush():
        if lines:
            location = f"section '{heading}'" if heading else "start of document"
            sections.append({"location": location, "text": clean_text(" ".join(lines))})

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
        if style.startswith("heading") or style == "title":
            if len(lines) <= 1:
                # previous heading had no body (e.g. document title followed by
                # a section heading): carry it forward instead of making a
                # chunk that contains only a title
                heading, lines = text, lines + [text]
                continue
            flush()
            # keep the heading inside the chunk text: it adds useful context
            heading, lines = text, [text]
        else:
            lines.append(text)
    flush()

    for table_number, table in enumerate(doc.tables, start=1):
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        text = clean_text(" ; ".join(r for r in rows if r.strip(" |")))
        if text:
            sections.append({"location": f"table {table_number}", "text": text})
    return sections


def extract_pptx(path):
    prs = Presentation(path)
    sections = []
    for slide_number, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)
        # bullet points have no final full stop; add one so sentences stay separate
        lines = [line.strip() for part in parts for line in part.splitlines() if line.strip()]
        text = clean_text(". ".join(line.rstrip(".") for line in lines))
        if text:
            sections.append({"location": f"slide {slide_number}", "text": text})
    return sections


def extract_csv(path):
    # Read everything as text so IDs like "007" are not converted to numbers.
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    sections = []
    # Each row becomes "column: value; column: value" so the column names give
    # the values meaning (e.g. "fuel_type: Electric").
    for row_number, row in enumerate(df.itertuples(index=False), start=1):
        pairs = [f"{col.replace('_', ' ')}: {val}" for col, val in zip(df.columns, row) if str(val).strip()]
        text = clean_text("; ".join(pairs))
        if text:
            sections.append({"location": f"row {row_number}", "text": text})
    return sections


EXTRACTORS = {".pdf": extract_pdf, ".docx": extract_docx, ".pptx": extract_pptx, ".csv": extract_csv}


def extract_sections(path):
    """Extract text sections from one file, raising DocumentError on any problem."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in EXTRACTORS:
        raise DocumentError(f"{path.name}: unsupported file type '{extension}' "
                            f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
    if path.stat().st_size == 0:
        raise DocumentError(f"{path.name}: file is empty")
    try:
        sections = EXTRACTORS[extension](path)
    except pd.errors.EmptyDataError:
        raise DocumentError(f"{path.name}: CSV file has no data")
    except Exception as error:  # corrupted files raise many different library errors
        raise DocumentError(f"{path.name}: could not be read ({type(error).__name__}: {error})") from error
    if not sections:
        hint = " (it may be a scanned PDF that needs OCR)" if extension == ".pdf" else ""
        raise DocumentError(f"{path.name}: no extractable text found{hint}")
    return sections


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_into_chunks(text, chunk_size=120, overlap=30):
    """Split text into windows of `chunk_size` words that overlap by `overlap` words.

    120 words is roughly 160 word-piece tokens, below MiniLM's 256-token limit
    (longer input would be silently truncated by the model). A test in
    tests/test_documents.py checks that every sample chunk fits.
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("need chunk_size > 0 and 0 <= overlap < chunk_size")
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)] if words else []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_document(path, chunk_size=120, overlap=30):
    """Extract and chunk one file. Returns a list of chunk dictionaries."""
    path = Path(path)
    chunks = []
    for section in extract_sections(path):
        # Chunks never cross a page/slide/row boundary, so the location stays exact.
        for text in split_into_chunks(section["text"], chunk_size, overlap):
            chunks.append({
                "chunk_id": f"{path.name}#{len(chunks)}",
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "location": section["location"],
                "text": text,
            })
    return chunks


def list_files(folder):
    """All files in a folder (non-recursive), sorted for a stable order."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and not p.name.startswith((".", "~$")))
