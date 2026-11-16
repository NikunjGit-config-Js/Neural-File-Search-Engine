"""Runs the real Streamlit script headlessly with Streamlit's AppTest."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app():
    os.chdir(ROOT)  # the app uses paths relative to the project folder
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def result_text(at):
    return " ".join(m.value for m in at.markdown)


def test_app_starts_and_indexes_sample_documents(app):
    sidebar = " ".join(m.value for m in app.sidebar.markdown)
    assert "32 files" in sidebar


def test_app_minilm_search_shows_ranked_results(app):
    app.text_input[0].input("how much time off do new dads get").run()
    assert not app.exception
    text = result_text(app)
    assert "1. employee_leave_policy.pdf" in text
    assert "similarity" in text


def test_app_tfidf_mode_and_no_results_message(app):
    app.sidebar.radio[1].set_value("TF-IDF (keyword baseline)").run()
    app.text_input[0].input("zebra xylophone").run()
    assert not app.exception
    assert any("No matching documents" in i.value for i in app.info)


def test_app_compare_mode_and_top_k(app):
    app.sidebar.radio[1].set_value("Compare both side by side").run()
    app.sidebar.slider[0].set_value(3).run()
    app.text_input[0].input("forklift certification").run()
    assert not app.exception
    headers = [h.value for h in app.subheader]
    assert any("TF-IDF" in h for h in headers) and any("FAISS" in h for h in headers)
    captions = [c.value for c in app.caption]
    # MiniLM always fills top-k; TF-IDF only returns files sharing a query word (2 here)
    assert "3 result(s)" in " ".join(captions) and "2 result(s)" in " ".join(captions)
    assert result_text(app).count("similarity") == 5


def test_app_upload_mode_asks_for_files(app):
    app.sidebar.radio[0].set_value("Upload my own files").run()
    assert not app.exception
    assert any("Upload one or more" in i.value for i in app.info)
