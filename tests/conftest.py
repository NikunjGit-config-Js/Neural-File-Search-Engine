import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_data import write_csv, write_docx, write_pdf, write_pptx  # noqa: E402


@pytest.fixture(scope="session")
def model():
    from search import load_model
    return load_model()


def make_small_corpus(folder):
    """Four tiny files, one per format, with clearly different topics."""
    folder.mkdir(parents=True, exist_ok=True)
    write_pdf(folder / "leave_policy.pdf", "Leave Policy", [
        ("Annual Leave", "Employees receive 25 days of paid annual leave each year."),
        ("Parental Leave", "Partners receive four weeks of paternity leave at full pay after the birth."),
    ])
    write_docx(folder / "api_guide.docx", "API Guide", [
        ("Authentication", "Merchants request an OAuth access token with their client ID and secret."),
        ("Rate Limits", "Each account may send 100 requests per minute; extra requests return HTTP 429."),
    ])
    write_pptx(folder / "roadmap.pptx", [
        ("Product Roadmap", ["Route optimisation with live traffic", "Driver app offline mode"]),
        ("Sustainability", ["Solar panels on warehouse roofs", "Recycled paper packaging"]),
    ])
    write_csv(folder / "fleet.csv", [
        {"vehicle_id": "V-1", "fuel_type": "Diesel", "depot": "Amsterdam"},
        {"vehicle_id": "V-2", "fuel_type": "Electric", "depot": "Utrecht"},
    ])
    return folder


@pytest.fixture(scope="session")
def small_corpus(tmp_path_factory):
    return make_small_corpus(tmp_path_factory.mktemp("docs"))
