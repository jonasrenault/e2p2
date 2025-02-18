from pathlib import Path

import pytest
from PIL import ImageChops

from e2p2.pdf.rasterize import rasterize_pdf

ROOT_DIR = Path(__file__).parent.parent.parent


@pytest.fixture
def pdf_path() -> Path:
    return ROOT_DIR / "resources/data/test-doc.pdf"


def test_rasterize_returns_pils_array(pdf_path: Path):
    pils = rasterize_pdf(pdf_path)
    assert len(pils) == 2

    assert pils[0].size == (816, 1056)
    assert pils[1].size == (816, 1056)

    page_1 = rasterize_pdf(pdf_path, pages=[1])
    assert len(page_1) == 1
    assert page_1[0].size == (816, 1056)

    diff = ImageChops.difference(pils[1], page_1[0])
    assert diff.getbbox() is None

    with pytest.raises(
        ValueError, match="Invalid page number 3 for PDF document with 2 pages."
    ):
        rasterize_pdf(pdf_path, pages=[0, 3])
