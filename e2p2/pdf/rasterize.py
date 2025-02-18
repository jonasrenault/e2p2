from pathlib import Path
from typing import BinaryIO

import pypdfium2 as pdfium
from PIL.Image import Image


def rasterize_pdf(
    pdf: BinaryIO | str | Path | pdfium.PdfDocument,
    dpi: int = 96,
    pages: list[int] | None = None,
) -> list[Image]:
    """
    Rasterize a PDF file to list of images.

    Args:
        pdf (BinaryIO | str | Path | pdfium.PdfDocument): The pdf file either as bytes,
            a file path, or a pdfium.PdfDocument.
        dpi (int, optional): output DPI. Defaults to 96.
        pages (list[int] | None, optional): list of pages to rasterize.
            If None, rasterize all pages. Defaults to None.

    Raises:
        ValueError: if page number in `pages` is out of bounds.

    Returns:
        list[Image]: list of PIL.Image.Image
    """
    if not isinstance(pdf, pdfium.PdfDocument):
        pdf = pdfium.PdfDocument(pdf)

    if pages is None:
        pages = list(range(len(pdf)))
    else:
        for page in pages:
            if page < 0 or page >= len(pdf):
                raise ValueError(
                    f"Invalid page number {page} for PDF document with {len(pdf)} pages."
                )

    pils = [
        pdf[page_idx]
        .render(
            scale=dpi / 72,  # 72dpi resolution by default
            rotation=0,  # no additional rotation
        )
        .to_pil()
        for page_idx in pages
    ]
    return pils
