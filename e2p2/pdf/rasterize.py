from pathlib import Path
from typing import BinaryIO

import numpy as np
import pypdfium2 as pdfium
from PIL import Image


def get_average_color(image: Image.Image) -> tuple[int, ...]:
    # Convert image to numpy array
    img_array = np.array(image)

    # Get average color, ignoring fully transparent pixels
    if img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:, :, 3]
        rgb = img_array[:, :, :3]
        mask = alpha > 0
        if mask.any():
            avg_color = rgb[mask].mean(axis=0)
        else:
            avg_color = rgb.mean(axis=(0, 1))
    else:  # RGB
        avg_color = img_array.mean(axis=(0, 1))
    return tuple(map(int, avg_color))


def get_contrasting_color(color: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(255 - c for c in color)


def convert_transparent_to_contrasting(image: Image.Image) -> Image.Image:
    """
    Convert transparent pixels to a contrasting color.
    Taken from https://github.com/breezedeus/Pix2Text/
    blob/c87047b1732d55d1e2cce123b768cea52303db77/pix2text/utils.py#L176

    Args:
        image (Image.Image): input image with alpha channel.

    Returns:
        Image.Image: RGB image without alpha channel.
    """
    # Check if the image has an alpha channel
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        # Get average color of non-transparent pixels
        avg_color = get_average_color(image)

        # Get contrasting color for background
        bg_color = get_contrasting_color(avg_color)

        # Create a new background image with the contrasting color
        background = Image.new("RGBA", image.size, bg_color)

        # Paste the image on the background.
        # If the image has an alpha channel, it will be used as a mask
        background.paste(image, (0, 0), image)

        # Convert to RGB (removes alpha channel)
        return background.convert("RGB")

    return image.convert("RGB")


def rasterize_pdf(
    pdf: BinaryIO | str | Path | pdfium.PdfDocument,
    dpi: int = 96,
    pages: list[int] | None = None,
) -> list[Image.Image]:
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

    pils = [convert_transparent_to_contrasting(image) for image in pils]
    return pils
