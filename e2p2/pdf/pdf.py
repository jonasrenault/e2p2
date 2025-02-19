from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image

from e2p2.layout.layout import LayoutDetection
from e2p2.pdf.rasterize import rasterize_pdf


@dataclass
class PageInfo:
    page_number: int
    height: int
    width: int
    layout_detections: list[LayoutDetection] | None = None


class PdfDoc(Iterable):
    fpath: Path
    page_ids: list[int]
    images: list[Image.Image]
    page_infos: list[PageInfo] | None = None

    def __init__(self, fpath: Path, dpi: int = 96, pages: list[int] | None = None):
        if fpath.suffix not in (".pdf", ".PDF"):
            raise ValueError(f"Invalid file extension {fpath.suffix}. Expected .pdf")
        self.fpath = fpath
        self.images = rasterize_pdf(fpath, dpi=dpi, pages=pages)
        if pages is None:
            self.page_ids = list(range(len(self.images)))
        else:
            self.page_ids = pages

    def page_file_stem(self, page_id: int):
        return f"{self.fpath.stem}_{page_id}"

    def get_image(self, page_id: int) -> Image.Image:
        """
        Get an image by its ID (page number).

        Args:
            page_id (int): the page id (number).

        Raises:
            ValueError: if page_id is invalid.

        Returns:
            Image.Image: the image.
        """
        try:
            idx = self.page_ids.index(page_id)
        except ValueError:
            raise ValueError(f"Invalid page id {page_id}.")
        return self.images[idx]

    def __len__(self) -> int:
        """
        Return the number of loaded pages in the document.

        Returns:
            int: number of pages in the document.
        """
        return len(self.images) if self.images is not None else 0

    def __getitem__(self, idx: int) -> tuple[int, Image.Image]:
        """
        Get an image and its corresponding ID (page number) by index.

        Args:
            idx (int): Index of the image to retrive.

        Raises:
            IndexError: if index is out of bounds

        Returns:
            tuple[int, Image.Image]: Image id and RGB image.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds ({len(self)}).")

        image = self.images[idx]
        image_id = self.page_ids[idx]
        return image_id, image

    def __iter__(self) -> Iterator[tuple[int, Image.Image]]:
        """
        Iterate through all the loaded images in the document.

        Yields:
            Iterator[tuple[int, Image.Image]]: image id and image.
        """
        for image_id, image in zip(self.page_ids, self.images):
            yield image_id, image
