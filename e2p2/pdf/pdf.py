from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import numpy.typing as npt
from PIL import Image

from e2p2.pdf.rasterize import rasterize_pdf


class LayoutElement(Enum):
    ABANDONED = -1
    UNKNOWN = 0
    PLAIN_TEXT = 1
    TITLE = 2
    FIGURE = 3
    FIGURE_CAPTION = 4
    TABLE = 5
    TABLE_CAPTION = 6
    TABLE_FOOTNOTE = 7
    FORMULA = 8
    FORMULA_INLINE = 9
    FORMULA_CAPTION = 10
    TEXT = 11

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


@dataclass
class ContentRecognition:
    text: str
    score: float
    cell_bboxes: npt.NDArray | None = None
    logic_points: npt.NDArray | None = None


@dataclass
class LayoutDetection:
    bbox: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    score: float
    category: LayoutElement
    column: int = 0
    content: ContentRecognition | None = None

    @property
    def polygon(
        self,
    ):
        xmin, ymin, xmax, ymax = self.bbox
        return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


@dataclass
class PageInfo:
    page_number: int
    width: int
    height: int
    layout_detections: list[LayoutDetection] = field(default_factory=list)


class PdfDoc(Iterable):
    fpath: Path
    page_ids: list[int]
    images: list[Image.Image]
    page_infos: list[PageInfo]

    def __init__(self, fpath: Path, dpi: int = 96, pages: list[int] | None = None):
        if fpath.suffix not in (".pdf", ".PDF"):
            raise ValueError(f"Invalid file extension {fpath.suffix}. Expected .pdf")
        self.fpath = fpath
        self.images = rasterize_pdf(fpath, dpi=dpi, pages=pages)
        if pages is None:
            self.page_ids = list(range(len(self.images)))
        else:
            self.page_ids = pages

        self.page_infos = [
            PageInfo(image_id, width=image.size[0], height=image.size[1])
            for image_id, image in zip(self.page_ids, self.images)
        ]

    def page_file_stem(self, page_id: int):
        return f"{self.fpath.stem}_{page_id}"

    def _page_index(self, page_id: int) -> int:
        """
        Get page index from page ID (page number).

        Args:
            page_id (int): the page id (number).

        Raises:
            ValueError: if page_id is invalid.

        Returns:
            int: the page index in images and page_infos array.
        """
        try:
            return self.page_ids.index(page_id)
        except ValueError:
            raise ValueError(f"Invalid page id {page_id}.")

    def get_image(self, page_id: int) -> Image.Image:
        """
        Get an image by its ID (page number).

        Args:
            page_id (int): the page id (number).

        Returns:
            Image.Image: the image.
        """
        return self.images[self._page_index(page_id)]

    def get_page_info(self, page_id: int) -> PageInfo:
        """
        Get page info by its ID (page number).

        Args:
            page_id (int): the page id (number).

        Returns:
            PageInfo: the page info.
        """
        return self.page_infos[self._page_index(page_id)]

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


def filter_formulas(layout_detections: list[LayoutDetection]) -> list[LayoutDetection]:
    """
    Filter layout detections returning elements that correspond to formulas.

    Args:
        layout_detections (list[LayoutDetection]): list of layout detections.

    Returns:
        list[LayoutDetection]: formula elements.
    """
    formulas = [
        detection
        for detection in layout_detections
        if detection.category in (LayoutElement.FORMULA, LayoutElement.FORMULA_INLINE)
    ]
    return formulas


def filter_ocr_elements(
    layout_detections: list[LayoutDetection],
) -> list[LayoutDetection]:
    """
    Filter layout detections, keeping only elements where text can be extracted (by OCR
    or pdf parsing).

    Args:
        layout_detections (list[LayoutDetection]): list of layout detections.

    Returns:
        list[LayoutDetection]: elements for OCR.
    """
    ocrs = [
        detection
        for detection in layout_detections
        if detection.category
        in (
            LayoutElement.ABANDONED,
            LayoutElement.PLAIN_TEXT,
            LayoutElement.TITLE,
            LayoutElement.FIGURE_CAPTION,
            LayoutElement.TABLE_CAPTION,
            LayoutElement.TABLE_FOOTNOTE,
        )
    ]
    return ocrs


def filter_tables(layout_detections: list[LayoutDetection]) -> list[LayoutDetection]:
    """
    Filter layout detections, keeping only tables.

    Args:
        layout_detections (list[LayoutDetection]): list of layout detections.

    Returns:
        list[LayoutDetection]: list of tables.
    """
    tables = [
        detection
        for detection in layout_detections
        if detection.category is LayoutElement.TABLE
    ]
    return tables
