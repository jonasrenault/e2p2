from abc import ABC, abstractmethod

from PIL import Image

from e2p2.pdf.pdf import LayoutDetection


class OCRModel(ABC):
    """
    Abstract base class for Layout Detection models.
    """

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        formula_bboxes: list[tuple[int, int, int, int]] | None = None,
        *args,
        **kwargs,
    ) -> list[LayoutDetection]:
        pass
