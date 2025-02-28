from abc import ABC, abstractmethod
from typing import Literal

from PIL import Image

from e2p2.pdf.pdf import ContentRecognition


class TableModel(ABC):
    """
    Abstract base class for Table extraction models.
    """

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        format: Literal["html", "latex", "markdown"] = "html",
        *args,
        **kwargs,
    ) -> ContentRecognition:
        pass
