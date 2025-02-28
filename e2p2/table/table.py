from abc import ABC, abstractmethod

from PIL import Image

from e2p2.pdf.pdf import ContentRecognition


class TableModel(ABC):
    """
    Abstract base class for Table extraction models.
    """

    @abstractmethod
    def predict(self, image: Image.Image, *args, **kwargs) -> ContentRecognition:
        pass
