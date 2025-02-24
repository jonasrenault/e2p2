from abc import ABC, abstractmethod

from PIL import Image

from e2p2.layout.layout import LayoutDetection


class OCRModel(ABC):
    """
    Abstract base class for Layout Detection models.
    """

    @abstractmethod
    def predict(self, image: Image.Image, *args, **kwargs) -> list[LayoutDetection]:
        pass
