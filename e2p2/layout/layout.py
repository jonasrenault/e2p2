from abc import ABC, abstractmethod

from PIL import Image

from e2p2.pdf.pdf import LayoutDetection


class LayoutDetectionModel(ABC):
    """
    Abstract base class for Layout Detection models.
    """

    @abstractmethod
    def predict(self, image: Image.Image, *args, **kwargs) -> list[LayoutDetection]:
        pass
