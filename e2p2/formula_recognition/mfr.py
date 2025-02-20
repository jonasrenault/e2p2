import re
from abc import ABC, abstractmethod

from PIL import Image


class FormulaRecognitionModel(ABC):
    """
    Abstract base class for Formula Recognition models.
    """

    @abstractmethod
    def predict(self, image: Image.Image, *args, **kwargs) -> str:
        pass


def latex_rm_whitespace(latex: str) -> str:
    """
    Remove unnecessary whitespace from LaTeX code.
    Taken from https://github.com/opendatalab/PDF-Extract-Kit/blob/
    fdb25fd4bd9058ba4e13ac16cb68d4f06b23df56/project/pdf2markdown/
    scripts/pdf2markdown.py#L23

    Args:
        latex (str): input latex string

    Returns:
        str: latex string without unnecessary whitespace.
    """
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = r"[a-zA-Z]"
    noletter = r"[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, latex)]
    latex = re.sub(text_reg, lambda _: str(names.pop(0)), latex)
    news = latex
    while True:
        latex = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", latex)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == latex:
            break
    return latex
