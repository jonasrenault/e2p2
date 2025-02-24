import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from e2p2.pdf.pdf import LayoutDetection, LayoutElement

LAYOUT_ELEMENT_TEXT_COLOR = {
    LayoutElement.ABANDONED: ("abandon", (141, 211, 199)),
    LayoutElement.UNKNOWN: ("unknown", (190, 186, 218)),
    LayoutElement.TEXT: ("text", (251, 128, 114)),
    LayoutElement.TITLE: ("title", (128, 177, 211)),
    LayoutElement.FIGURE: ("figure", (253, 180, 98)),
    LayoutElement.FIGURE_CAPTION: ("figure caption", (179, 222, 105)),
    LayoutElement.TABLE: ("table", (252, 205, 229)),
    LayoutElement.TABLE_CAPTION: ("table caption", (217, 217, 217)),
    LayoutElement.TABLE_FOOTNOTE: ("table footnote", (188, 128, 189)),
    LayoutElement.FORMULA: ("formula", (204, 235, 197)),
    LayoutElement.FORMULA_INLINE: ("formula inline", (255, 255, 179)),
    LayoutElement.FORMULA_CAPTION: ("formula caption", (255, 237, 111)),
}


def visualize_bbox(
    image: Image.Image,
    detections: list[LayoutDetection],
    alpha: float = 0.3,
) -> npt.NDArray:
    """
    Visualize layout detection results on an image.

    Args:
        image (Image.Image): input image.
        detections (list[LayoutDetection]) : list of layout detection results.
        alpha (float, optional): Transparency factor for the filled color.
            Defaults to 0.3.

    Returns:
        npt.NDArray[np.uint8]: OpenCV image with layout detection results displayed
            as overlays.
    """
    img = cv2.cvtColor(
        np.array(image), cv2.COLOR_RGB2BGR
    )  # Convert RGB to BGR for OpenCV
    overlay = img.copy()

    for detection in detections:
        x_min, y_min, x_max, y_max = detection.bbox
        class_name, color = LAYOUT_ELEMENT_TEXT_COLOR[detection.category]
        text = class_name + f":{detection.score:.3f}"

        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Add the class name with a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        cv2.rectangle(
            img,
            (x_min, y_min - text_height - baseline),
            (x_min + text_width, y_min),
            color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img
