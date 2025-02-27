import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from e2p2.pdf.pdf import LayoutDetection, LayoutElement
from e2p2.utils.image import bbox_to_points

LAYOUT_ELEMENT_TEXT_COLOR = {
    LayoutElement.ABANDONED: ("abandon", (141, 211, 199)),
    LayoutElement.UNKNOWN: ("unknown", (190, 186, 218)),
    LayoutElement.TEXT: ("text", (251, 128, 114)),
    LayoutElement.PLAIN_TEXT: ("text", (251, 128, 114)),
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
        npt.NDArray: OpenCV image with layout detection results displayed
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


def visualize_ocr(
    image: Image.Image,
    detections: list[LayoutDetection],
) -> npt.NDArray:
    """
    Visualize the results of OCR detection and recognition. Creates an image
    where the original image is displayed on the left, and the results of OCR
    are displayed on the right.

    Taken from https://github.com/PaddlePaddle/PaddleOCR/blob/
    1d52a2b2e1d89bfc747a4bda0b85aab2e9771031/tools/infer/utility.py#L494

    Args:
        image (Image.Image): input Image.
        detections (list[LayoutDetection]): list of OCR results.

    Returns:
        npt.NDArray: image with OCR results displayed on the right hand side.
    """
    h, w = image.height, image.width
    img_left = image.copy()
    img_right: cv2.typing.MatLike = np.ones((h, w, 3), dtype=np.uint8) * 255
    draw_left = ImageDraw.Draw(img_left)

    for detection in detections:
        if detection.content is None:
            continue

        print(detection.content.text)
        poly = bbox_to_points(detection.bbox)
        _, color = LAYOUT_ELEMENT_TEXT_COLOR[detection.category]

        draw_left.polygon(poly, fill=color)  # type: ignore
        img_right_text = draw_box_txt_fine(w, h, detection.bbox, detection.content.text)
        pts = poly.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img_right_text, [pts], True, color, 1)
        img_right = cv2.bitwise_and(img_right, img_right_text)

    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
    return np.array(img_show)


def draw_box_txt_fine(
    img_width: int, img_height: int, bbox: tuple[int, int, int, int], text: str
) -> npt.NDArray:
    """
    Draw a box with given text inside.

    Args:
        img_width (int): width of the original image the box will be inserted into.
        img_height (int): height of the original image the box will be inserted into.
        bbox (tuple[int, int, int, int]): bounding box to draw.
        text (str): text to draw.

    Returns:
        npt.NDArray: an image of given text and bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    box_height = y_max - y_min
    box_width = x_max - x_min

    portrait = box_height > 2 * box_width and box_height > 30
    text_dim = (box_height, box_width) if portrait else (box_width, box_height)
    img_text = Image.new("RGB", text_dim, (255, 255, 255))
    draw_text = ImageDraw.Draw(img_text)

    font_size = int(text_dim[1] * 0.99)
    font = ImageFont.load_default(font_size)
    if (length := font.getlength(text)) > text_dim[0]:
        font_size = int(font_size * text_dim[0] / length)
        font = ImageFont.load_default(font_size)
    draw_text.text((0.0, 0.0), text, fill=(0, 0, 0), font=font)

    if portrait:
        img_text = img_text.transpose(Image.Transpose.ROTATE_270)

    pts1 = np.array(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]],
        dtype=np.float32,
    )
    pts2 = bbox_to_points(bbox).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_right_text = cv2.warpPerspective(
        np.array(img_text, dtype=np.uint8),
        M,
        (img_width, img_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text
