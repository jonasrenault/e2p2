from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from e2p2.pdf.pdf import LayoutDetection


@dataclass
class CropedImageInfo:
    padding_x: int
    padding_y: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    croped_width: int
    croped_height: int


def crop_image(
    image: Image.Image,
    layout_detection: LayoutDetection,
    padding_x: int = 0,
    padding_y: int = 0,
) -> tuple[Image.Image, CropedImageInfo]:
    """
    Crop an Image to the bbox of a LayoutDetection, with optional padding on the x and y
    axis (filled with white).

    Args:
        image (Image.Image): the input image.
        layout_detection (LayoutDetection): the layout detection to crop to.
        padding_x (int, optional): padding on the x axis. Defaults to 0.
        padding_y (int, optional): padding on the y axis. Defaults to 0.

    Returns:
        tuple[Image.Image, CroppedImageInfo]: the cropped image and CropImageInfo.
    """
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = layout_detection.bbox

    # Create a white background with an additional width and height of 2 * padding
    crop_new_width = crop_xmax - crop_xmin + padding_x * 2
    crop_new_height = crop_ymax - crop_ymin + padding_y * 2
    cropped_image = Image.new("RGB", (crop_new_width, crop_new_height), "white")
    # and paste the cropped image into it
    cropped_image.paste(image.crop(layout_detection.bbox), (padding_x, padding_y))

    cropped_info = CropedImageInfo(
        padding_x,
        padding_y,
        crop_xmin,
        crop_ymin,
        crop_xmax,
        crop_ymax,
        crop_new_width,
        crop_new_height,
    )
    return cropped_image, cropped_info


def binarize_img(img: npt.NDArray) -> npt.NDArray:
    """
    Binarize cv2 image to black and white.
    Taken from https://github.com/PaddlePaddle/PaddleOCR/blob/
    54e53cc45544d56b089bdc32b5661b1400a65c59/ppocr/utils/utility.py#L97

    Args:
        img (npt.NDArray): the input image.

    Returns:
        npt.NDArray: binarized image.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversion to grayscale image
        # use cv2 threshold binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(
    img: npt.NDArray, alpha_color: tuple[int, int, int] = (255, 255, 255)
) -> npt.NDArray:
    """
    Set alpha chanel to color.
    Taken from https://github.com/PaddlePaddle/PaddleOCR/blob/
    54e53cc45544d56b089bdc32b5661b1400a65c59/ppocr/utils/utility.py#L106

    Args:
        img (npt.NDArray): input image
        alpha_color (tuple[int, int, int], optional): RGB color for transparent pixels
            replacement. Defaults to (255, 255, 255) (pure white).

    Returns:
        npt.NDArray: modified image.
    """
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def bbox_to_points(bbox: tuple[float, float, float, float]) -> npt.NDArray[np.float32]:
    """
    Change bounding box (xmin, ymin, xmax, ymax) to polygon (coordinates of 4 corners).

    Args:
        bbox (tuple[float, float, float, float]): xmin, ymin, xmax, ymax array

    Returns:
        npt.NDArray[np.float32]: array of corner coordinates
    """
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype("float32")


def points_to_bbox(points: npt.NDArray[np.float32]) -> tuple[float, float, float, float]:
    """
    Change polygon (array of corner coordinates) to bounding box
    (xmin, ymin, xmax, ymax).

    Args:
        points (npt.NDArray[np.float32]): list of corner coordinates.

    Returns:
        tuple[float, float, float, float]: BBox
    """

    x0, y0 = points[0]
    x1, _ = points[1]
    _, y1 = points[2]
    return (x0, y0, x1, y1)
