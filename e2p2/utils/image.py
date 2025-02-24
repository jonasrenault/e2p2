import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image


def get_average_color(image: Image.Image) -> tuple[int, ...]:
    # Convert image to numpy array
    img_array = np.array(image)

    # Get average color, ignoring fully transparent pixels
    if img_array.shape[2] == 4:  # RGBA
        alpha = img_array[:, :, 3]
        rgb = img_array[:, :, :3]
        mask = alpha > 0
        if mask.any():
            avg_color = rgb[mask].mean(axis=0)
        else:
            avg_color = rgb.mean(axis=(0, 1))
    else:  # RGB
        avg_color = img_array.mean(axis=(0, 1))
    return tuple(map(int, avg_color))


def get_contrasting_color(color: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(255 - c for c in color)


def convert_transparent_to_contrasting(image: Image.Image) -> Image.Image:
    """
    Convert transparent pixels to a contrasting color.
    Taken from https://github.com/breezedeus/Pix2Text/
    blob/c87047b1732d55d1e2cce123b768cea52303db77/pix2text/utils.py#L176

    Args:
        image (Image.Image): input image with alpha channel.

    Returns:
        Image.Image: RGB image without alpha channel.
    """
    # Check if the image has an alpha channel
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        # Get average color of non-transparent pixels
        avg_color = get_average_color(image)

        # Get contrasting color for background
        bg_color = get_contrasting_color(avg_color)

        # Create a new background image with the contrasting color
        background = Image.new("RGBA", image.size, bg_color)

        # Paste the image on the background.
        # If the image has an alpha channel, it will be used as a mask
        background.paste(image, (0, 0), image)

        # Convert to RGB (removes alpha channel)
        return background.convert("RGB")

    return image.convert("RGB")


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
