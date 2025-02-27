from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
from paddleocr import PaddleOCR
from PIL import Image

from e2p2.ocr.ocr import OCRModel
from e2p2.pdf.pdf import ContentRecognition, LayoutDetection, LayoutElement
from e2p2.utils.image import alpha_to_color, binarize_img

type BBox = tuple[float, float, float, float]
type Interval = tuple[float, float]


def get_rotate_crop_image(
    img: npt.NDArray, points: npt.NDArray[np.float32]
) -> npt.NDArray:
    """
    Crop an image to the given polygon, eventually rotating it as well.

    Args:
        img (npt.NDArray): the image.
        points (npt.NDArray[np.float32]): the croping area.

    Returns:
        npt.NDArray: the croped image.
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
    )
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(
    img: npt.NDArray, points: npt.NDArray[np.float32]
) -> npt.NDArray:
    """
    Crop an image to the minimum area rectangle covering the polygon points.

    Args:
        img (npt.NDArray): the image.
        points (npt.NDArray[np.float32]): the croping points.

    Returns:
        npt.NDArray: the croped image.
    """
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_d = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    index_b, index_c = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


def sorted_boxes(dt_boxes: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
    """
    Sort text boxes in order from top to bottom, left to right.

    Args:
        dt_boxes (npt.NDArray): array of detected text boxes, each box as a polygon
            of shape (4, 2).

    Returns:
        list[npt.NDArray[np.float32]]: sorted array of boxes.
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def __is_overlaps_y_exceeds_threshold(
    bbox1: BBox, bbox2: BBox, overlap_ratio_threshold=0.8
) -> bool:
    """
    Check if two bounding boxes overlap on the y-axis, and if the height of the
    overlapping region exceeds 80% of the height of the shorter bounding box.

    Args:
        bbox1 (BBox): xmin, ymin, xmax, ymax array
        bbox2 (BBox): xmin, ymin, xmax, ymax array
        overlap_ratio_threshold (float, optional): threshold of height that must overlap.
            Defaults to 0.8.

    Returns:
        bool: True if more than `overlap_ratio_threshold` of bounding boxes heights
            overlap.
    """
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2

    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    height1, height2 = y1_1 - y0_1, y1_2 - y0_2
    min_height = min(height1, height2)

    return (overlap / min_height) > overlap_ratio_threshold


def bbox_to_points(bbox: BBox) -> npt.NDArray[np.float32]:
    """
    Change bounding box (xmin, ymin, xmax, ymax) to polygon (coordinates of 4 corners).

    Args:
        bbox (BBox): xmin, ymin, xmax, ymax array

    Returns:
        npt.NDArray[np.float32]: array of corner coordinates
    """
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype("float32")


def points_to_bbox(points: npt.NDArray[np.float32]) -> BBox:
    """
    Change polygon (array of corner coordinates) to bounding box
    (xmin, ymin, xmax, ymax).

    Args:
        points (npt.NDArray[np.float32]): list of corner coordinates.

    Returns:
        list[float]: BBox
    """

    x0, y0 = points[0]
    x1, _ = points[1]
    _, y1 = points[2]
    return (x0, y0, x1, y1)


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """
    Merge overlapping intervals.

    Args:
        intervals (list[Interval]): a list of one dimension intervals
            [(xmin, xmax), (xmin, xmax), ...]

    Returns:
        list[Interval]: list of overlapping intervals merged.
    """
    # Sort the intervals based on the start value
    intervals.sort(key=lambda x: x[0])

    merged: list[Interval] = []
    for interval in intervals:
        # If the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # Otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

    return merged


def remove_intervals(original: Interval, masks: list[Interval]) -> list[Interval]:
    """
    Return the original interval (xmin, xmax) minus all the intervals in masks.

    Args:
        original (Interval): a horizontal interval (xmin, xmax)
        masks (list[Interval]): a list of horizontal intervals
            [(xmin, xmax), (xmin, xmax), ...]

    Returns:
        list[Interval]: original - masks
    """
    # Merge all mask intervals
    merged_masks = merge_intervals(masks)

    result: list[Interval] = []
    original_start, original_end = original

    for mask in merged_masks:
        mask_start, mask_end = mask

        # If the mask starts after the original range, ignore it
        if mask_start > original_end:
            continue

        # If the mask ends before the original range starts, ignore it
        if mask_end < original_start:
            continue

        # Remove the masked part from the original range
        if original_start < mask_start:
            result.append((original_start, mask_start - 1))

        original_start = max(mask_end + 1, original_start)

    # Add the remaining part of the original range, if any
    if original_start <= original_end:
        result.append((original_start, original_end))

    return result


def update_det_boxes(
    dt_boxes: list[npt.NDArray[np.float32]],
    mfd_detections: list[tuple[int, int, int, int]],
) -> list[npt.NDArray[np.float32]]:
    """
    Given a list of detected text boxes, remove the spans corresponding to formulas
    which have already been recognized

    Args:
        dt_boxes (list[npt.NDArray[np.float32]]): list of detected text bounding boxes
        mfd_detections (list[tuple[int, int, int, int]]): list of layout detecting
            containing bounding boxes for formulas which already have been ocr'd.

    Returns:
        list[npt.NDArray[np.float32]]: updated list of text bounding boxes.
    """
    new_dt_boxes = []
    for text_box in dt_boxes:
        text_bbox = points_to_bbox(text_box)
        masks_list: list[Interval] = []

        for mf_box in mfd_detections:
            mf_bbox = cast(tuple[float, float, float, float], tuple(map(float, mf_box)))
            if __is_overlaps_y_exceeds_threshold(text_bbox, mf_bbox):
                masks_list.append((mf_bbox[0], mf_bbox[2]))

        text_x_range = (text_bbox[0], text_bbox[2])
        text_remove_mask_range = remove_intervals(text_x_range, masks_list)
        temp_dt_box = []
        for text_remove_mask in text_remove_mask_range:
            temp_dt_box.append(
                bbox_to_points(
                    (text_remove_mask[0], text_bbox[1], text_remove_mask[1], text_bbox[3])
                )
            )
        if len(temp_dt_box) > 0:
            new_dt_boxes.extend(temp_dt_box)
    return new_dt_boxes


def merge_spans_to_line(spans: list[BBox]) -> list[list[BBox]]:
    """
    Merge given spans into lines. Spans are considered based on their position
    in the document. If spans overlap sufficiently on the Y-axis, they are merged
    into the same line; otherwise, a new line is started.

    Args:
        spans (list[BBox]): A list of spans where each span is bounding box
            (xmin,ymin, xmax, ymax)

    Returns:
        list[list[BBox]]: A list of lines, where each line is a list of spans.
    """
    # Return an empty list if the spans list is empty
    if len(spans) == 0:
        return []
    else:
        # Sort spans by the Y0 coordinate
        spans.sort(key=lambda span: span[1])

        lines: list[list[BBox]] = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # If the current span overlaps with the last span in the current line on the
            # Y-axis, add it to the current line
            if __is_overlaps_y_exceeds_threshold(span, current_line[-1]):
                current_line.append(span)
            else:
                # Otherwise, start a new line
                lines.append(current_line)
                current_line = [span]

        # Add the last line if it exists
        if current_line:
            lines.append(current_line)

        return lines


def merge_overlapping_spans(spans: list[BBox]) -> list[BBox]:
    """
    Merges overlapping spans on the same line.

    Args:
        spans (list[BBox]): A list of span bounding boxes
            [(xmin, ymin, xmax, ymax), ...].

    Returns:
        list[BBox]: A list of merged spans.
    """
    # Return an empty list if the input spans list is empty
    if not spans:
        return []

    # Sort spans by their starting x-coordinate
    spans.sort(key=lambda x: x[0])

    # Initialize the list of merged spans
    merged: list[BBox] = []
    for span in spans:
        # Unpack span coordinates
        x1, y1, x2, y2 = span
        # If the merged list is empty or there's no horizontal overlap,
        # add the span directly
        if not merged or merged[-1][2] < x1:
            merged.append(span)
        else:
            # If there is horizontal overlap, merge the current span with
            # the previous one
            last_span = merged.pop()
            # Update the merged span's top-left corner to the smaller (x1, y1)
            # and bottom-right to the larger (x2, y2)
            x1 = min(last_span[0], x1)
            y1 = min(last_span[1], y1)
            x2 = max(last_span[2], x2)
            y2 = max(last_span[3], y2)
            # Add the merged span back to the list
            merged.append((x1, y1, x2, y2))

    # Return the list of merged spans
    return merged


def merge_det_boxes(
    dt_boxes: list[npt.NDArray[np.float32]],
) -> list[npt.NDArray[np.float32]]:
    """
    Merge detection boxes, each represented by its corner coordinates. Merge boxes
    into larger text regions.

    Args:
        dt_boxes (list[npt.NDArray[np.float32]]): A list of detected text boxes.

    Returns:
        list[npt.NDArray[np.float32]]: list of merged text regions.
    """

    # Convert the detection boxes into bounding boxes
    dt_bboxes: list[BBox] = [points_to_bbox(text_box) for text_box in dt_boxes]

    # Merge adjacent text regions into lines
    lines = merge_spans_to_line(dt_bboxes)

    # Initialize a new list for storing the merged text regions
    new_dt_boxes: list[npt.NDArray[np.float32]] = []
    for line in lines:
        # Merge overlapping text regions within the same line
        merged_spans = merge_overlapping_spans(line)

        # Convert the merged text regions back to point format
        # and add them to the new detection box list
        for span in merged_spans:
            new_dt_boxes.append(bbox_to_points(span))

    return new_dt_boxes


def preprocess_image(
    image: npt.NDArray, alpha_color: tuple[int, int, int], inv: bool, bin: bool
) -> npt.NDArray:
    """
    Preprocess image for OCR by removing alpha chanel and applying inversion and
    binarizing if `bin` or `inv` are True.

    Args:
        image (npt.NDArray): input cv2 image.
        alpha_color (tuple[int, int, int]): RGB color for alpha replacement.
        inv (bool): if True, invert image.
        bin (bool): if True, binarize image.

    Returns:
        npt.NDArray: preprocessed image.
    """
    image = alpha_to_color(image, alpha_color)
    if inv:
        image = cv2.bitwise_not(image)
    if bin:
        image = binarize_img(image)
    return image


class ModifiedPaddleOCR(PaddleOCR, OCRModel):
    def __init__(
        self,
        show_log: bool = True,
        det_db_box_thresh: float = 0.3,
        lang: str | None = None,
        use_dilation: bool = True,
        det_db_unclip_ratio: float = 1.8,
        *args,
        **kwargs,
    ):
        kwargs["show_log"] = show_log
        kwargs["det_db_box_thresh"] = det_db_box_thresh
        kwargs["use_dilation"] = use_dilation
        kwargs["det_db_unclip_ratio"] = det_db_unclip_ratio
        if lang is not None and lang != "":
            kwargs["lang"] = lang
        else:
            kwargs.pop("lang", None)
        super().__init__(*args, **kwargs)

    def predict(
        self,
        image: Image.Image,
        formula_bboxes: list[tuple[int, int, int, int]] | None = None,
        ocr_drop_score: float = 0.6,
        cls: bool = True,
        bin: bool = False,
        inv: bool = False,
        alpha_color=(255, 255, 255),
        *args,
        **kwargs,
    ) -> list[LayoutDetection]:
        img = cv2.cvtColor(
            np.array(image), cv2.COLOR_RGB2BGR
        )  # Convert RGB to BGR for OpenCV

        img = preprocess_image(img, alpha_color, inv, bin)

        ori_img = img.copy()

        # first run text detection to detect text boxes
        dt_boxes, _ = self.text_detector(img)

        if dt_boxes is None:
            return []

        dt_boxes = sorted_boxes(dt_boxes)
        dt_boxes = merge_det_boxes(dt_boxes)

        if formula_bboxes:
            dt_boxes = update_det_boxes(dt_boxes, formula_bboxes)

        img_crop_list = [
            (
                get_rotate_crop_image(ori_img, dt_box)
                if self.args.det_box_type == "quad"
                else get_minarea_rect_crop(ori_img, dt_box)
            )
            for dt_box in dt_boxes
        ]
        if self.use_angle_cls and cls:
            img_crop_list, _ = self.text_classifier(img_crop_list)

        rec_res, _ = self.text_recognizer(img_crop_list)
        results = []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score > ocr_drop_score:
                results.append(
                    LayoutDetection(
                        cast(
                            tuple[int, int, int, int],
                            tuple(map(int, points_to_bbox(box))),
                        ),
                        score=0.0,
                        category=LayoutElement.TEXT,
                        content=ContentRecognition(text, score),
                    )
                )
        return results
