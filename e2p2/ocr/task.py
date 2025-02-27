from pathlib import Path
from typing import Annotated, cast

import numpy as np
import numpy.typing as npt
import typer

from e2p2.layout.task import layout_pdf
from e2p2.layout.yolo import LayoutDetectionYOLO
from e2p2.mfd.yolo import FormulaDetectionYOLO
from e2p2.ocr.ocr import OCRModel
from e2p2.ocr.paddle import ModifiedPaddleOCR, bbox_to_points, points_to_bbox
from e2p2.pdf.pdf import LayoutDetection, PdfDoc, filter_formulas, filter_ocr_elements
from e2p2.utils.image import CropedImageInfo, crop_image

app = typer.Typer()


def get_adjusted_formula_bboxes(
    formulas: list[LayoutDetection], croped_image_info: CropedImageInfo
) -> list[tuple[int, int, int, int]]:
    """
    Get the bounding boxes of formula detections relative to the cropped image
    being passed to OCR.

    Args:
        formulas (list[LayoutDetection]): list of formula detection boxes.
        croped_image_info (CropedImageInfo): croped image infos.

    Returns:
        list[tuple[int, int, int, int]]: list of bounding boxes relative to the croped
        image.
    """
    # Adjust the coordinates of the formula area
    adjusted_formula_bboxes = []
    for formula in formulas:
        mf_xmin, mf_ymin, mf_xmax, mf_ymax = formula.bbox
        # Adjust the coordinates of the formula area to the coordinates relative
        # to the cropping area
        x0 = mf_xmin - croped_image_info.xmin + croped_image_info.padding_x
        y0 = mf_ymin - croped_image_info.ymin + croped_image_info.padding_y
        x1 = mf_xmax - croped_image_info.xmin + croped_image_info.padding_x
        y1 = mf_ymax - croped_image_info.ymin + croped_image_info.padding_y
        # Filter formula blocks outside the croped image
        if any([x1 < 0, y1 < 0]) or any(
            [x0 > croped_image_info.croped_width, y0 > croped_image_info.croped_height]
        ):
            continue
        else:
            adjusted_formula_bboxes.append((x0, y0, x1, y1))
    return adjusted_formula_bboxes


def is_angle(poly: npt.NDArray[np.float32]) -> bool:
    """
    Check if polygon is at an angle.

    Args:
        poly (npt.NDArray[np.float32]): 4 corner polygon

    Returns:
        bool: True if polygon is rotated.
    """
    p1, p2, p3, p4 = poly
    height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
    if 0.8 * height <= (p3[1] - p1[1]) <= 1.2 * height:
        return False
    else:
        return True


def convert_ocr_results(
    ocr_results: list[LayoutDetection],
    croped_image_info: CropedImageInfo,
):
    """
    Convert OCR results back to original coordinate system of the page.

    Args:
        ocr_results (list[LayoutDetection]): ocr results.
        croped_image_info (CropedImageInfo): croped image info.
    """
    for ocr_result in ocr_results:
        poly = bbox_to_points(ocr_result.bbox)
        p1, p2, p3, p4 = poly

        if is_angle(poly):
            x_center = sum(point[0] for point in poly) / 4
            y_center = sum(point[1] for point in poly) / 4
            new_height = ((p4[1] - p1[1]) + (p3[1] - p2[1])) / 2
            new_width = p3[0] - p1[0]
            p1 = [x_center - new_width / 2, y_center - new_height / 2]
            p2 = [x_center + new_width / 2, y_center - new_height / 2]
            p3 = [x_center + new_width / 2, y_center + new_height / 2]
            p4 = [x_center - new_width / 2, y_center + new_height / 2]

        # Convert the coordinates back to the original coordinate system
        p1 = [
            p1[0] - croped_image_info.padding_x + croped_image_info.xmin,
            p1[1] - croped_image_info.padding_y + croped_image_info.ymin,
        ]
        p2 = [
            p2[0] - croped_image_info.padding_x + croped_image_info.xmin,
            p2[1] - croped_image_info.padding_y + croped_image_info.ymin,
        ]
        p3 = [
            p3[0] - croped_image_info.padding_x + croped_image_info.xmin,
            p3[1] - croped_image_info.padding_y + croped_image_info.ymin,
        ]
        p4 = [
            p4[0] - croped_image_info.padding_x + croped_image_info.xmin,
            p4[1] - croped_image_info.padding_y + croped_image_info.ymin,
        ]

        ocr_result.bbox = cast(
            tuple[int, int, int, int],
            tuple(map(int, points_to_bbox(np.array([p1, p2, p3, p4])))),
        )


def ocr_pdf(
    pdf: PdfDoc,
    ocr_model: OCRModel,
    visualize: bool = False,
    save_dir: Path | None = None,
):
    for page_number, image in pdf:
        formulas = filter_formulas(pdf.get_page_info(page_number).layout_detections)
        ocrs = filter_ocr_elements(pdf.get_page_info(page_number).layout_detections)

        for ocr_element in ocrs:
            # Crop image to layout element
            ocr_image, ocr_image_info = crop_image(image, ocr_element, 50, 50)
            # Get formula bounding boxes in the croped area
            formula_bboxes = get_adjusted_formula_bboxes(formulas, ocr_image_info)
            # OCR recognition
            ocr_results = ocr_model.predict(ocr_image, formula_bboxes=formula_bboxes)
            # Convert results back to original coordinates
            convert_ocr_results(ocr_results, ocr_image_info)

            pdf.get_page_info(page_number).layout_detections.extend(ocr_results)


@app.command()
def ocr(
    input: Annotated[
        Path,
        typer.Argument(
            help="Path to input pdf.",
            exists=True,
            file_okay=True,
            readable=True,
        ),
    ],
    save_dir: Annotated[
        Path | None,
        typer.Option(
            help="Output directory where results are saved.",
            exists=True,
            dir_okay=True,
            file_okay=False,
            writable=True,
        ),
    ] = None,
    visualize: Annotated[bool, typer.Option(help="Visualize results.")] = False,
    dpi: Annotated[int, typer.Option(help="DPI for image rasterization.")] = 96,
    pages: Annotated[
        list[int] | None, typer.Option(help="list of pages to process.")
    ] = None,
    img_size: Annotated[int, typer.Option(help="prediction image size")] = 1280,
    conf: Annotated[float, typer.Option(help="prediction image size")] = 0.25,
    iou: Annotated[float, typer.Option(help="prediction image size")] = 0.45,
    device: Annotated[str, typer.Option(help="device to use for predictions")] = "cpu",
):
    """
    Run formula recognition on given pdf file.

    Args:
        input (Path): path to input pdf.
        save_dir (Path | None, optional): Output directory where results are saved.
            Defaults to None.
        visualize (bool | None, optional): Visualize results. Defaults to False.
        dpi (int, optional): DPI for image rasterization. Defaults to 96.
        pages (list[int] | None, optional): list of pages to process. Defaults to None.
        img_size (int, optional): Prediction image size. Defaults to 1280.
        conf (float, optional): Confidence threshold. Defaults to 0.25.
        iou (float, optional): NMS IoU threshold. Defaults to 0.45.
        device (str, optional): Device to use for predictions. Defaults to "cpu".
    """
    # load pdf
    pdf = PdfDoc(input, dpi, pages)

    # load LayoutDetectionModel
    model = LayoutDetectionYOLO(device=device)

    # run layout detection: detect tables, text, figures, ...
    layout_pdf(pdf, model, visualize, save_dir, img_size, conf, iou)

    # load FormulaDetection model
    mfd_model = FormulaDetectionYOLO(device=device)

    # run MFD: detect formulas in pdf
    layout_pdf(pdf, mfd_model, visualize, save_dir, img_size, conf, iou, task_name="MFD")

    # load OCR model
    ocr_model = ModifiedPaddleOCR()

    # run OCR: extract text in pdf
    ocr_pdf(pdf, ocr_model, visualize, save_dir)
