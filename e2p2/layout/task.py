from pathlib import Path
from typing import Annotated

import cv2
import typer

from e2p2.layout.layout import LayoutDetectionModel
from e2p2.layout.yolo import LayoutDetectionYOLO
from e2p2.pdf.pdf import PageInfo, PdfDoc
from e2p2.utils.visualize import visualize_bbox

app = typer.Typer()


def layout_pdf(
    pdf: PdfDoc,
    layout_model: LayoutDetectionModel,
    visualize: bool = False,
    save_dir: Path | None = None,
    img_size: int = 1280,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    task_name: str = "layout",
):
    """
    Run layout detection on given pdf document.

    Args:
        pdf (PdfDoc): pdf document.
        layout_model (LayoutDetectionModel): layout detection model.
        visualize (bool, optional): visualize results. Defaults to False.
        save_dir (Path | None, optional): output directory where results are saved.
            Defaults to None.
        img_size (int, optional): Prediction image size. Defaults to 1280.
        conf (float, optional): Confidence threshold. Defaults to 0.25.
        iou (float, optional): NMS IoU threshold. Defaults to 0.45.
        task_name (str, optional): Detection task name. Defaults to "layout".
    """
    page_infos: list[PageInfo] = []
    for page_number, image in pdf:
        img_W, img_H = image.size
        layout_detections = layout_model.predict(
            image, img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres
        )
        pdf.get_page_info(page_number).layout_detections.extend(layout_detections)
        page_infos.append(
            PageInfo(
                page_number=page_number,
                width=img_W,
                height=img_H,
                layout_detections=layout_detections,
            )
        )

    if visualize and save_dir is not None:
        for page_info in page_infos:
            image = pdf.get_image(page_info.page_number)
            vis_result = visualize_bbox(image, page_info.layout_detections)
            layout_file_name = (
                f"{pdf.page_file_stem(page_info.page_number)}_{task_name}.png"
            )
            cv2.imwrite(str(save_dir / layout_file_name), vis_result)


@app.command()
def layout(
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
    Run layout detection on given pdf file.

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

    # layout pdf
    layout_pdf(pdf, model, visualize, save_dir, img_size, conf, iou)
