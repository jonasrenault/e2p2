from pathlib import Path
from typing import Annotated

import typer

from e2p2.layout.task import layout_pdf
from e2p2.layout.yolo import LayoutDetectionYOLO
from e2p2.ocr.paddle import ModifiedPaddleOCR
from e2p2.pdf.pdf import PdfDoc, filter_tables
from e2p2.table.rapid_table import RapidTableModel
from e2p2.table.table import TableModel
from e2p2.utils.image import crop_image

app = typer.Typer()


def table_pdf(
    pdf: PdfDoc,
    table_model: TableModel,
    visualize: bool = False,
    save_dir: Path | None = None,
):
    for page_number, image in pdf:
        tables = filter_tables(pdf.get_page_info(page_number).layout_detections)

        for table in tables:
            # Crop image to layout element
            table_image, _ = crop_image(image, table)
            table_content = table_model.predict(table_image)
            table.content = table_content


@app.command()
def table(
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
    Run OCR on given pdf file.

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

    # load OCR model
    ocr_model = ModifiedPaddleOCR(lang="fr")
    # load Table model
    table_model = RapidTableModel(ocr_model, device=device)

    # run Table extraction
    table_pdf(pdf, table_model, visualize, save_dir)
