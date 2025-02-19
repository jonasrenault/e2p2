from pathlib import Path
from typing import Annotated

import typer

from e2p2.formula_detection.yolo import FormulaDetectionYOLO
from e2p2.layout.task import layout_pdf
from e2p2.pdf.pdf import PdfDoc

app = typer.Typer()


@app.command()
def mfd(
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
    Run formula detection on given pdf file.

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
    model = FormulaDetectionYOLO(device=device)

    # layout pdf
    layout_pdf(pdf, model, visualize, save_dir, img_size, conf, iou, task_name="MFD")
