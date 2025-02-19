from pathlib import Path
from typing import Annotated

import cv2
import typer

from e2p2.layout.layout import LayoutDetectionModel, visualize_bbox
from e2p2.layout.yolo import LayoutDetectionYOLO
from e2p2.pdf.pdf import PageInfo, PdfDoc

app = typer.Typer()


def layout_pdf(
    pdf: PdfDoc,
    layout_model: LayoutDetectionModel,
    visualize: bool = False,
    save_dir: Path | None = None,
):
    """
    Run layout detection on given pdf document.

    Args:
        pdf (PdfDoc): pdf document.
        layout_model (LayoutDetectionModel): layout detection model.
        visualize (bool, optional): visualize results. Defaults to False.
        save_dir (Path | None, optional): output directory where results are saved.
            Defaults to None.
    """
    page_infos: list[PageInfo] = []
    for page_number, image in pdf:
        img_W, img_H = image.size
        layout_detections = layout_model.predict(image)
        page_infos.append(
            PageInfo(
                page_number=page_number,
                height=img_H,
                width=img_W,
                layout_detections=layout_detections,
            )
        )
    pdf.page_infos = page_infos

    if visualize and save_dir is not None:
        for page_info in page_infos:
            image = pdf.get_image(page_info.page_number)
            if page_info.layout_detections is not None:
                vis_result = visualize_bbox(image, page_info.layout_detections)
                layout_file_name = (
                    f"{pdf.page_file_stem(page_info.page_number)}_layout.png"
                )
                cv2.imwrite(save_dir / layout_file_name, vis_result)


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
    """
    # load pdf
    pdf = PdfDoc(input, dpi, pages)

    # load LayoutDetectionModel
    model = LayoutDetectionYOLO()

    # layout pdf
    layout_pdf(pdf, model, visualize, save_dir)


if __name__ == "__main__":
    model = LayoutDetectionYOLO()

    pdf = PdfDoc(Path("resources/data/test-doc.pdf"))
    layout_pdf(pdf, model, True, Path("resources/data"))
