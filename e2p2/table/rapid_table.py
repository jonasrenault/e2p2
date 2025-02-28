from typing import Literal

import numpy as np
from PIL import Image
from rapid_table import RapidTable, RapidTableInput

from e2p2.ocr.ocr import OCRModel
from e2p2.pdf.pdf import ContentRecognition
from e2p2.table.table import TableModel
from e2p2.utils.image import bbox_to_points


class RapidTableModel(TableModel):
    def __init__(
        self, ocr_model: OCRModel, model_type: str = "unitable", device: str | None = None
    ):
        use_cuda = device is not None and device.startswith("cuda")
        self.model = RapidTable(
            RapidTableInput(
                model_type=model_type,
                use_cuda=use_cuda,
                device=device if use_cuda else "cpu",
            )
        )
        self.ocr_model = ocr_model

    def predict(
        self,
        image: Image.Image,
        format: Literal["html", "latex", "markdown"] = "html",
    ) -> ContentRecognition:
        if format != "html":
            raise ValueError(
                f"Output format {format} is not supported. "
                "RapidTable can only predict html."
            )

        ocr_results = self.ocr_model.predict(image)
        ocr_results_for_table = []
        for ocr_result in ocr_results:
            if ocr_result.content is None:
                continue

            poly = bbox_to_points(ocr_result.bbox)
            ocr_results_for_table.append(
                [poly, ocr_result.content.text, ocr_result.content.score]
            )

        table_results = self.model(np.asarray(image), ocr_results_for_table)

        return ContentRecognition(
            text=table_results.pred_html,
            score=0.0,
            cell_bboxes=table_results.cell_bboxes,
            logic_points=table_results.logic_points,
        )
