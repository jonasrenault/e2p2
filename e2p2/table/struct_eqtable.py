from typing import Literal

import torch
from PIL import Image
from struct_eqtable import build_model

from e2p2.pdf.pdf import ContentRecognition
from e2p2.table.table import TableModel


class StructEqTableModel(TableModel):

    def __init__(
        self,
        model_name: str = "U4R/StructTable-InternVL2-1B",
        max_new_tokens: int = 1024,
        max_time: int = 60,
    ):
        assert (
            torch.cuda.is_available()
        ), "CUDA must be available for StructEqTable model."

        self.model = build_model(
            model_ckpt=model_name,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            lmdeploy=False,
            flash_attn=True,
            batch_size=1,
        ).cuda()

    def predict(
        self,
        image: Image.Image,
        format: Literal["html", "latex", "markdown"] = "html",
    ):
        if format not in ["latex", "markdown", "html"]:
            raise ValueError(
                f"Output format {format} is not supported. "
                "Must be one of (latex, html, markdown)"
            )

        with torch.no_grad():
            result = self.model([image], output_format=format)[0]

        return ContentRecognition(text=result, score=0.0)
