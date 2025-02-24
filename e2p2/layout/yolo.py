from pathlib import Path
from typing import cast

import numpy as np
import torch
import torchvision
from doclayout_yolo import YOLOv10
from huggingface_hub import snapshot_download
from PIL import Image

from e2p2.layout.layout import LayoutDetectionModel
from e2p2.pdf.pdf import LayoutDetection, LayoutElement


class LayoutDetectionYOLO(LayoutDetectionModel):
    def __init__(
        self,
        model_repo: str = "juliozhao/DocLayout-YOLO-DocStructBench",
        model_name: str = "doclayout_yolo_docstructbench_imgsz1024.pt",
        model_dir: str | Path | None = None,
        device: str | None = "cpu",
    ):
        """
        Initialize the LayoutDetectionYOLO class. Downloads the model weights from
        HuggingFace Hub using `model_repo` and `model_name`. Optionally stores the
        downloaded model into local folder `model_dir`.

        Args:
            model_repo (str, optional): model repository on HuggingFace Hub.
                Defaults to "juliozhao/DocLayout-YOLO-DocStructBench".
            model_name (str, optional): model weights file name within the repostiroy.
                Defaults to "doclayout_yolo_docstructbench_imgsz1024.pt".
            model_dir (str | Path | None, optional): local folder path to save the
                model files in. Defaults to None.
            device (str | None, optional): device to run predictions on.
                Defaults to "cpu".
        """
        # Mapping from class IDs to LayoutElement
        self.class_mapping = {
            0: LayoutElement.TITLE,
            1: LayoutElement.TEXT,
            2: LayoutElement.ABANDONED,
            3: LayoutElement.FIGURE,
            4: LayoutElement.FIGURE_CAPTION,
            5: LayoutElement.TABLE,
            6: LayoutElement.TABLE_CAPTION,
            7: LayoutElement.TABLE_FOOTNOTE,
            8: LayoutElement.FORMULA,
            9: LayoutElement.FORMULA_CAPTION,
        }

        model_dir = Path(snapshot_download(model_repo, local_dir=model_dir))
        self.model = YOLOv10(model_dir / model_name)
        self.device = device

    def predict(
        self,
        image: Image.Image,
        img_size: int = 1280,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> list[LayoutDetection]:
        """
        Run LayoutDetection on given image.

        Args:
            image (Image.Image): image.
            img_size (int, optional): Prediction image size. Defaults to 1280.
            conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): NMS IoU threshold. Defaults to 0.45.

        Returns:
            list[LayoutDetection]: list of LayoutDetections.
        """
        result = self.model.predict(
            image,
            imgsz=img_size,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
            device=self.device,
        )[0]

        boxes = result.__dict__["boxes"].xyxy
        classes = result.__dict__["boxes"].cls
        scores = result.__dict__["boxes"].conf

        if iou_thres > 0:
            indices = torchvision.ops.nms(
                boxes=torch.Tensor(boxes),
                scores=torch.Tensor(scores),
                iou_threshold=iou_thres,
            )
            boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
            if len(boxes.shape) == 1:
                boxes = np.expand_dims(boxes, 0)
                scores = np.expand_dims(scores, 0)
                classes = np.expand_dims(classes, 0)

        layout_results = []
        for xyxy, conf, cla in zip(boxes.cpu(), scores.cpu(), classes.cpu()):
            layout_result = LayoutDetection(
                bbox=cast(tuple[int, int, int, int], tuple(int(p.item()) for p in xyxy)),
                score=round(float(conf.item()), 2),
                category=self.class_mapping[int(cla.item())],
            )
            layout_results.append(layout_result)

        return layout_results
