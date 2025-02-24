from pathlib import Path
from typing import cast

from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from e2p2.layout.layout import LayoutDetectionModel
from e2p2.pdf.pdf import LayoutDetection, LayoutElement


class FormulaDetectionYOLO(LayoutDetectionModel):
    def __init__(
        self,
        model_repo: str = "opendatalab/pdf-extract-kit-1.0",
        model_name: str = "models/MFD/YOLO/yolo_v8_ft.pt",
        model_dir: str | Path | None = None,
        device: str | None = "cpu",
    ):
        """
        Initialize the FormulaDetectionYOLO class. Downloads the model weights from
        HuggingFace Hub using `model_repo` and `model_name`. Optionally stores the
        downloaded model into local folder `model_dir`.

        Args:
            model_repo (str, optional): model repository on HuggingFace Hub.
                Defaults to "opendatalab/pdf-extract-kit-1.0".
            model_name (str, optional): model weights file name within the repostiroy.
                Defaults to "models/MFD/YOLO/yolo_v8_ft.pt".
            model_dir (str | Path | None, optional): local folder path to save the
                model files in. Defaults to None.
            device (str | None, optional): device to run predictions on.
                Defaults to "cpu".
        """
        # Mapping from class IDs to LayoutElement
        self.class_mapping = {
            0: LayoutElement.FORMULA_INLINE,
            1: LayoutElement.FORMULA,
        }

        model = Path(hf_hub_download(model_repo, model_name, local_dir=model_dir))
        self.model = YOLO(model)
        self.device = device

    def predict(
        self,
        image: Image.Image,
        img_size: int = 1280,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> list[LayoutDetection]:
        """
        Run FormulaDetection on given image.

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

        layout_results = []
        for xyxy, conf, cla in zip(boxes.cpu(), scores.cpu(), classes.cpu()):
            layout_result = LayoutDetection(
                bbox=cast(tuple[int, int, int, int], tuple(int(p.item()) for p in xyxy)),
                score=round(float(conf.item()), 2),
                category=self.class_mapping[int(cla.item())],
            )
            layout_results.append(layout_result)

        return layout_results
