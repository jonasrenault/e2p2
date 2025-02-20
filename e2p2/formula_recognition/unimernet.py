import argparse
from pathlib import Path
from typing import Literal

import torch
import unimernet.tasks as tasks
from huggingface_hub import snapshot_download
from PIL import Image
from unimernet.common.config import Config
from unimernet.processors import load_processor

from e2p2.formula_recognition.mfr import FormulaRecognitionModel, latex_rm_whitespace


class FormulaRecognitionUniMERNet(FormulaRecognitionModel):
    def __init__(
        self,
        model_repo: Literal[
            "wanderkid/unimernet_tiny",
            "wanderkid/unimernet_small",
            "wanderkid/unimernet_base",
        ] = "wanderkid/unimernet_small",
        model_name: Literal[
            "unimernet_tiny.pth", "unimernet_small.pth", "pytorch_model.pth"
        ] = "unimernet_small.pth",
        model_dir: str | Path | None = None,
        device: str | torch.device = "cpu",
    ):
        """
        Initialize the FormulaRecognitionUniMERNet class. Downloads the model weights
        from HuggingFace Hub using `model_repo` and `model_name`. Optionally stores the
        downloaded model into local folder `model_dir`. `model_name` must match
        `model_repo` (wanderkid/unimernet_tiny goes with unimernet_tiny.pth).

        Args:
            model_repo (Literal["wanderkid/unimernet_tiny", "wanderkid/unimernet_small",
                "wanderkid/unimernet_base"], optional): unimernet model repo.
                Defaults to "wanderkid/unimernet_small".
            model_name (Literal["unimernet_tiny.pth", "unimernet_small.pth",
                "pytorch_model.pth"], optional): unimernet model name. Should match
                model_repo Defaults to "unimernet_small.pth".
            model_dir (str | Path | None, optional): local folder path to save the
                model files in. Defaults to None.
            device (str | torch.device, optional): device to run predictions on.
                Defaults to "cpu".
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        model_dir = Path(snapshot_download(model_repo, local_dir=model_dir))
        cfg_path = Path(__file__).parent / "config/unimernet.yaml"
        self.model, self.vis_processor = self.load_model_and_processor(
            model_dir, model_name, cfg_path
        )

    def load_model_and_processor(self, model_dir: Path, model_name: str, cfg_path: Path):
        args = argparse.Namespace(cfg_path=cfg_path, options=None)
        cfg = Config(args)
        cfg.config.model.pretrained = str(model_dir / model_name)
        cfg.config.model.model_config.model_name = str(model_dir)
        cfg.config.model.tokenizer_config.path = str(model_dir)
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg).to(self.device)
        vis_processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )
        return model, vis_processor

    def predict(self, image: Image.Image) -> str:
        image = self.vis_processor(image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image})
        pred = output["pred_str"][0]
        return latex_rm_whitespace(pred)
