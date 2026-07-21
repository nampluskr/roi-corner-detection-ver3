# src/methods/detr/model.py: Hugging Face DETR whole-model detection adapted to 4 corner classes

import os

from src.methods.base.model import BaseModel
from src.components.heads import NUM_CORNER_CLASSES

DETRDET_MODEL_DIR = {
    "detr_resnet50": "/mnt/d/backbones/facebook-detr-resnet-50",
}
SUPPORTED_DETRDET_MODELS = tuple(DETRDET_MODEL_DIR.keys())


class DetrModel(BaseModel):
    """Hugging Face DETR whole detection model adapted to 4 corner classes."""

    def __init__(self, model="detr_resnet50"):
        super().__init__()
        model = model or "detr_resnet50"
        if model not in DETRDET_MODEL_DIR:
            raise ValueError("Unknown detr model: %s. Supported: %s"
                             % (model, ", ".join(SUPPORTED_DETRDET_MODELS)))

        self.model_name = model
        self.net = self.build_model(DETRDET_MODEL_DIR[model])

    def build_model(self, path):
        if not os.path.isdir(path):
            raise FileNotFoundError("Local detr snapshot not found: %s" % path)
        try:
            from transformers import DetrForObjectDetection
        except ImportError as e:
            raise ImportError("DetrModel requires transformers installed in pytorch_env") from e

        id2label = {i: "corner%d" % i for i in range(NUM_CORNER_CLASSES)}
        label2id = {v: k for k, v in id2label.items()}
        return DetrForObjectDetection.from_pretrained(
            path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

    def forward(self, images, labels=None):
        return self.net(pixel_values=images, labels=labels)
