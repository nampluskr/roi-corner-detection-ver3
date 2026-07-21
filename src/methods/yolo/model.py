# src/methods/yolo/model.py: Ultralytics YOLOv8 whole-model detection adapted to 4 corner classes

import os
import torch
import torch.nn as nn
from ultralytics.cfg import get_cfg

from src.methods.base.model import BaseModel
from src.components.heads import NUM_CORNER_CLASSES

YOLODET_WEIGHTS = {
    "yolov8n": "/mnt/d/backbones/yolov8n.pt",
}
SUPPORTED_YOLODET_MODELS = tuple(YOLODET_WEIGHTS.keys())


class YoloModel(BaseModel):
    """Ultralytics YOLOv8 whole detection model adapted to 4 corner classes via head replacement."""

    def __init__(self, model="yolov8n"):
        super().__init__()
        model = model or "yolov8n"
        if model not in YOLODET_WEIGHTS:
            raise ValueError("Unknown yolo model: %s. Supported: %s"
                             % (model, ", ".join(SUPPORTED_YOLODET_MODELS)))

        self.model_name = model
        self.net = self.build_model(YOLODET_WEIGHTS[model])

    def build_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local yolo weight not found: %s" % path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = ckpt["model"].float()
        # Ultralytics saves inference checkpoints with requires_grad=False on every
        # parameter (deploy-only assumption); this project fine-tunes the whole net.
        net.requires_grad_(True)
        self.replace_classifier(net, NUM_CORNER_CLASSES)
        net.args = get_cfg()
        return net

    def replace_classifier(self, net, num_classes):
        detect = net.model[-1]
        for seq in detect.cv3:
            in_channels = seq[-1].in_channels
            seq[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        detect.nc = num_classes
        detect.no = num_classes + detect.reg_max * 4
        net.nc = num_classes
        net.names = {i: "corner%d" % i for i in range(num_classes)}
        net.yaml["nc"] = num_classes

    def forward(self, images):
        return self.net(images)
