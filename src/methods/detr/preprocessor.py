# src/methods/detr/preprocessor.py: convert standard corners into DETR label targets

import torch

from src.methods.base.preprocessor import BasePreprocessor

NUM_CORNER_CLASSES = 4


class DetrPreprocessor(BasePreprocessor):
    """Turns each of the 4 corners into a fixed-size normalized pseudo-box label for DETR."""

    def __init__(self, box_size=0.1):
        self.box_size = box_size

    def __call__(self, corners):
        labels = torch.arange(NUM_CORNER_CLASSES, dtype=torch.long, device=corners.device)
        targets = []
        for sample in corners:
            centers = sample[:NUM_CORNER_CLASSES].clamp(0.0, 1.0)
            wh = torch.full((NUM_CORNER_CLASSES, 2), self.box_size, device=corners.device)
            boxes = torch.cat([centers, wh], dim=1)
            targets.append({"class_labels": labels, "boxes": boxes})
        return targets
