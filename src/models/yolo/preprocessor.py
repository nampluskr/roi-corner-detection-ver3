# src/models/yolo/preprocessor.py: convert standard corners into Ultralytics loss batch targets

import torch

from src.models.base.preprocessor import BasePreprocessor

NUM_CORNER_CLASSES = 4


class YoloPreprocessor(BasePreprocessor):
    """Turns each of the 4 corners into a fixed-size normalized pseudo-box for Ultralytics loss."""

    def __init__(self, box_size=0.1):
        self.box_size = box_size

    def __call__(self, corners):
        batch_idx, cls, bboxes = [], [], []
        for i, sample in enumerate(corners):
            n = sample.shape[0]
            batch_idx.append(torch.full((n,), i, dtype=torch.float32, device=corners.device))
            cls.append(torch.arange(NUM_CORNER_CLASSES, dtype=torch.float32, device=corners.device))
            wh = torch.full((n, 2), self.box_size, device=corners.device)
            bboxes.append(torch.cat([sample, wh], dim=1))
        return {
            "batch_idx": torch.cat(batch_idx),
            "cls": torch.cat(cls),
            "bboxes": torch.cat(bboxes),
        }
