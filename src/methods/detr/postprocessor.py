# src/methods/detr/postprocessor.py: convert DETR outputs into standard corners

import torch

from src.methods.base.postprocessor import BasePostprocessor

NUM_CORNER_CLASSES = 4


class DetrPostprocessor(BasePostprocessor):
    """Selects the highest-scoring query per corner class and decodes its box center."""

    def __call__(self, raw_output):
        logits = raw_output.logits
        boxes = raw_output.pred_boxes
        scores = torch.softmax(logits, dim=-1)[:, :, :NUM_CORNER_CLASSES]
        n = logits.shape[0]
        corners = torch.zeros((n, NUM_CORNER_CLASSES, 2), device=boxes.device)
        idx = torch.arange(n, device=boxes.device)
        for c in range(NUM_CORNER_CLASSES):
            best = scores[:, :, c].argmax(dim=1)
            corners[:, c] = boxes[idx, best, 0:2].clamp(0.0, 1.0)
        return corners
