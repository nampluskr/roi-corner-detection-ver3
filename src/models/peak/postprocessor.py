# src/models/peak/postprocessor.py: convert raw peak logits into standard corners

import torch

from src.models.base.postprocessor import BasePostprocessor


class PeakPostprocessor(BasePostprocessor):
    """Applies sigmoid then hard-argmax to four corner peak logits and returns normalized corners."""

    def __call__(self, raw_output):
        n, c, height, width = raw_output.shape
        probs = torch.sigmoid(raw_output).reshape(n, c, height * width)
        indices = probs.argmax(dim=2)
        y = (indices // width).to(raw_output.dtype) / max(height - 1, 1)
        x = (indices % width).to(raw_output.dtype) / max(width - 1, 1)
        return torch.stack([x, y], dim=2)
