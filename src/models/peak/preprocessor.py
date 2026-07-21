# src/models/peak/preprocessor.py: rasterize standard corners into Gaussian peak targets

import torch

from src.models.base.preprocessor import BasePreprocessor


class PeakPreprocessor(BasePreprocessor):
    """Converts (N, 4, 2) normalized corners into (N, 4, H, W) Gaussian peak targets."""

    def __init__(self, peak_size, sigma=2.0):
        self.peak_size = peak_size
        self.sigma = sigma

    def __call__(self, corners):
        device = corners.device
        height = self.peak_size
        width = self.peak_size
        ys = torch.arange(height, device=device, dtype=corners.dtype).view(1, 1, height, 1)
        xs = torch.arange(width, device=device, dtype=corners.dtype).view(1, 1, 1, width)
        centers_x = corners[:, :, 0].clamp(0.0, 1.0) * (width - 1)
        centers_y = corners[:, :, 1].clamp(0.0, 1.0) * (height - 1)
        dx = xs - centers_x.view(corners.shape[0], 4, 1, 1)
        dy = ys - centers_y.view(corners.shape[0], 4, 1, 1)
        maps = torch.exp(-(dx.pow(2) + dy.pow(2)) / (2.0 * self.sigma * self.sigma))
        peaks = maps.flatten(2).amax(dim=2).clamp(min=1e-6).view(corners.shape[0], 4, 1, 1)
        return maps / peaks
