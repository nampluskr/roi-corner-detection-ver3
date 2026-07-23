# src/models/ridge/preprocessor.py: rasterize corner-pair edge lines into dense ridge targets

import torch

from src.models.base.preprocessor import BasePreprocessor


class RidgePreprocessor(BasePreprocessor):
    """Converts (N, 4, 2) normalized corners into (N, 4, H, W) ridge targets.

    Channel i is a Gaussian ridge along the infinite line through corners i and
    (i + 1) % 4, extended across the full image rather than stopping at the two
    corners, so its crest forms a straight line cutting the map in two. Pixels
    within half a cell of the line are snapped to exactly 1.0 so HeatmapFocalLoss,
    which treats target == 1.0 pixels as positives, has a dense positive line per
    channel instead of zero positives when no pixel center lies exactly on the line.
    """

    def __init__(self, ridge_size, sigma=None):
        self.ridge_size = ridge_size
        # scale ridge thickness with map resolution so the relative crest width is
        # constant across strides; ridge_size=56 keeps the original sigma=2.0
        self.sigma = sigma if sigma is not None else ridge_size / 28.0

    def __call__(self, corners):
        device = corners.device
        height = self.ridge_size
        width = self.ridge_size
        n = corners.shape[0]

        ys = torch.arange(height, device=device, dtype=corners.dtype).view(1, height, 1)
        xs = torch.arange(width, device=device, dtype=corners.dtype).view(1, 1, width)

        p1 = corners * (width - 1)
        p2 = corners.roll(-1, dims=1) * (width - 1)
        direction = p2 - p1
        length = direction.norm(dim=2).clamp(min=1e-6)
        normal = torch.stack([-direction[:, :, 1], direction[:, :, 0]], dim=2) / length.unsqueeze(2)

        px = xs.unsqueeze(1) - p1[:, :, 0].view(n, 4, 1, 1)
        py = ys.unsqueeze(1) - p1[:, :, 1].view(n, 4, 1, 1)
        distance = px * normal[:, :, 0].view(n, 4, 1, 1) + py * normal[:, :, 1].view(n, 4, 1, 1)

        target = torch.exp(-distance.pow(2) / (2.0 * self.sigma * self.sigma))
        crest = distance.abs() <= 0.5
        return target.masked_fill(crest, 1.0)
