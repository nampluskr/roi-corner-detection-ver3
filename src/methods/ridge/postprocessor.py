# src/methods/ridge/postprocessor.py: recover standard corners from four edge ridge-line maps

import torch

from src.methods.base.postprocessor import BasePostprocessor


class RidgePostprocessor(BasePostprocessor):
    """Fits a line to each of the four edge ridge maps and intersects adjacent lines to recover corners.

    Channel i is treated as a weighted point cloud (weight = sigmoid probability):
    its weighted centroid and dominant principal direction define the infinite line
    through corners i and (i + 1) % 4. Corner i is then the intersection of edge
    line (i - 1) % 4 and edge line i.
    """

    def __call__(self, raw_output):
        n, c, height, width = raw_output.shape
        probs = torch.sigmoid(raw_output)

        ys = torch.arange(height, device=raw_output.device, dtype=raw_output.dtype).view(1, 1, height, 1)
        xs = torch.arange(width, device=raw_output.device, dtype=raw_output.dtype).view(1, 1, 1, width)
        ys = ys.expand(n, c, height, width)
        xs = xs.expand(n, c, height, width)

        weights = probs.reshape(n, c, -1)
        weight_sum = weights.sum(dim=2).clamp(min=1e-6)
        mean_x = (weights * xs.reshape(n, c, -1)).sum(dim=2) / weight_sum
        mean_y = (weights * ys.reshape(n, c, -1)).sum(dim=2) / weight_sum

        dx = xs.reshape(n, c, -1) - mean_x.unsqueeze(2)
        dy = ys.reshape(n, c, -1) - mean_y.unsqueeze(2)
        cov_xx = (weights * dx * dx).sum(dim=2) / weight_sum
        cov_yy = (weights * dy * dy).sum(dim=2) / weight_sum
        cov_xy = (weights * dx * dy).sum(dim=2) / weight_sum

        cov = torch.stack([
            torch.stack([cov_xx, cov_xy], dim=-1),
            torch.stack([cov_xy, cov_yy], dim=-1),
        ], dim=-2)
        _, eigvecs = torch.linalg.eigh(cov)
        direction = eigvecs[..., -1]

        points = torch.stack([mean_x, mean_y], dim=-1)
        corners = self._intersect_adjacent_lines(points, direction)
        corners[..., 0] = corners[..., 0] / max(width - 1, 1)
        corners[..., 1] = corners[..., 1] / max(height - 1, 1)
        return corners

    def _intersect_adjacent_lines(self, points, direction):
        p1 = points
        d1 = direction
        p2 = points.roll(1, dims=1)
        d2 = direction.roll(1, dims=1)

        denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        denom = denom.where(denom.abs() > 1e-6, torch.full_like(denom, 1e-6))
        diff = p2 - p1
        t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / denom
        return p1 + t.unsqueeze(-1) * d1
