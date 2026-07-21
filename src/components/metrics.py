# src/components/metrics.py: BaseMetric and the concrete sample-level metrics (corner distance / polygon IoU / success rate)

import numpy as np

from src.utils.geometry import polygon_area


# --- base class for stateful sample-level metrics ---

class BaseMetric:
    """Base class for a stateful (reset/update/compute) sample-level metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            if np.isnan(pred).any():
                continue
            value = self(pred, target)
            if isinstance(value, float) and np.isnan(value):
                continue
            self.total += value
            self.count += 1

    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

    def __call__(self, preds, targets):
        raise NotImplementedError


# --- normalized-coordinate corner distance metrics ---

class CornerDistanceMetric(BaseMetric):
    """Base metric for Euclidean distances between corresponding corners."""

    def distances(self, preds, targets):
        pred = np.asarray(preds, dtype=np.float64).reshape(4, 2)
        target = np.asarray(targets, dtype=np.float64).reshape(4, 2)
        return np.linalg.norm(pred - target, axis=1)


class MeanCornerDistance(CornerDistanceMetric):
    """Computes dataset mean of sample-wise mean corner distances."""

    def __call__(self, preds, targets):
        return float(self.distances(preds, targets).mean())


class MaxCornerDistance(CornerDistanceMetric):
    """Computes dataset mean of sample-wise maximum corner distances."""

    def __call__(self, preds, targets):
        return float(self.distances(preds, targets).max())


class PCK(CornerDistanceMetric):
    """Computes the fraction of corners within one normalized distance threshold."""

    def __init__(self, threshold):
        self.threshold = threshold
        super().__init__()

    def __call__(self, preds, targets):
        return float((self.distances(preds, targets) <= self.threshold).mean())


# --- polygon IoU between predicted and ground-truth quadrilaterals ---

class PolygonIoU(BaseMetric):
    """Computes area-based IoU between predicted and ground-truth quadrilaterals."""

    def __call__(self, preds, targets):
        pred = np.array(preds, dtype=np.float64).reshape(4, 2)
        target = np.array(targets, dtype=np.float64).reshape(4, 2)

        inter_pts = self._clip_polygon(pred, target)
        inter_area = polygon_area(inter_pts) if len(inter_pts) >= 3 else 0.0
        union_area = polygon_area(pred) + polygon_area(target) - inter_area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)

    def _clip_polygon(self, subject, clip):
        # Sutherland-Hodgman clipping: clip subject against each edge of clip
        output = subject.tolist()
        for i in range(len(clip)):
            if not output:
                break
            a, b = clip[i], clip[(i + 1) % len(clip)]
            input_pts = output
            output = []
            for j in range(len(input_pts)):
                cur, prev = input_pts[j], input_pts[j - 1]
                cur_inside = self._is_inside(cur, a, b)
                prev_inside = self._is_inside(prev, a, b)
                if cur_inside:
                    if not prev_inside:
                        output.append(self._intersect(prev, cur, a, b))
                    output.append(cur)
                elif prev_inside:
                    output.append(self._intersect(prev, cur, a, b))
        return np.array(output, dtype=np.float64)

    def _is_inside(self, p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def _intersect(self, p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]


# --- validity rate for corner predictions ---

class SuccessRate(BaseMetric):
    """Computes the fraction of samples with finite predicted corner values."""

    def update(self, preds, targets):
        for pred in preds:
            self.total += float(np.isfinite(pred).all())
            self.count += 1

    def __call__(self, preds, targets):
        return float(np.isfinite(preds).all())
