# src/models/yolo/postprocessor.py: convert Ultralytics decoded outputs into standard corners

import torch
from ultralytics.utils.nms import non_max_suppression

from src.models.base.postprocessor import BasePostprocessor

NUM_CORNER_CLASSES = 4


class YoloPostprocessor(BasePostprocessor):
    """Runs Ultralytics NMS on a decoded eval-mode tensor and decodes it to (N,4,2) corners."""

    def __init__(self, image_size=224, conf_thres=0.001, iou_thres=0.5, max_det=10):
        self.image_size = image_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def __call__(self, decoded):
        results = non_max_suppression(decoded, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                      max_det=self.max_det, nc=NUM_CORNER_CLASSES)
        corners = torch.full((len(results), NUM_CORNER_CLASSES, 2), 0.5)
        for i, pred in enumerate(results):
            boxes, scores, labels = pred[:, :4], pred[:, 4], pred[:, 5]
            for c in range(NUM_CORNER_CLASSES):
                mask = labels == c
                if not mask.any():
                    continue
                box = boxes[mask][scores[mask].argmax()]
                corners[i, c, 0] = (box[0] + box[2]) / 2 / self.image_size
                corners[i, c, 1] = (box[1] + box[3]) / 2 / self.image_size
        return corners
