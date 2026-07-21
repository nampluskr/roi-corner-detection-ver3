# src/components/losses.py: BaseLoss and the concrete training losses (BCE/Dice/BCEDice/DeepSupervisedSmoothL1/Focal/HeatmapMSE/SmoothL1/Wing)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- base class for reusable training losses ---

class BaseLoss:
    """Base class computing a batch-mean loss tensor and accumulating its running mean."""

    def __init__(self, weight=1.0):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count):
        self.total += value * count
        self.count += count

    def compute(self):
        return self.total / self.count if self.count > 0 else 0.0

    def __call__(self, raw_output, target):
        loss = self.forward(raw_output, target)
        self.update(loss.item(), len(target))
        return loss

    def forward(self, raw_output, target):
        raise NotImplementedError


# --- binary cross-entropy loss for mask logits ---

class BCELoss(BaseLoss):
    """Binary cross-entropy on raw mask logits against a binary mask target."""

    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, raw_output, target):
        return self.criterion(raw_output, target)


# --- soft Dice loss for mask logits ---

class DiceLoss(BaseLoss):
    """Soft Dice loss between sigmoid mask probabilities and a binary mask target."""

    def __init__(self, smooth=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.smooth = smooth

    def forward(self, raw_output, target):
        probs = torch.sigmoid(raw_output).reshape(raw_output.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


# --- combined BCE + soft Dice loss for mask logits ---

class BCEDiceLoss(BaseLoss):
    """Weighted sum of binary cross-entropy and soft Dice loss on mask logits against a binary mask target."""

    def __init__(self, dice_weight=1.0, smooth=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, raw_output, target):
        bce = self.bce(raw_output, target)
        probs = torch.sigmoid(raw_output).reshape(raw_output.shape[0], -1)
        flat_target = target.reshape(target.shape[0], -1)
        intersection = (probs * flat_target).sum(dim=1)
        union = probs.sum(dim=1) + flat_target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return bce + self.dice_weight * (1.0 - dice).mean()


# --- deep-supervised smooth L1 loss for iterative corner refinement ---

class DeepSupervisedSmoothL1Loss(BaseLoss):
    """Weighted sum of per-step smooth L1 losses over (N, T+1, 4, 2) refinement corners against (N, 4, 2) targets."""

    def __init__(self, beta=1.0, late_emphasis=False, weight=1.0):
        super().__init__(weight=weight)
        self.beta = beta
        self.late_emphasis = late_emphasis

    def forward(self, raw_output, target):
        num_steps = raw_output.shape[1]
        target = target.unsqueeze(1).expand_as(raw_output)
        diff = (raw_output - target).abs()
        per_element = torch.where(diff < self.beta, 0.5 * diff.pow(2) / self.beta, diff - 0.5 * self.beta)
        per_step = per_element.mean(dim=(0, 2, 3))
        if self.late_emphasis:
            step_weight = torch.arange(1, num_steps + 1, device=raw_output.device, dtype=raw_output.dtype)
        else:
            step_weight = torch.ones(num_steps, device=raw_output.device, dtype=raw_output.dtype)
        return (per_step * step_weight).sum() / step_weight.sum()


# --- sigmoid focal loss for the sparse per-cell corner classification map ---

class FocalLoss(BaseLoss):
    """RetinaNet-style sigmoid focal loss between a per-class classification map and a binary target."""

    def __init__(self, alpha=0.25, gamma=2.0, weight=1.0):
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, raw_output, target):
        logits = raw_output["cls"]
        cls_target = target["cls"]
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, cls_target, reduction="none")
        p_t = prob * cls_target + (1.0 - prob) * (1.0 - cls_target)
        alpha_t = self.alpha * cls_target + (1.0 - self.alpha) * (1.0 - cls_target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        return loss.mean()


# --- mean squared error loss for sigmoid corner heatmaps ---

class HeatmapMSELoss(BaseLoss):
    """Mean squared error between sigmoid heatmap logits and Gaussian heatmap targets."""

    def __init__(self, weight=1.0):
        super().__init__(weight=weight)
        self.criterion = nn.MSELoss()

    def forward(self, raw_output, target):
        return self.criterion(torch.sigmoid(raw_output), target)


# --- CenterNet-style penalty-reduced pixelwise focal loss for sparse Gaussian heatmaps ---

class HeatmapFocalLoss(BaseLoss):
    """Penalty-reduced focal loss down-weighting easy background pixels around a sparse Gaussian peak."""

    def __init__(self, alpha=2.0, beta=4.0, weight=1.0):
        super().__init__(weight=weight)
        self.alpha = alpha
        self.beta = beta

    def forward(self, raw_output, target):
        prob = torch.sigmoid(raw_output).clamp(min=1e-6, max=1.0 - 1e-6)
        pos_mask = target.eq(1.0).float()
        neg_mask = 1.0 - pos_mask
        neg_weight = (1.0 - target).pow(self.beta)

        pos_loss = (1.0 - prob).pow(self.alpha) * torch.log(prob) * pos_mask
        neg_loss = prob.pow(self.alpha) * torch.log(1.0 - prob) * neg_weight * neg_mask

        num_pos = pos_mask.sum()
        loss = -(pos_loss.sum() + neg_loss.sum())
        return loss / num_pos.clamp(min=1.0)


# --- masked smooth L1 loss for the per-cell box/point regression map ---

class SmoothL1Loss(BaseLoss):
    """Smooth L1 loss on box/point regression, masked to positive cells and sigmoid-bounded offset channels."""

    def __init__(self, beta=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.beta = beta

    def forward(self, raw_output, target):
        pred = raw_output["box"].clone()
        pred[:, 0:2] = torch.sigmoid(pred[:, 0:2])
        box_target = target["box"]
        pos_mask = target["pos_mask"]

        diff = (pred - box_target).abs() * pos_mask
        loss = torch.where(diff < self.beta, 0.5 * diff.pow(2) / self.beta, diff - 0.5 * self.beta)
        denom = pos_mask.sum().clamp(min=1.0) * pred.shape[1]
        return loss.sum() / denom


# --- Wing loss for coordinate regression ---

class WingLoss(BaseLoss):
    """Wing loss: log penalty for small errors, linear penalty for large errors."""

    def __init__(self, apply_sigmoid=False, w=10.0, epsilon=2.0, weight=1.0):
        super().__init__(weight=weight)
        self.apply_sigmoid = apply_sigmoid
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * math.log(1.0 + w / epsilon)

    def forward(self, raw_output, target):
        pred = torch.sigmoid(raw_output) if self.apply_sigmoid else raw_output
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.c,
        )
        return loss.mean()
