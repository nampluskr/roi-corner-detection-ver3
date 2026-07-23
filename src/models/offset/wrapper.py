# src/models/offset/wrapper.py: composes OffsetModel/Preprocessor/Postprocessor with a canonical-offset smooth L1 loss

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.wrapper import BaseWrapper
from src.models.offset.model import OffsetModel, ALPHA
from src.models.offset.preprocessor import OffsetPreprocessor
from src.models.offset.postprocessor import OffsetPostprocessor
from src.components.losses import BaseLoss
from src.components.metrics import PolygonIoU


class OffsetSmoothL1Loss(BaseLoss):
    """Smooth L1 loss on flat (N, 8) canonical-offset predictions against offset targets."""

    def __init__(self, beta=1.0, weight=1.0):
        super().__init__(weight=weight)
        self.beta = beta

    def forward(self, raw_output, target):
        diff = (raw_output - target).abs()
        loss = torch.where(diff < self.beta, 0.5 * diff.pow(2) / self.beta, diff - 0.5 * self.beta)
        return loss.mean()


class OffsetWrapper(BaseWrapper):
    """Wraps OffsetModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, dropout=0.2, network="custom", head="spatial",
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None, warmup_epochs=1):
        model = OffsetModel(in_channels=in_channels, network=network, dropout=dropout, head=head)
        preprocessor = preprocessor or OffsetPreprocessor()
        postprocessor = postprocessor or OffsetPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device,
                         warmup_epochs=warmup_epochs)
        self.applied_warmup_epochs = warmup_epochs
        if self.optimizer is None:
            phase = 1 if warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_losses(self.losses or {"loss": OffsetSmoothL1Loss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def build_optimizer(self, phase):
        if not hasattr(self.model, "extractor") or self.applied_warmup_epochs == 0:
            return AdamW(self.model.parameters(), lr=1e-4)
        backbone_ids = {id(p) for p in self.model.extractor.parameters()}
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        if phase == 1:
            return AdamW(head_params, lr=1e-4)
        return AdamW([
            {"params": self.model.extractor.parameters(), "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ])

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def compute_losses(self, raw_output, targets):
        target = self.preprocessor(targets)
        offsets = ALPHA * torch.tanh(raw_output)
        return {name: loss_fn(offsets, target) for name, loss_fn in self.losses.items()}
