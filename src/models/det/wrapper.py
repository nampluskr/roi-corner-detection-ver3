# src/models/det/wrapper.py: composes DetModel with DetPreprocessor/DetPostprocessor and Focal+SmoothL1 loss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.wrapper import BaseWrapper
from src.models.det.model import DetModel
from src.models.det.preprocessor import DetPreprocessor
from src.models.det.postprocessor import DetPostprocessor
from src.components.losses import FocalLoss, SmoothL1Loss
from src.components.metrics import PolygonIoU


class DetWrapper(BaseWrapper):
    """Wraps DetModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, network="custom", head="box", neck_channels=256,
                 grid_stride=16, box_size=0.1, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None, warmup_epochs=1):
        model = DetModel(in_channels=in_channels, network=network, neck_channels=neck_channels,
                         grid_stride=grid_stride, head=head)
        preprocessor = preprocessor or DetPreprocessor(
            grid_stride=model.grid_stride, image_size=image_size,
            head=head, box_size=box_size)
        postprocessor = postprocessor or DetPostprocessor(
            grid_stride=model.grid_stride, image_size=image_size)
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device,
                         warmup_epochs=warmup_epochs)
        self.applied_warmup_epochs = warmup_epochs
        if self.optimizer is None:
            phase = 1 if warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_losses(self.losses or {"cls": FocalLoss(), "box": SmoothL1Loss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def build_optimizer(self, phase):
        backbone_ids = {id(p) for p in self.model.extractor.parameters()}
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        if self.applied_warmup_epochs > 0 and phase == 1:
            return AdamW(head_params, lr=1e-4)
        return AdamW([
            {"params": self.model.extractor.parameters(), "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ])

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
