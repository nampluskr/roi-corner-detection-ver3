# src/methods/reg/wrapper.py: composes RegModel with RegPreprocessor/RegPostprocessor and WingLoss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.methods.base.wrapper import BaseWrapper
from src.methods.reg.model import RegModel
from src.methods.reg.preprocessor import RegPreprocessor
from src.methods.reg.postprocessor import RegPostprocessor
from src.components.losses import WingLoss
from src.components.metrics import PolygonIoU


class RegWrapper(BaseWrapper):
    """Wraps RegModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, dropout=0.2, backbone="custom", head="gap",
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None, warmup_epochs=1):
        model = RegModel(in_channels=in_channels, backbone=backbone, dropout=dropout, head=head)
        preprocessor = preprocessor or RegPreprocessor()
        postprocessor = postprocessor or RegPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device,
                         warmup_epochs=warmup_epochs)
        self.applied_warmup_epochs = warmup_epochs
        if self.optimizer is None:
            phase = 1 if warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
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
