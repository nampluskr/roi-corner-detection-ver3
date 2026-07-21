# src/methods/heatmap/wrapper.py: composes HeatmapModel/HeatmapPreprocessor/HeatmapPostprocessor and MSE loss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.methods.base.wrapper import BaseWrapper
from src.methods.heatmap.model import HeatmapModel
from src.methods.heatmap.preprocessor import HeatmapPreprocessor
from src.methods.heatmap.postprocessor import HeatmapPostprocessor
from src.components.losses import HeatmapFocalLoss
from src.components.metrics import PolygonIoU


class HeatmapWrapper(BaseWrapper):
    """Wraps heatmap models behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, backbone="custom", head="heatmap", image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None, warmup_epochs=1):
        if head not in (None, "heatmap"):
            raise ValueError("Unknown heatmap head: %s. Supported: heatmap" % head)
        model = HeatmapModel(in_channels=in_channels, backbone=backbone)
        preprocessor = preprocessor or HeatmapPreprocessor(image_size // model.heatmap_stride)
        postprocessor = postprocessor or HeatmapPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device,
                         warmup_epochs=warmup_epochs)
        self.applied_warmup_epochs = warmup_epochs
        if self.optimizer is None:
            phase = 1 if warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_losses(self.losses or {"focal": HeatmapFocalLoss()})
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
