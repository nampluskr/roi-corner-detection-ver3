# src/models/torchdet/wrapper.py: wraps TorchDetModel with native torchvision detection semantics

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.wrapper import BaseWrapper
from src.models.torchdet.model import TorchDetModel
from src.models.torchdet.preprocessor import TorchDetPreprocessor
from src.models.torchdet.postprocessor import TorchDetPostprocessor
from src.components.losses import BaseLoss
from src.components.metrics import PolygonIoU

HEAD_BOX_SIZE = {"box": 0.3, "point": 0.1}


class TorchDetWrapper(BaseWrapper):
    """Wraps TorchDetModel with native torchvision detection train/eval semantics."""

    def __init__(self, network=None, head="box", box_size=None, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 metrics=None, device=None, warmup_epochs=1):
        if head not in HEAD_BOX_SIZE:
            raise ValueError("Unknown det head: %s. Supported: %s" % (head, ", ".join(HEAD_BOX_SIZE)))
        if box_size is None:
            box_size = HEAD_BOX_SIZE[head]
        net = TorchDetModel(network=network)
        preprocessor = preprocessor or TorchDetPreprocessor(
            image_size=image_size, box_size=box_size, label_offset=net.label_offset)
        postprocessor = postprocessor or TorchDetPostprocessor(
            image_size=image_size, label_offset=net.label_offset)
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=None, metrics=metrics, device=device,
                         warmup_epochs=warmup_epochs)
        self.applied_warmup_epochs = warmup_epochs
        if self.optimizer is None:
            phase = 1 if warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def get_backbone_module(self):
        return self.model.net.backbone

    def build_optimizer(self, phase):
        backbone_ids = {id(p) for p in self.get_backbone_module().parameters()}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        if phase == 1:
            return AdamW(other_params, lr=1e-4)
        return AdamW([
            {"params": self.get_backbone_module().parameters(), "lr": 1e-5},
            {"params": other_params, "lr": 1e-4},
        ])

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        native_targets = self.preprocessor(targets)

        self.optimizer.zero_grad()
        loss_dict = self.model(list(images), native_targets)
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()

        for name, value in loss_dict.items():
            self.losses.setdefault(name, BaseLoss()).update(value.item(), len(images))
        return self.get_loss_results()

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        raw_output = self.model(list(images))
        self.compute_metrics(raw_output, targets)
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        raw_output = self.model(list(images.to(self.device, non_blocking=True)))
        preds = self.postprocessor(raw_output)
        return preds.cpu().numpy()
