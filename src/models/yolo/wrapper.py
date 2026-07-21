# src/models/yolo/wrapper.py: wraps YoloModel with native Ultralytics v8DetectionLoss semantics

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.wrapper import BaseWrapper
from src.models.yolo.model import YoloModel
from src.models.yolo.preprocessor import YoloPreprocessor
from src.models.yolo.postprocessor import YoloPostprocessor
from src.components.losses import BaseLoss
from src.components.metrics import PolygonIoU

YOLO_LOSS_NAMES = ("box", "cls", "dfl")
HEAD_BOX_SIZE = {"box": 0.3, "point": 0.1}


class YoloWrapper(BaseWrapper):
    """Wraps YoloModel with native Ultralytics v8DetectionLoss train/eval semantics."""

    def __init__(self, network=None, head="box", box_size=None, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 metrics=None, device=None, warmup_epochs=1):
        if head not in HEAD_BOX_SIZE:
            raise ValueError("Unknown det head: %s. Supported: %s" % (head, ", ".join(HEAD_BOX_SIZE)))
        if box_size is None:
            box_size = HEAD_BOX_SIZE[head]
        net = YoloModel(network=network)
        preprocessor = preprocessor or YoloPreprocessor(box_size=box_size)
        postprocessor = postprocessor or YoloPostprocessor(image_size=image_size)
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

    def get_backbone_layers(self):
        return list(self.model.net.model[:-1])

    def set_backbone_trainable(self, trainable):
        for layer in self.get_backbone_layers():
            for p in layer.parameters():
                p.requires_grad = trainable

    def build_optimizer(self, phase):
        backbone_ids = {id(p) for layer in self.get_backbone_layers() for p in layer.parameters()}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        if phase == 1:
            return AdamW(other_params, lr=1e-4)
        backbone_params = [p for layer in self.get_backbone_layers() for p in layer.parameters()]
        return AdamW([
            {"params": backbone_params, "lr": 1e-5},
            {"params": other_params, "lr": 1e-4},
        ])

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def build_batch(self, images, targets):
        batch = self.preprocessor(targets)
        batch["img"] = images
        return batch

    def update_yolo_losses(self, loss_detach, count):
        for name, value in zip(YOLO_LOSS_NAMES, loss_detach):
            self.losses.setdefault(name, BaseLoss()).update(value.item(), count)

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        batch = self.build_batch(images, targets)

        self.optimizer.zero_grad()
        raw_output = self.model.net(images)
        loss, loss_detach = self.model.net.loss(batch, preds=raw_output)
        loss.sum().backward()
        self.optimizer.step()

        self.update_yolo_losses(loss_detach, len(images))
        return self.get_loss_results()

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        batch = self.build_batch(images, targets)

        decoded, raw_dict = self.model.net(images)
        _, loss_detach = self.model.net.loss(batch, preds=raw_dict)
        self.update_yolo_losses(loss_detach, len(images))

        preds = self.postprocessor(decoded).to(self.device)
        self.update_metrics(preds.cpu().numpy(), targets.cpu().numpy())
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        decoded, _ = self.model.net(images.to(self.device, non_blocking=True))
        preds = self.postprocessor(decoded)
        return preds.cpu().numpy()
