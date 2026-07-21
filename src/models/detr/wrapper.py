# src/models/detr/wrapper.py: wraps DetrModel with Hugging Face native Hungarian loss semantics

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.wrapper import BaseWrapper
from src.models.detr.model import DetrModel
from src.models.detr.preprocessor import DetrPreprocessor
from src.models.detr.postprocessor import DetrPostprocessor
from src.components.losses import BaseLoss
from src.components.metrics import PolygonIoU

HEAD_BOX_SIZE = {"box": 0.3, "point": 0.1}


class DetrWrapper(BaseWrapper):
    """Wraps DetrModel with Hugging Face native Hungarian loss semantics."""

    def __init__(self, network=None, head="box", box_size=None, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 metrics=None, device=None, grad_clip=1.0, warmup_epochs=1):
        # image_size kwarg accepted for CLI compatibility; unused by HF DETR here
        if head not in HEAD_BOX_SIZE:
            raise ValueError("Unknown det head: %s. Supported: %s" % (head, ", ".join(HEAD_BOX_SIZE)))
        if box_size is None:
            box_size = HEAD_BOX_SIZE[head]
        net = DetrModel(network=network)
        preprocessor = preprocessor or DetrPreprocessor(box_size=box_size)
        postprocessor = postprocessor or DetrPostprocessor()
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=None, metrics=metrics, device=device,
                         warmup_epochs=warmup_epochs)
        self.grad_clip = grad_clip
        self.applied_warmup_epochs = warmup_epochs
        if self.optimizer is None:
            phase = 1 if warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def set_backbone_trainable(self, trainable):
        for name, param in self.model.named_parameters():
            if name.startswith("net.model.backbone"):
                param.requires_grad = trainable

    def build_optimizer(self, phase):
        """Return DETR fine-tuning parameter groups with conservative pretrained-model learning rates."""
        backbone_params, classifier_params, other_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("net.model.backbone"):
                backbone_params.append(param)
            elif name.startswith("net.class_labels_classifier"):
                classifier_params.append(param)
            else:
                other_params.append(param)
        if phase == 1:
            return AdamW([
                {"params": other_params, "lr": 1e-4},
                {"params": classifier_params, "lr": 1e-4},
            ], weight_decay=1e-4)
        return AdamW([
            {"params": backbone_params, "lr": 1e-5},
            {"params": other_params, "lr": 1e-4},
            {"params": classifier_params, "lr": 1e-4},
        ], weight_decay=1e-4)

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def update_detr_losses(self, loss_dict, count):
        for name, value in loss_dict.items():
            self.losses.setdefault(name, BaseLoss()).update(value.item(), count)

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        labels = self.preprocessor(targets)

        self.optimizer.zero_grad()
        output = self.model(images, labels=labels)
        output.loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_detr_losses(output.loss_dict, len(images))
        return self.get_loss_results()

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        labels = self.preprocessor(targets)

        output = self.model(images, labels=labels)
        self.update_detr_losses(output.loss_dict, len(images))
        preds = self.postprocessor(output)
        self.update_metrics(preds.cpu().numpy(), targets.cpu().numpy())
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        output = self.model(images.to(self.device, non_blocking=True))
        preds = self.postprocessor(output)
        return preds.cpu().numpy()
