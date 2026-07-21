# src/models/torchseg/wrapper.py: composes TorchSegModel with seg mask pre/post and BCE+Dice loss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.wrapper import BaseWrapper
from src.models.torchseg.model import TorchSegModel
from src.models.torchseg.preprocessor import TorchSegPreprocessor
from src.models.torchseg.postprocessor import TorchSegPostprocessor
from src.components.losses import BCELoss, DiceLoss
from src.components.metrics import PolygonIoU


class TorchSegWrapper(BaseWrapper):
    """Wraps torchvision whole segmentation models behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, network=None, head="mask", image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None, warmup_epochs=1):
        # head kwarg accepted for CLI compatibility with get_wrapper_kwargs; unused here
        net = TorchSegModel(network=network)
        preprocessor = preprocessor or TorchSegPreprocessor(image_size // net.mask_stride)
        postprocessor = postprocessor or TorchSegPostprocessor()
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device,
                         warmup_epochs=warmup_epochs)
        if self.optimizer is None:
            self.set_optimizer(self.build_optimizer(phase=2))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_losses(self.losses or {"bce": BCELoss(), "dice": DiceLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def build_optimizer(self, phase):
        return AdamW(self.model.parameters(), lr=1e-4)

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
