# src/methods/det/model.py: composable custom detection model over stage-returning backbones

from src.methods.base.model import BaseModel
from src.components.backbones import CustomBackbone
from src.components.backbones import EFFICIENTNET_BACKBONES, RESNET_BACKBONES
from src.components.backbones import SWIN_BACKBONES, VGG_BACKBONES, VIT_BACKBONES, TorchBackbone
from src.components.backbones import TIMM_CNN_BACKBONES, TIMM_VIT_BACKBONES, TimmBackbone
from src.components.adapters import CNNBackboneAdapter
from src.components.features import FeatureExtractor, FeatureSpec
from src.components.necks import MultiScaleNeck
from src.components.heads import NUM_CORNER_CLASSES, DetectionHead

TORCH_DET_BACKBONES = RESNET_BACKBONES + EFFICIENTNET_BACKBONES + SWIN_BACKBONES + VGG_BACKBONES
SUPPORTED_DET_BACKBONES = ("custom",) + TORCH_DET_BACKBONES + TIMM_CNN_BACKBONES


class DetModel(BaseModel):
    """Stage-returning backbone plus a multi-scale neck feeding a per-cell detection head."""

    def __init__(self, in_channels=3, backbone="custom", neck_channels=256, grid_stride=16,
                 head="box", upsample="interpolate_conv"):
        super().__init__()
        backbone = backbone or "custom"
        if backbone == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
        elif backbone in VIT_BACKBONES or backbone in TIMM_VIT_BACKBONES:
            raise ValueError("det backbone %s has no stages capability (ViT/DINOv2 family). Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_DET_BACKBONES)))
        elif backbone in TORCH_DET_BACKBONES:
            encoder = TorchBackbone(backbone)
        elif backbone in TIMM_CNN_BACKBONES:
            encoder = TimmBackbone(backbone)
        else:
            raise ValueError("Unknown det backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_DET_BACKBONES)))

        adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=True)
        spec = FeatureSpec(backbone, "cnn",
                           stage_channels=encoder.stage_channels, stage_strides=encoder.stage_strides)
        spec.require("stages")

        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.neck = MultiScaleNeck(spec.stage_channels, spec.stage_strides,
                                   grid_stride=grid_stride, out_channels=neck_channels, upsample=upsample)
        self.head = DetectionHead(self.neck.out_channels, head=head)
        self.grid_stride = grid_stride

    def forward(self, images):
        bundle = self.extractor(images)
        feature = self.neck(bundle.stages)
        return self.head(feature)
