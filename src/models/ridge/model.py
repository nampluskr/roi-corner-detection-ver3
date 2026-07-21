# src/models/ridge/model.py: stage-based composable ridge model for corner localization

from src.models.base.model import BaseModel
from src.components.backbones import CustomBackbone
from src.components.backbones import (
    EFFICIENTNET_BACKBONES,
    RESNET_BACKBONES,
    SWIN_BACKBONES,
    VGG_BACKBONES,
    TorchBackbone,
)
from src.components.backbones import TIMM_CNN_BACKBONES, TimmBackbone
from src.components.adapters import CNNBackboneAdapter
from src.components.features import FeatureExtractor, FeatureSpec
from src.components.heads import FourChannelDenseHead
from src.components.decoders import SegDecoder

TORCH_RIDGE_BACKBONES = RESNET_BACKBONES + EFFICIENTNET_BACKBONES + SWIN_BACKBONES + VGG_BACKBONES
SUPPORTED_RIDGE_BACKBONES = ("custom",) + TORCH_RIDGE_BACKBONES + TIMM_CNN_BACKBONES


class RidgeModel(BaseModel):
    """Stage-returning backbone plus a U-Net additive-skip decoder feeding a four-edge ridge-line head."""

    def __init__(self, in_channels=3, network="custom", upsample="interpolate_conv"):
        super().__init__()
        network = network or "custom"
        if network == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
        elif network in TORCH_RIDGE_BACKBONES:
            encoder = TorchBackbone(network)
        elif network in TIMM_CNN_BACKBONES:
            encoder = TimmBackbone(network)
        else:
            raise ValueError("Unknown ridge network: %s. Supported: %s"
                             % (network, ", ".join(SUPPORTED_RIDGE_BACKBONES)))

        adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=True)
        spec = FeatureSpec(network, "cnn",
                           stage_channels=encoder.stage_channels,
                           stage_strides=encoder.stage_strides)
        spec.require("stages")

        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.decoder = SegDecoder(spec.stage_channels, upsample=upsample)
        self.head = FourChannelDenseHead(self.decoder.out_channels)
        self.ridge_stride = spec.stage_strides[0]

    def forward(self, images):
        bundle = self.extractor(images)
        decoded = self.decoder(bundle.stages)
        return self.head(decoded)
