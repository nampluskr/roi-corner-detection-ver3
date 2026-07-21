# src/models/hybrid/model.py: MobileNetV3 U-Net seg model producing a quad mask for hybrid postprocessing

from src.models.base.model import BaseModel
from src.components.backbones import CustomBackbone
from src.components.backbones import (
    EFFICIENTNET_BACKBONES,
    MOBILENET_BACKBONES,
    RESNET_BACKBONES,
    SWIN_BACKBONES,
    VGG_BACKBONES,
    TorchBackbone,
)
from src.components.backbones import TIMM_CNN_BACKBONES, TimmBackbone
from src.components.adapters import CNNBackboneAdapter
from src.components.features import FeatureExtractor, FeatureSpec
from src.components.decoders import UNetDecoder
from src.components.heads import MaskHead

TORCH_HYBRID_BACKBONES = (RESNET_BACKBONES + EFFICIENTNET_BACKBONES + MOBILENET_BACKBONES
                          + SWIN_BACKBONES + VGG_BACKBONES)
SUPPORTED_HYBRID_BACKBONES = ("custom",) + TORCH_HYBRID_BACKBONES + TIMM_CNN_BACKBONES


class HybridModel(BaseModel):
    """Lightweight MobileNetV3 encoder plus a U-Net additive-skip decoder feeding a binary mask head."""

    def __init__(self, in_channels=3, network="mobilenet_v3_large", upsample="interpolate_conv"):
        super().__init__()
        network = network or "mobilenet_v3_large"
        if network == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
        elif network in TORCH_HYBRID_BACKBONES:
            encoder = TorchBackbone(network)
        elif network in TIMM_CNN_BACKBONES:
            encoder = TimmBackbone(network)
        else:
            raise ValueError("Unknown hybrid network: %s. Supported: %s"
                             % (network, ", ".join(SUPPORTED_HYBRID_BACKBONES)))

        adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=True)
        spec = FeatureSpec(network, "cnn",
                           stage_channels=encoder.stage_channels, stage_strides=encoder.stage_strides)
        spec.require("stages")

        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.decoder = UNetDecoder(spec.stage_channels, upsample=upsample)
        self.head = MaskHead(self.decoder.out_channels)
        self.mask_stride = spec.stage_strides[0]

    def forward(self, images):
        bundle = self.extractor(images)
        decoded = self.decoder(bundle.stages)
        return self.head(decoded)
