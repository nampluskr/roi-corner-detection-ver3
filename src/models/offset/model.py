# src/models/offset/model.py: unified backbone canonical-offset regression model

from src.models.base.model import BaseModel
from src.components.backbones import CustomBackbone
from src.components.backbones import SUPPORTED_BACKBONES, VIT_BACKBONES, TorchBackbone
from src.components.backbones import SUPPORTED_TIMM_BACKBONES, TIMM_VIT_BACKBONES, TimmBackbone
from src.components.adapters import CNNBackboneAdapter, TransformerBackboneAdapter
from src.components.features import FeatureExtractor, FeatureSpec
from src.components.heads import GapHead, SpatialHead

MARGIN = 0.25
ALPHA = 0.25
CANONICAL_CORNERS = [
    [MARGIN, MARGIN],
    [1.0 - MARGIN, MARGIN],
    [1.0 - MARGIN, 1.0 - MARGIN],
    [MARGIN, 1.0 - MARGIN],
]


def _build_extractor_and_head(encoder, backbone_name, is_vit, head, dropout):
    """Build a FeatureExtractor and a matching coordinate head for the given head type."""
    adapter_name = "vit" if is_vit else "cnn"
    if head == "gap":
        if is_vit:
            adapter = TransformerBackboneAdapter(keep_spatial=False, keep_global=True)
        else:
            adapter = CNNBackboneAdapter(keep_spatial=False, keep_stages=False)
        spec = FeatureSpec(backbone_name, adapter_name, global_channels=encoder.out_channels)
        coordinate_head = GapHead(spec.global_channels, dropout=dropout)
    elif head == "spatial":
        if is_vit:
            adapter = TransformerBackboneAdapter(keep_spatial=True, keep_global=False)
        else:
            adapter = CNNBackboneAdapter(keep_spatial=True, keep_stages=False)
        spec = FeatureSpec(backbone_name, adapter_name,
                           global_channels=encoder.out_channels,
                           spatial_channels=encoder.out_channels)
        coordinate_head = SpatialHead(spec.spatial_channels, dropout=dropout)
    else:
        raise ValueError("Unknown offset head: %s. Supported: gap, spatial" % head)
    return FeatureExtractor(encoder, adapter, spec), coordinate_head


class OffsetModel(BaseModel):
    """Custom, timm, or torchvision backbone plus a matching adapter feeding a canonical-offset head."""

    def __init__(self, in_channels=3, network="custom", dropout=0.2, head="spatial"):
        super().__init__()
        network = network or "custom"
        head = head or "spatial"
        if network == "custom":
            encoder, is_vit = CustomBackbone(in_channels=in_channels), False
        elif network in SUPPORTED_BACKBONES:
            encoder, is_vit = TorchBackbone(network), network in VIT_BACKBONES
        elif network in SUPPORTED_TIMM_BACKBONES:
            encoder, is_vit = TimmBackbone(network), network in TIMM_VIT_BACKBONES
        else:
            supported = ("custom",) + SUPPORTED_BACKBONES + SUPPORTED_TIMM_BACKBONES
            raise ValueError("Unknown offset network: %s. Supported: %s"
                             % (network, ", ".join(supported)))
        self.head_name = head
        self.extractor, self.head = _build_extractor_and_head(
            encoder, network, is_vit, head, dropout)

    def forward(self, images):
        bundle = self.extractor(images)
        if self.head_name == "gap":
            return self.head(bundle.global_feature)
        return self.head(bundle.spatial_feature)
