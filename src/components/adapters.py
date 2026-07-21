# src/components/adapters.py: convert native backbone features into a FeatureBundle (base, CNN, transformer)

import torch.nn as nn

from src.components.features import FeatureBundle


# --- base class converting native backbone features to a FeatureBundle ---

class BaseBackboneAdapter(nn.Module):
    """Base class converting a backbone's native feature dict into a FeatureBundle."""

    def forward(self, native_features):
        raise NotImplementedError


# --- adapts CNN backbone features into a FeatureBundle ---

class CNNBackboneAdapter(BaseBackboneAdapter):
    """Pools the final CNN feature map into FeatureBundle.global_feature and passes stages through."""

    def __init__(self, keep_spatial=True, keep_stages=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.keep_spatial = keep_spatial
        self.keep_stages = keep_stages

    def forward(self, native_features):
        final = native_features["final"]
        global_feature = self.pool(final).flatten(1)
        spatial_feature = final if self.keep_spatial else None
        stages = native_features.get("stages") if self.keep_stages else None
        return FeatureBundle(global_feature=global_feature, spatial_feature=spatial_feature, stages=stages)


# --- adapts ViT token features into a FeatureBundle ---

class TransformerBackboneAdapter(BaseBackboneAdapter):
    """Reshapes ViT cls/token features into FeatureBundle.global_feature and spatial_feature."""

    def __init__(self, keep_spatial=True, keep_global=True):
        super().__init__()
        self.keep_spatial = keep_spatial
        self.keep_global = keep_global

    def forward(self, native_features):
        global_feature = native_features["cls"] if self.keep_global else None
        spatial_feature = None
        if self.keep_spatial:
            tokens = native_features["tokens"]
            grid_h, grid_w = native_features["grid_size"]
            n, l, c = tokens.shape
            spatial_feature = tokens.transpose(1, 2).reshape(n, c, grid_h, grid_w)
        return FeatureBundle(global_feature=global_feature, spatial_feature=spatial_feature, stages=None)
