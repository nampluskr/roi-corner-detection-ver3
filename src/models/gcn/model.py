# src/models/gcn/model.py: CNN backbone with spatial-preserving init head and iterative GCN corner refinement

import torch
import torch.nn as nn
import torch.nn.functional as F

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

TORCH_GCN_BACKBONES = RESNET_BACKBONES + EFFICIENTNET_BACKBONES + SWIN_BACKBONES + VGG_BACKBONES
SUPPORTED_GCN_BACKBONES = ("custom",) + TORCH_GCN_BACKBONES + TIMM_CNN_BACKBONES

NUM_CORNERS = 4


def build_normalized_adjacency():
    """Return the symmetrically normalized adjacency of the 4-cycle corner graph with self-loops."""
    adjacency = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    adjacency = adjacency + torch.eye(NUM_CORNERS)
    degree = adjacency.sum(dim=1)
    d_inv_sqrt = torch.diag(degree.pow(-0.5))
    return d_inv_sqrt @ adjacency @ d_inv_sqrt


class InitHead(nn.Module):
    """Strided convolutions and pooling projecting a spatial feature to NUM_CORNERS * 2 raw values."""

    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, NUM_CORNERS * 2),
        )

    def forward(self, spatial_feature):
        return self.layers(spatial_feature)


class GCNRefiner(nn.Module):
    """Fixed-adjacency graph convolution stack predicting a bounded per-vertex coordinate offset."""

    def __init__(self, in_dim, hidden_dim, num_layers, offset_radius):
        super().__init__()
        self.offset_radius = offset_radius
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.offset_head = nn.Linear(hidden_dim, 2)

    def forward(self, vertex_features, adjacency):
        hidden = vertex_features
        for layer in self.layers:
            hidden = F.relu(torch.matmul(adjacency, layer(hidden)))
        return self.offset_radius * torch.tanh(self.offset_head(hidden))


class GCNModel(BaseModel):
    """CNN backbone plus a spatial-preserving init head and fixed-graph GCN refiners that iteratively move corners."""

    def __init__(self, in_channels=3, network="custom", iterations=3, num_layers=2,
                 shared_weights=True, offset_radius=0.1, hidden_dim=256):
        super().__init__()
        network = network or "custom"
        if network == "custom":
            encoder = CustomBackbone(in_channels=in_channels)
        elif network in TORCH_GCN_BACKBONES:
            encoder = TorchBackbone(network)
        elif network in TIMM_CNN_BACKBONES:
            encoder = TimmBackbone(network)
        else:
            raise ValueError("Unknown gcn network: %s. Supported: %s"
                             % (network, ", ".join(SUPPORTED_GCN_BACKBONES)))

        adapter = CNNBackboneAdapter(keep_spatial=True, keep_stages=False)
        spec = FeatureSpec(network, "cnn",
                           global_channels=encoder.out_channels,
                           spatial_channels=encoder.out_channels)
        spec.require("spatial")

        self.iterations = iterations
        self.shared_weights = shared_weights
        self.extractor = FeatureExtractor(encoder, adapter, spec)
        self.init_head = InitHead(spec.spatial_channels)

        in_dim = spec.spatial_channels + 2
        num_refiners = 1 if shared_weights else iterations
        self.refiners = nn.ModuleList([
            GCNRefiner(in_dim, hidden_dim, num_layers, offset_radius) for _ in range(num_refiners)
        ])
        self.register_buffer("adjacency", build_normalized_adjacency())

    def sample_vertex_features(self, features, corners):
        grid = (corners * 2.0 - 1.0).unsqueeze(1)
        sampled = F.grid_sample(features, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        sampled = sampled.squeeze(2).permute(0, 2, 1)
        return torch.cat([sampled, corners], dim=2)

    def forward(self, images):
        bundle = self.extractor(images)
        features = bundle.spatial_feature
        corners = torch.sigmoid(self.init_head(features)).reshape(-1, NUM_CORNERS, 2)

        outputs = [corners]
        for t in range(self.iterations):
            refiner = self.refiners[0] if self.shared_weights else self.refiners[t]
            vertex_features = self.sample_vertex_features(features, corners)
            corners = corners + refiner(vertex_features, self.adjacency)
            outputs.append(corners)
        return torch.stack(outputs, dim=1)
