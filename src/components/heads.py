# src/components/heads.py: task prediction heads (coordinate, detection, dense, mask)

import torch.nn as nn

from src.components.blocks import ConvBlock


# --- predicts flattened corner coordinates from CNN features ---

class CoordGapHead(nn.Module):
    """Dropout followed by a linear projection from a global feature to 8 raw corner values."""

    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, 8)

    def forward(self, global_feature):
        return self.fc(self.dropout(global_feature))


class CoordSpatialHead(nn.Module):
    """Strided convolutions and pooling followed by a linear projection to 8 raw corner values."""

    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 8),
        )

    def forward(self, spatial_feature):
        return self.layers(spatial_feature)


# --- predicts per-cell corner classification and box/point regression maps ---

NUM_CORNER_CLASSES = 4
BOX_CHANNELS = {"box": 4, "point": 2}


class DetectionHead(nn.Module):
    """Splits a shared trunk into a per-class classification map and a class-agnostic box/point regression map."""

    def __init__(self, in_channels, hidden_channels=256, head="box"):
        super().__init__()
        if head not in BOX_CHANNELS:
            raise ValueError("Unknown det head: %s. Supported: %s"
                              % (head, ", ".join(BOX_CHANNELS)))
        self.head = head
        self.trunk = ConvBlock(in_channels, hidden_channels, kernel_size=3, stride=1)
        self.cls_conv = nn.Conv2d(hidden_channels, NUM_CORNER_CLASSES, kernel_size=1)
        self.box_conv = nn.Conv2d(hidden_channels, BOX_CHANNELS[head], kernel_size=1)

    def forward(self, feature):
        x = self.trunk(feature)
        return {"cls": self.cls_conv(x), "box": self.box_conv(x)}


# --- predicts four dense channel logits from decoded features (shared by peak and ridge) ---

class FourChannelDenseHead(nn.Module):
    """Projects a decoded spatial feature to four dense channel logits."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, decoded_feature):
        return self.conv(decoded_feature)


# --- predicts binary mask logits from a decoded spatial feature ---

class MaskHead(nn.Module):
    """Projects a decoded spatial feature to single-channel binary mask logits."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, decoded_feature):
        return self.conv(decoded_feature)
