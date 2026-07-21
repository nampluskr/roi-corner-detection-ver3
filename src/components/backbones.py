# src/components/backbones.py: image encoders (BaseBackbone, CustomBackbone, TimmBackbone, TorchBackbone)

import os

import timm
import torch
import torch.nn as nn
import torchvision.models as models
from safetensors.torch import load_file

from src.components.blocks import ConvBlock


# --- base class for image encoders ---

class BaseBackbone(nn.Module):
    """Base class for an image encoder producing a native final feature and stage features."""

    def forward(self, images):
        raise NotImplementedError


# --- project baseline encoder with no pretrained weights ---

DEFAULT_STAGE_CHANNELS = (64, 128, 256, 512)


class CustomBackbone(BaseBackbone):
    """Stem plus four downsampling ConvBlock stages, reaching output stride 16."""

    def __init__(self, in_channels=3, stage_channels=DEFAULT_STAGE_CHANNELS):
        super().__init__()
        self.stem = ConvBlock(in_channels, stage_channels[0], kernel_size=3, stride=2)
        self.stage1 = ConvBlock(stage_channels[0], stage_channels[0], kernel_size=3, stride=1)
        self.stage2 = ConvBlock(stage_channels[0], stage_channels[1], kernel_size=3, stride=2)
        self.stage3 = ConvBlock(stage_channels[1], stage_channels[2], kernel_size=3, stride=2)
        self.stage4 = ConvBlock(stage_channels[2], stage_channels[3], kernel_size=3, stride=2)
        self.out_channels = stage_channels[3]
        self.out_stride = 16
        self.stage_channels = stage_channels
        self.stage_strides = (2, 4, 8, 16)

    def forward(self, images):
        x = self.stem(images)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return {"final": s4, "stages": [s1, s2, s3, s4]}


# --- timm CNN/transformer backbone wrappers ---

TIMM_BACKBONE_WEIGHTS = {
    "wide_resnet50_2.tv_in1k": "/mnt/d/backbones/wide_resnet50_2.tv_in1k/model.safetensors",
    "deit_base_distilled_patch16_224.fb_in1k": "/mnt/d/backbones/deit_base_distilled_patch16_224.fb_in1k/model.safetensors",
    "cait_s24_224.fb_dist_in1k": "/mnt/d/backbones/cait_s24_224.fb_dist_in1k/model.safetensors",
}

TIMM_CNN_BACKBONES = ("wide_resnet50_2.tv_in1k",)
TIMM_VIT_PREFIX_TOKENS = {
    "deit_base_distilled_patch16_224.fb_in1k": 2,
    "cait_s24_224.fb_dist_in1k": 1,
}
TIMM_VIT_BACKBONES = tuple(TIMM_VIT_PREFIX_TOKENS.keys())
SUPPORTED_TIMM_BACKBONES = TIMM_CNN_BACKBONES + TIMM_VIT_BACKBONES


class TimmBackbone(BaseBackbone):
    """timm model wrapper returning the same native CNN/ViT feature contract as TorchBackbone."""

    def __init__(self, backbone="wide_resnet50_2.tv_in1k", pretrained=True):
        super().__init__()
        if backbone not in SUPPORTED_TIMM_BACKBONES:
            raise ValueError("Unknown timm backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_TIMM_BACKBONES)))

        net = timm.create_model(backbone, pretrained=False)
        if pretrained:
            self.load_local_weights(net, TIMM_BACKBONE_WEIGHTS[backbone])
        net.reset_classifier(0)

        self.backbone_name = backbone
        self.net = net
        self.out_channels = net.num_features
        if backbone in TIMM_CNN_BACKBONES:
            self.family = "cnn"
            feature_info = net.feature_info[1:]
            self.stage_channels = tuple(info["num_chs"] for info in feature_info)
            self.stage_strides = tuple(info["reduction"] for info in feature_info)
        else:
            self.family = "vit"
            self.patch_size = net.patch_embed.patch_size[0]
            self.prefix_tokens = TIMM_VIT_PREFIX_TOKENS[backbone]
        self.out_stride = 32

    def load_local_weights(self, net, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local timm weight not found: %s" % path)
        state_dict = load_file(path)
        net.load_state_dict(state_dict, strict=True)

    def forward(self, images):
        if self.family == "cnn":
            stages = list(self.net.forward_intermediates(images, intermediates_only=True))[1:]
            return {"final": stages[-1], "stages": stages}

        tokens = self.net.forward_features(images)
        grid_h = images.shape[2] // self.patch_size
        grid_w = images.shape[3] // self.patch_size
        return {"cls": tokens[:, 0], "tokens": tokens[:, self.prefix_tokens:], "grid_size": (grid_h, grid_w)}


# --- torchvision CNN backbone wrappers ---

BACKBONE_WEIGHTS = {
    "resnet18": "/mnt/d/backbones/resnet18-f37072fd.pth",
    "resnet34": "/mnt/d/backbones/resnet34-b627a593.pth",
    "resnet50": "/mnt/d/backbones/resnet50-0676ba61.pth",
    "efficientnet_b0": "/mnt/d/backbones/efficientnet_b0_rwightman-7f5810bc.pth",
    "mobilenet_v3_large": "/mnt/d/backbones/mobilenet_v3_large-8738ca79.pth",
    "vgg16": "/mnt/d/backbones/vgg16-397923af.pth",
    "vgg19": "/mnt/d/backbones/vgg19-dcbb9e9d.pth",
    "vit_b_16": "/mnt/d/backbones/vit_b_16-c867db91.pth",
    "swin_t": "/mnt/d/backbones/swin_t-704ceda3.pth",
}

BACKBONE_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "vit_b_16": models.vit_b_16,
    "swin_t": models.swin_t,
}

SUPPORTED_BACKBONES = tuple(BACKBONE_BUILDERS.keys())
RESNET_BACKBONES = ("resnet18", "resnet34", "resnet50")
EFFICIENTNET_BACKBONES = ("efficientnet_b0",)
MOBILENET_BACKBONES = ("mobilenet_v3_large",)
VGG_BACKBONES = ("vgg16", "vgg19")
VIT_BACKBONES = ("vit_b_16",)
SWIN_BACKBONES = ("swin_t",)


class TorchBackbone(BaseBackbone):
    """Torchvision CNN backbone returning final and per-stage feature maps."""

    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown torch backbone: %s. Supported: %s"
                             % (backbone, ", ".join(SUPPORTED_BACKBONES)))

        net = BACKBONE_BUILDERS[backbone](weights=None)
        if pretrained:
            self.load_local_weights(net, BACKBONE_WEIGHTS[backbone])

        self.backbone_name = backbone
        if backbone in RESNET_BACKBONES:
            self.family = "resnet"
            self.conv1 = net.conv1
            self.bn1 = net.bn1
            self.relu = net.relu
            self.maxpool = net.maxpool
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.out_channels = net.fc.in_features
            self.stage_channels = self.resnet_stage_channels(backbone)
            self.stage_strides = (4, 8, 16, 32)
        elif backbone in EFFICIENTNET_BACKBONES:
            self.family = "efficientnet"
            self.features = net.features
            self.out_channels = net.classifier[-1].in_features
            self.stage_indices = (1, 2, 3, 5, 8)
            self.stage_channels = (16, 24, 40, 112, 1280)
            self.stage_strides = (2, 4, 8, 16, 32)
        elif backbone in MOBILENET_BACKBONES:
            self.family = "mobilenet"
            self.features = net.features
            self.out_channels = net.classifier[0].in_features
            self.stage_indices = (1, 3, 6, 12, 16)
            self.stage_channels = (16, 24, 40, 112, 960)
            self.stage_strides = (2, 4, 8, 16, 32)
        elif backbone in SWIN_BACKBONES:
            self.family = "swin"
            self.stem = net.features
            self.norm = net.norm
            self.out_channels = net.head.in_features
            self.stage_indices = (1, 3, 5, 7)
            self.stage_channels = (96, 192, 384, 768)
            self.stage_strides = (4, 8, 16, 32)
        elif backbone in VIT_BACKBONES:
            self.family = "vit"
            self.conv_proj = net.conv_proj
            self.class_token = net.class_token
            self.encoder = net.encoder
            self.out_channels = net.hidden_dim
            self.patch_size = net.patch_size
        elif backbone in VGG_BACKBONES:
            self.family = "vgg"
            self.features = net.features
            self.out_channels = 512
            self.stage_channels = (64, 128, 256, 512, 512)
            self.stage_strides = (2, 4, 8, 16, 32)
        self.out_stride = 32

    def load_local_weights(self, net, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local torchvision weight not found: %s" % path)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict, strict=True)

    def resnet_stage_channels(self, backbone):
        if backbone in ("resnet18", "resnet34"):
            return (64, 128, 256, 512)
        return (256, 512, 1024, 2048)

    def forward(self, images):
        if self.family in ("efficientnet", "mobilenet"):
            x = images
            stages = []
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in self.stage_indices:
                    stages.append(x)
            return {"final": stages[-1], "stages": stages}

        if self.family == "swin":
            x = images
            stages = []
            for i, layer in enumerate(self.stem):
                x = layer(x)
                if i in self.stage_indices:
                    stage = x
                    if i == self.stage_indices[-1]:
                        stage = self.norm(stage)
                    stages.append(stage.permute(0, 3, 1, 2).contiguous())
            return {"final": stages[-1], "stages": stages}

        if self.family == "vit":
            n = images.shape[0]
            patches = self.conv_proj(images).flatten(2).transpose(1, 2)
            cls = self.class_token.expand(n, -1, -1)
            tokens = self.encoder(torch.cat([cls, patches], dim=1))
            grid_h = images.shape[2] // self.patch_size
            grid_w = images.shape[3] // self.patch_size
            return {"cls": tokens[:, 0], "tokens": tokens[:, 1:], "grid_size": (grid_h, grid_w)}

        if self.family == "vgg":
            x = images
            stages = []
            for layer in self.features:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2d):
                    stages.append(x)
            return {"final": stages[-1], "stages": stages}

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return {"final": s4, "stages": [s1, s2, s3, s4]}
