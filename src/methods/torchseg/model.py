# src/methods/torchseg/model.py: torchvision whole-model segmentation adapted to binary mask logits

import os
import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation

from src.methods.base.model import BaseModel

TORCHSEG_WEIGHTS = {
    "fcn_resnet50": "/mnt/d/backbones/fcn_resnet50_coco-1167a1af.pth",
    "deeplabv3_resnet50": "/mnt/d/backbones/deeplabv3_resnet50_coco-cd0a2569.pth",
    "deeplabv3_mobilenet_v3_large": "/mnt/d/backbones/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
    "lraspp_mobilenet_v3_large": "/mnt/d/backbones/lraspp_mobilenet_v3_large-d234d4ea.pth",
}
TORCHSEG_BUILDERS = {
    "fcn_resnet50": segmentation.fcn_resnet50,
    "deeplabv3_resnet50": segmentation.deeplabv3_resnet50,
    "deeplabv3_mobilenet_v3_large": segmentation.deeplabv3_mobilenet_v3_large,
    "lraspp_mobilenet_v3_large": segmentation.lraspp_mobilenet_v3_large,
}
SUPPORTED_TORCHSEG_MODELS = tuple(TORCHSEG_BUILDERS.keys())


class TorchSegModel(BaseModel):
    """Torchvision whole segmentation model adapted to project binary mask logits."""

    def __init__(self, model="fcn_resnet50", pretrained=True):
        super().__init__()
        model = model or "fcn_resnet50"
        if model not in TORCHSEG_BUILDERS:
            raise ValueError("Unknown torchseg model: %s. Supported: %s"
                             % (model, ", ".join(SUPPORTED_TORCHSEG_MODELS)))

        self.model_name = model
        self.net = self.build_model(model, pretrained=pretrained)
        self.mask_stride = 1

    def build_model(self, model, pretrained=True):
        builder = TORCHSEG_BUILDERS[model]
        kwargs = {"weights": None, "weights_backbone": None}
        if pretrained:
            if model != "lraspp_mobilenet_v3_large":
                kwargs["aux_loss"] = True
            net = builder(**kwargs)
            self.load_local_weights(net, TORCHSEG_WEIGHTS[model])
            self.replace_binary_classifier(net)
            return net

        kwargs["num_classes"] = 1
        return builder(**kwargs)

    def load_local_weights(self, net, path):
        if not os.path.exists(path):
            raise FileNotFoundError("Local torchseg weight not found: %s" % path)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state_dict, strict=True)

    def replace_binary_classifier(self, net):
        if self.model_name == "lraspp_mobilenet_v3_large":
            low = net.classifier.low_classifier
            high = net.classifier.high_classifier
            net.classifier.low_classifier = nn.Conv2d(low.in_channels, 1, kernel_size=1)
            net.classifier.high_classifier = nn.Conv2d(high.in_channels, 1, kernel_size=1)
            return

        last = net.classifier[-1]
        net.classifier[-1] = nn.Conv2d(last.in_channels, 1, kernel_size=1)
        if net.aux_classifier is not None:
            aux_last = net.aux_classifier[-1]
            net.aux_classifier[-1] = nn.Conv2d(aux_last.in_channels, 1, kernel_size=1)

    def forward(self, images):
        return self.net(images)["out"]
