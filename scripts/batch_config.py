# scripts/batch_config.py: batch experiment combinations for batch_run.py

# reg
REG_CONFIGS = [
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "gap"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "resnet18", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "resnet34", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "resnet50", "head": "spatial", "warmup_epochs": 1},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "vgg16", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "vgg19", "head": "spatial"},

    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "vit_b_16", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "swin_t", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "wide_resnet50_2.tv_in1k", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "deit_base_distilled_patch16_224.fb_in1k", "head": "spatial"},
    # {"model": "reg", "batch_size": 4, "max_epochs": 5, "network": "cait_s24_224.fb_dist_in1k", "head": "spatial"},
]

# seg
SEG_CONFIGS = [
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "mask"},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "resnet18", "head": "mask", "warmup_epochs": 1},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "resnet34", "head": "mask"},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "resnet50", "head": "mask"},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "efficientnet_b0", "head": "mask"},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "swin_t", "head": "mask"},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "vgg16", "head": "mask"},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "vgg19", "head": "mask"},
    # {"model": "seg", "batch_size": 4, "max_epochs": 5, "network": "wide_resnet50_2.tv_in1k", "head": "mask"},

    # {"model": "torchseg", "batch_size": 4, "max_epochs": 5, "network": "fcn_resnet50", "head": "mask"},
    # {"model": "torchseg", "batch_size": 4, "max_epochs": 5, "network": "deeplabv3_resnet50", "head": "mask"},
    # {"model": "torchseg", "batch_size": 4, "max_epochs": 5, "network": "deeplabv3_mobilenet_v3_large", "head": "mask"},
    # {"model": "torchseg", "batch_size": 4, "max_epochs": 5, "network": "lraspp_mobilenet_v3_large", "head": "mask"},
]

# peak
PEAK_CONFIGS = [
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "resnet18", "head": "peak", "warmup_epochs": 1},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "peak"},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "resnet34", "head": "peak"},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "resnet50", "head": "peak"},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "efficientnet_b0", "head": "peak"},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "swin_t", "head": "peak"},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "vgg16", "head": "peak"},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "vgg19", "head": "peak"},
    # {"model": "peak", "batch_size": 4, "max_epochs": 5, "network": "wide_resnet50_2.tv_in1k", "head": "peak"},
]

# ridge
RIDGE_CONFIGS = [
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "resnet18", "head": "ridge", "warmup_epochs": 1},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "ridge"},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "resnet34", "head": "ridge"},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "resnet50", "head": "ridge"},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "efficientnet_b0", "head": "ridge"},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "swin_t", "head": "ridge"},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "vgg16", "head": "ridge"},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "vgg19", "head": "ridge"},
    # {"model": "ridge", "batch_size": 4, "max_epochs": 5, "network": "wide_resnet50_2.tv_in1k", "head": "ridge"},
]

# gcn
GCN_CONFIGS = [
    # {"model": "gcn", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "gcn"},
    # {"model": "gcn", "batch_size": 4, "max_epochs": 5, "network": "resnet18", "head": "gcn", "warmup_epochs": 1},
    # {"model": "gcn", "batch_size": 4, "max_epochs": 5, "network": "resnet34", "head": "gcn"},
    # {"model": "gcn", "batch_size": 4, "max_epochs": 5, "network": "resnet50", "head": "gcn"},
    # {"model": "gcn", "batch_size": 4, "max_epochs": 5, "network": "efficientnet_b0", "head": "gcn"},
    # {"model": "gcn", "batch_size": 4, "max_epochs": 5, "network": "vgg16", "head": "gcn"},
]

# hybrid
HYBRID_CONFIGS = [
    # {"model": "hybrid", "batch_size": 4, "max_epochs": 5, "network": "mobilenet_v3_large", "head": "hybrid", "warmup_epochs": 1},
    # {"model": "hybrid", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "hybrid"},
    # {"model": "hybrid", "batch_size": 4, "max_epochs": 5, "network": "resnet18", "head": "hybrid"},
    # {"model": "hybrid", "batch_size": 4, "max_epochs": 5, "network": "resnet50", "head": "hybrid"},
    # {"model": "hybrid", "batch_size": 4, "max_epochs": 5, "network": "vgg16", "head": "hybrid"},
]

# det
DET_CONFIGS = [
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "custom", "head": "point"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "resnet18", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "resnet34", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "resnet50", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "efficientnet_b0", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "swin_t", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "vgg16", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "vgg19", "head": "box"},
    # {"model": "det", "batch_size": 4, "max_epochs": 5, "network": "wide_resnet50_2.tv_in1k", "head": "box"},

    # {"model": "torchdet", "batch_size": 4, "max_epochs": 5, "network": "fasterrcnn_resnet50_fpn", "head": "box", "warmup_epochs": 1},
    # {"model": "torchdet", "batch_size": 4, "max_epochs": 5, "network": "fasterrcnn_resnet50_fpn", "head": "point", "warmup_epochs": 1},
    # {"model": "torchdet", "batch_size": 4, "max_epochs": 5, "network": "retinanet_resnet50_fpn", "head": "box"},
    # {"model": "torchdet", "batch_size": 4, "max_epochs": 5, "network": "retinanet_resnet50_fpn", "head": "point"},
    # {"model": "torchdet", "batch_size": 4, "max_epochs": 5, "network": "ssd300_vgg16", "head": "box"},
    # {"model": "torchdet", "batch_size": 4, "max_epochs": 5, "network": "ssd300_vgg16", "head": "point"},
    # {"model": "detr", "batch_size": 4, "max_epochs": 5, "network": "detr_resnet50", "head": "box"},
    # {"model": "detr", "batch_size": 4, "max_epochs": 5, "network": "detr_resnet50", "head": "point"},
    # {"model": "yolo", "batch_size": 4, "max_epochs": 5, "network": "yolov8n", "head": "box"},
    # {"model": "yolo", "batch_size": 4, "max_epochs": 5, "network": "yolov8n", "head": "point"},
]

# method comparison templates; keep separate from the active CONFIGS queue
METHOD_COMPARISON_CONFIGS = [
    # {"model": "reg", "batch_size": 4, "network": "custom", "head": "gap", "test_size": 1000, "checkpoint": "outputs/public/reg/custom_gap/example/model.pth"},
    # {"model": "seg", "batch_size": 4, "network": "custom", "head": "mask", "test_size": 1000, "checkpoint": "outputs/public/seg/custom_mask/example/model.pth"},
    # {"model": "det", "batch_size": 4, "network": "custom", "head": "box", "test_size": 1000, "checkpoint": "outputs/public/det/custom_box/example/model.pth"},
    # {"model": "peak", "batch_size": 4, "network": "custom", "head": "peak", "test_size": 1000, "checkpoint": "outputs/public/peak/custom_peak/example/model.pth"},
    # {"model": "ridge", "batch_size": 4, "network": "custom", "head": "ridge", "test_size": 1000, "checkpoint": "outputs/public/ridge/custom_ridge/example/model.pth"},
]

CONFIGS = REG_CONFIGS + SEG_CONFIGS + PEAK_CONFIGS + RIDGE_CONFIGS + GCN_CONFIGS + HYBRID_CONFIGS + DET_CONFIGS
