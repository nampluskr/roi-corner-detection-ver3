# scripts/batch_config.py: default batch experiment combinations for batch_run.py

# This is the default config used when batch_run.py runs without --config. For
# dataset stage specific runs, use configs/public.py, configs/synthetic.py,
# configs/measured.py, and configs/predict.py with --config. See
# docs/guides/06-use-cases.md for the scenario workflows.

# BASE holds the fields shared by every experiment. Each entry starts from BASE
# and overrides only model, network, head, and any per-run option.
# BASE = {
#     "dataset": "public",
#     "csv_path": ["data/public/smartdoc/gt_corners.csv", "data/public/midv2020/gt_corners.csv"],
#     "batch_size": 4,
#     "max_epochs": 10,
#     "train_size": 2000,
#     "valid_size": 1000,
#     "test_size": 1000,
# }

BASE = {
    "dataset": "synthetic",
    "csv_path": ["data/synthetic/gt_corners.csv"],
    "batch_size": 4,
    "max_epochs": 10,
    "train_size": 200,
    "valid_size": 50,
    "test_size": 50,
}

# reg
REG_CONFIGS = [
    # {**BASE, "model": "reg", "network": "custom", "head": "gap"},
    # {**BASE, "model": "reg", "network": "custom", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "resnet18", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "resnet34", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "resnet50", "head": "spatial", "warmup_epochs": 1},
    # {**BASE, "model": "reg", "network": "vgg16", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "vgg19", "head": "spatial"},

    # {**BASE, "model": "reg", "network": "vit_b_16", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "swin_t", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "wide_resnet50_2", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "deit_base_distilled", "head": "spatial"},
    # {**BASE, "model": "reg", "network": "cait_s24", "head": "spatial"},
]

# seg
SEG_CONFIGS = [
    # {**BASE, "model": "seg", "network": "custom", "head": "mask"},
    # {**BASE, "model": "seg", "network": "resnet18", "head": "mask", "warmup_epochs": 1},
    # {**BASE, "model": "seg", "network": "resnet34", "head": "mask"},
    # {**BASE, "model": "seg", "network": "resnet50", "head": "mask"},
    # {**BASE, "model": "seg", "network": "efficientnet_b0", "head": "mask"},
    # {**BASE, "model": "seg", "network": "swin_t", "head": "mask"},
    # {**BASE, "model": "seg", "network": "vgg16", "head": "mask"},
    # {**BASE, "model": "seg", "network": "vgg19", "head": "mask"},
    # {**BASE, "model": "seg", "network": "wide_resnet50_2", "head": "mask"},

    # {**BASE, "model": "torchseg", "network": "fcn_resnet50", "head": "mask"},
    # {**BASE, "model": "torchseg", "network": "deeplabv3_resnet50", "head": "mask"},
    # {**BASE, "model": "torchseg", "network": "deeplabv3_mobilenet_v3_large", "head": "mask"},
    # {**BASE, "model": "torchseg", "network": "lraspp_mobilenet_v3_large", "head": "mask"},
]

# peak
PEAK_CONFIGS = [
    # {**BASE, "model": "peak", "network": "resnet18", "head": "peak", "warmup_epochs": 1},
    # {**BASE, "model": "peak", "network": "custom", "head": "peak"},
    # {**BASE, "model": "peak", "network": "resnet34", "head": "peak"},
    # {**BASE, "model": "peak", "network": "resnet50", "head": "peak"},
    # {**BASE, "model": "peak", "network": "efficientnet_b0", "head": "peak"},
    # {**BASE, "model": "peak", "network": "swin_t", "head": "peak"},
    # {**BASE, "model": "peak", "network": "vgg16", "head": "peak"},
    # {**BASE, "model": "peak", "network": "vgg19", "head": "peak"},
    # {**BASE, "model": "peak", "network": "wide_resnet50_2", "head": "peak"},
]

# ridge
RIDGE_CONFIGS = [
    # {**BASE, "model": "ridge", "network": "resnet18", "head": "pcaline", "warmup_epochs": 1},
    # {**BASE, "model": "ridge", "network": "custom", "head": "pcaline"},
    # {**BASE, "model": "ridge", "network": "resnet34", "head": "pcaline"},
    # {**BASE, "model": "ridge", "network": "resnet50", "head": "pcaline"},
    # {**BASE, "model": "ridge", "network": "efficientnet_b0", "head": "pcaline"},
    # {**BASE, "model": "ridge", "network": "swin_t", "head": "pcaline"},
    # {**BASE, "model": "ridge", "network": "vgg16", "head": "pcaline"},
    # {**BASE, "model": "ridge", "network": "vgg16", "head": "peakprod"},
    # {**BASE, "model": "ridge", "network": "vgg19", "head": "pcaline"},
    # {**BASE, "model": "ridge", "network": "wide_resnet50_2", "head": "pcaline"},
]

# gcn
GCN_CONFIGS = [
    # {**BASE, "model": "gcn", "network": "custom", "head": "gcn"},
    # {**BASE, "model": "gcn", "network": "resnet18", "head": "gcn", "warmup_epochs": 1},
    # {**BASE, "model": "gcn", "network": "resnet34", "head": "gcn"},
    # {**BASE, "model": "gcn", "network": "resnet50", "head": "gcn"},
    # {**BASE, "model": "gcn", "network": "efficientnet_b0", "head": "gcn"},
    # {**BASE, "model": "gcn", "network": "vgg16", "head": "gcn"},
]

# hybrid
HYBRID_CONFIGS = [
    # {**BASE, "model": "hybrid", "network": "mobilenet_v3_large", "head": "hybrid", "warmup_epochs": 1},
    # {**BASE, "model": "hybrid", "network": "custom", "head": "hybrid"},
    # {**BASE, "model": "hybrid", "network": "resnet18", "head": "hybrid"},
    # {**BASE, "model": "hybrid", "network": "resnet50", "head": "hybrid"},
    # {**BASE, "model": "hybrid", "network": "vgg16", "head": "hybrid"},
]

# det
DET_CONFIGS = [
    # {**BASE, "model": "det", "network": "custom", "head": "box"},
    # {**BASE, "model": "det", "network": "custom", "head": "point"},
    # {**BASE, "model": "det", "network": "resnet18", "head": "box"},
    # {**BASE, "model": "det", "network": "resnet34", "head": "box"},
    # {**BASE, "model": "det", "network": "resnet50", "head": "box"},
    # {**BASE, "model": "det", "network": "efficientnet_b0", "head": "box"},
    # {**BASE, "model": "det", "network": "swin_t", "head": "box"},
    # {**BASE, "model": "det", "network": "vgg16", "head": "box"},
    # {**BASE, "model": "det", "network": "vgg19", "head": "box"},
    # {**BASE, "model": "det", "network": "wide_resnet50_2", "head": "box"},

    # {**BASE, "model": "torchdet", "network": "fasterrcnn_resnet50_fpn", "head": "box", "warmup_epochs": 1},
    # {**BASE, "model": "torchdet", "network": "fasterrcnn_resnet50_fpn", "head": "point", "warmup_epochs": 1},
    # {**BASE, "model": "torchdet", "network": "retinanet_resnet50_fpn", "head": "box"},
    # {**BASE, "model": "torchdet", "network": "retinanet_resnet50_fpn", "head": "point"},
    # {**BASE, "model": "torchdet", "network": "ssd300_vgg16", "head": "box"},
    # {**BASE, "model": "torchdet", "network": "ssd300_vgg16", "head": "point"},
    # {**BASE, "model": "detr", "network": "detr_resnet50", "head": "box"},
    # {**BASE, "model": "detr", "network": "detr_resnet50", "head": "point"},
    {**BASE, "model": "yolo", "network": "yolov8n", "head": "box"},
    # {**BASE, "model": "yolo", "network": "yolov8n", "head": "point"},
]

# method comparison templates; keep separate from the active CONFIGS queue
METHOD_COMPARISON_CONFIGS = [
    # {**BASE, "model": "reg", "network": "custom", "head": "gap", "checkpoint": "outputs/public/reg/reg_custom_gap_public/model.pth"},
    # {**BASE, "model": "seg", "network": "custom", "head": "mask", "checkpoint": "outputs/public/seg/seg_custom_mask_public/model.pth"},
    # {**BASE, "model": "det", "network": "custom", "head": "box", "checkpoint": "outputs/public/det/det_custom_box_public/model.pth"},
    # {**BASE, "model": "peak", "network": "custom", "head": "peak", "checkpoint": "outputs/public/peak/peak_custom_peak_public/model.pth"},
    # {**BASE, "model": "ridge", "network": "custom", "head": "pcaline", "checkpoint": "outputs/public/ridge/ridge_custom_pcaline_public/model.pth"},
]

CONFIGS = REG_CONFIGS + SEG_CONFIGS + PEAK_CONFIGS + RIDGE_CONFIGS + GCN_CONFIGS + HYBRID_CONFIGS + DET_CONFIGS
