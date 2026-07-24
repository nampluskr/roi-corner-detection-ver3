# configs/public.py: batch experiment combinations for the public dataset stage

# BASE holds the fields shared by every experiment on the public stage. Each
# entry in CONFIGS starts from BASE and overrides only what changes, so dataset
# and csv_path are set once. Every dict must include dataset and csv_path.

PUBLIC_CSV = [
    "data/public/smartdoc/gt_corners.csv",
    "data/public/midv2020/gt_corners.csv",
]

BASE = {
    "dataset": "public",
    "csv_path": PUBLIC_CSV,
    "batch_size": 4,
    "max_epochs": 5,
    "train_size": 5000,
    "valid_size": 1000,
    "test_size": 1000,
}

CONFIGS = [
    {**BASE, "model": "reg", "network": "custom", "head": "gap"},

    # {**BASE, "model": "reg", "network": "resnet18", "head": "spatial"},
    # {**BASE, "model": "seg", "network": "custom", "head": "mask"},
    # {**BASE, "model": "peak", "network": "custom", "head": "peak"},
]
