# configs/measured.py: batch experiment combinations for the measured dataset stage

# BASE holds the fields shared by every experiment on the measured stage. Reuse
# the same model, network, head, and experiment identity as the synthetic stage
# so train.py carries over weights from the synthetic stage model.pth. Each entry
# in CONFIGS starts from BASE and overrides only what changes.

MEASURED_CSV = [
    "data/measured/gt_corners.csv",
]

BASE = {
    "dataset": "measured",
    "csv_path": MEASURED_CSV,
    "batch_size": 4,
    "max_epochs": 5,
    "train_size": 5000,
    "valid_size": 1000,
    "test_size": 1000,
}

CONFIGS = [
    {**BASE, "model": "reg", "network": "custom", "head": "gap"},

    # {**BASE, "model": "seg", "network": "custom", "head": "mask"},
    # {**BASE, "model": "peak", "network": "custom", "head": "peak"},
]
