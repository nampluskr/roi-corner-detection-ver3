# configs/predict.py: inference-only combinations for the measured dataset stage

# BASE holds the fields shared by every inference target run with
# batch_run.py --mode predict. These configs skip metric evaluation and save
# predictions.csv only. Set checkpoint explicitly or omit it to use the stage
# output model.pth. Each entry in CONFIGS starts from BASE and overrides only
# what changes.

MEASURED_CSV = [
    "data/measured/gt_corners.csv",
]

BASE = {
    "dataset": "measured",
    "csv_path": MEASURED_CSV,
    "batch_size": 4,
    "test_size": 1000,
}

CONFIGS = [
    {**BASE, "model": "reg", "network": "custom", "head": "gap"},

    # {**BASE, "model": "seg", "network": "custom", "head": "mask"},
]
