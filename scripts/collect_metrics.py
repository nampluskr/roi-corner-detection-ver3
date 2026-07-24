# scripts/collect_metrics.py: collect metrics.json files into one summary CSV

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import pandas as pd

DEFAULT_OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")


def parse_identity(metrics_path, outputs_dir):
    """Return dataset, model, and exp_name parsed from a metrics.json path."""
    exp_dir = os.path.dirname(metrics_path)
    rel = os.path.relpath(exp_dir, outputs_dir)
    parts = rel.split(os.sep)
    if len(parts) != 3:
        return None
    dataset, model, exp_name = parts
    return {
        "dataset": dataset,
        "model": model,
        "exp_name": exp_name,
    }


def collect_rows(outputs_dir, dataset=None):
    """Walk the outputs tree and return one row per metrics.json file."""
    root = outputs_dir
    if dataset is not None:
        root = os.path.join(outputs_dir, dataset)
    rows = []
    for current, _, files in os.walk(root):
        if "metrics.json" not in files:
            continue
        metrics_path = os.path.join(current, "metrics.json")
        identity = parse_identity(metrics_path, outputs_dir)
        if identity is None:
            continue
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        row = dict(identity)
        row.update(metrics)
        rows.append(row)
    return rows


def main():
    """Collect metrics.json files under outputs into a single summary CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default=DEFAULT_OUTPUTS)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = collect_rows(args.outputs_dir, dataset=args.dataset)
    if not rows:
        print("no metrics.json found under %s" % (
            os.path.join(args.outputs_dir, args.dataset) if args.dataset else args.outputs_dir))
        return

    frame = pd.DataFrame(rows)
    lead = ["dataset", "model", "exp_name"]
    metric_cols = [c for c in frame.columns if c not in lead]
    frame = frame[lead + sorted(metric_cols)]
    frame = frame.sort_values(lead).reset_index(drop=True)

    output = args.output
    if output is None:
        name = "metrics_summary.csv" if args.dataset is None else "%s_metrics_summary.csv" % args.dataset
        output = os.path.join(args.outputs_dir, name)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    frame.to_csv(output, index=False)
    print("saved %d rows to %s" % (len(frame), output))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
