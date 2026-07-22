# src/data/labelme.py: parse LabelMe ROI polygon JSON into gt_corners.csv

import os
import csv
import json
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.geometry import order_corners, is_invalid_corners

CSV_HEADER = ["image_dir", "image_name", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def create_data(data_dir, output_path):
    """Parse LabelMe JSON annotations under data_dir recursively and write gt_corners.csv to output_path."""
    json_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))

    rows = []
    for json_path in tqdm(json_paths, desc="labelme"):
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)

        shapes = [s for s in payload.get("shapes", []) if s.get("label") == "roi"]
        if not shapes:
            continue
        points = np.array(shapes[0]["points"], dtype=np.float32)
        if points.shape != (4, 2):
            continue

        json_dir = os.path.dirname(json_path)
        image_name = payload.get("imagePath") or os.path.splitext(os.path.basename(json_path))[0] + ".png"
        img_path = os.path.join(json_dir, image_name)
        if not os.path.exists(img_path):
            continue

        img_w = payload.get("imageWidth")
        img_h = payload.get("imageHeight")
        if not img_w or not img_h:
            img_w, img_h = Image.open(img_path).size

        normalized = points / np.array([img_w, img_h], dtype=np.float32)
        ordered = order_corners(normalized)
        if is_invalid_corners(ordered):
            continue

        rows.append([os.path.abspath(json_dir), image_name] + ["%.4f" % v for v in ordered.reshape(8)])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
