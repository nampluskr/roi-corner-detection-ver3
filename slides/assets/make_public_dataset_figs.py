# slides/assets/make_public_dataset_figs.py: render public dataset corner examples for slides

import csv
import os

CACHE_DIR = os.path.join("/tmp", "roi_corner_public_dataset_figs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(CACHE_DIR, "cache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = ["TL", "TR", "BR", "BL"]


def read_first_valid_sample(csv_path):
    """Return the first CSV row whose image file exists."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV file not found: {}".format(csv_path))
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = os.path.join(row["image_dir"], row["image_name"])
            if not os.path.exists(image_path):
                continue
            corners = np.array(
                [
                    [float(row["x1"]), float(row["y1"])],
                    [float(row["x2"]), float(row["y2"])],
                    [float(row["x3"]), float(row["y3"])],
                    [float(row["x4"]), float(row["y4"])],
                ],
                dtype=np.float32,
            )
            return image_path, corners
    raise FileNotFoundError("No image referenced by CSV exists: {}".format(csv_path))


def draw_example(dataset_name, csv_path, output_name):
    """Draw one public dataset sample with ordered corner labels."""
    image_path, corners = read_first_valid_sample(csv_path)
    image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    width, height = image.size
    scale = np.array([width, height], dtype=np.float32)
    pixel_corners = corners * scale
    polygon = np.vstack([pixel_corners, pixel_corners[0]])

    aspect = width / float(height)
    fig_width = 6.4 if aspect >= 1.0 else 4.2
    fig_height = fig_width / aspect
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.imshow(image)
    ax.plot(polygon[:, 0], polygon[:, 1], "-", color="#d62728", lw=3.0)
    ax.scatter(pixel_corners[:, 0], pixel_corners[:, 1], c="#d62728", s=70, zorder=5)
    for (x, y), label in zip(pixel_corners, LABELS):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(7, -7),
            fontsize=10,
            color="white",
            weight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#d62728", "edgecolor": "none"},
        )
    ax.set_title("{} public sample: ordered ROI corners".format(dataset_name), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    out_path = os.path.join(OUT_DIR, output_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved {} from {}".format(out_path, image_path))


def main():
    """Render SmartDoc and MIDV2020 public dataset examples."""
    examples = [
        (
            "SmartDoc",
            os.path.join(PROJECT_ROOT, "data", "public", "smartdoc", "gt_corners.csv"),
            "public_smartdoc_example.png",
        ),
        (
            "MIDV2020",
            os.path.join(PROJECT_ROOT, "data", "public", "midv2020", "gt_corners.csv"),
            "public_midv2020_example.png",
        ),
    ]
    for dataset_name, csv_path, output_name in examples:
        draw_example(dataset_name, csv_path, output_name)


if __name__ == "__main__":
    main()
