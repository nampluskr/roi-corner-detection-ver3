# slides/assets/make_transform_figs.py: render real transform examples on a synthetic fringe panel

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.data.transforms import (
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    ColorJitter, GaussianBlur,
    RandomPerspective, RandomScale, RandomAffine,
    ToTensor, GaussianNoise, ToNumpy,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIZE = 256
SEED = 7

TL = (0.18, 0.20)
TR = (0.82, 0.24)
BR = (0.80, 0.80)
BL = (0.22, 0.78)
CORNERS = np.array([TL, TR, BR, BL], dtype=np.float32)
LABELS = ["TL", "TR", "BR", "BL"]


def make_panel():
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    xn = xs / SIZE
    yn = ys / SIZE
    from matplotlib.path import Path
    poly = Path(CORNERS * SIZE)
    pts = np.vstack([xs.ravel(), ys.ravel()]).T
    mask = poly.contains_points(pts).reshape(SIZE, SIZE)
    background = 0.10 + 0.05 * np.sin(2 * np.pi * (xn + yn))
    fringe = 0.5 + 0.4 * np.cos(2 * np.pi * 11 * xn + 0.6)
    img = np.where(mask, fringe, background)
    cx, cy = 0.5 * SIZE, 0.28 * SIZE
    hole = (xs - cx) ** 2 + (ys - cy) ** 2 < (0.03 * SIZE) ** 2
    img = np.where(hole & mask, 0.0, img)
    rgb = np.clip(np.stack([img, img, img], axis=-1), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def draw(ax, image, corners, title):
    ax.imshow(np.asarray(image))
    px = corners * SIZE
    poly = np.vstack([px, px[0]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color="#c0392b", lw=1.8)
    ax.scatter(px[:, 0], px[:, 1], c="#c0392b", s=30, zorder=5)
    for (x, y), lab in zip(px, LABELS):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(5, -5), fontsize=7, color="#c0392b")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def tensor_transform(image, corners, tf):
    to_tensor = ToTensor()
    to_numpy = ToNumpy()
    t_img, t_cor = to_tensor(image, corners.copy())
    t_img, t_cor = tf(t_img, t_cor)
    np_img, np_cor = to_numpy(t_img, t_cor)
    return Image.fromarray((np.clip(np_img, 0, 1) * 255).astype(np.uint8)), np_cor


def fig_offline(panel):
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    steps = [
        ("original", None),
        ("RandomPerspective(0.35)", RandomPerspective(distortion_scale=0.35, p=1.0)),
        ("RandomAffine(deg=12, shear=8)", RandomAffine(degrees=12, translate=(0.06, 0.06),
                                                       scale_range=(0.85, 1.15), shear=8)),
        ("RandomScale(0.75, 1.2)", RandomScale(scale_range=(0.75, 1.2))),
        ("GaussianNoise(0.08)", "noise"),
    ]
    fig, axes = plt.subplots(1, len(steps), figsize=(3.0 * len(steps), 3.2))
    for ax, (title, tf) in zip(axes, steps):
        if tf is None:
            draw(ax, panel, CORNERS.copy(), title)
        elif tf == "noise":
            img, cor = tensor_transform(panel, CORNERS.copy(), GaussianNoise(std=0.08))
            draw(ax, img, cor, title)
        else:
            img, cor = tf(panel.copy(), CORNERS.copy())
            draw(ax, img, cor, title)
    fig.suptitle("offline pre-augmentation: strong distortion, corners transformed jointly", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "transform_offline.png"), dpi=140)
    plt.close(fig)


def fig_online(panel):
    random.seed(SEED + 1)
    np.random.seed(SEED + 1)
    steps = [
        ("original", None),
        ("RandomHorizontalFlip", RandomHorizontalFlip(p=1.0)),
        ("RandomVerticalFlip", RandomVerticalFlip(p=1.0)),
        ("RandomRotation(5)", RandomRotation(degrees=5)),
        ("ColorJitter + GaussianBlur", "photo"),
    ]
    fig, axes = plt.subplots(1, len(steps), figsize=(3.0 * len(steps), 3.2))
    for ax, (title, tf) in zip(axes, steps):
        if tf is None:
            draw(ax, panel, CORNERS.copy(), title)
        elif tf == "photo":
            jit = ColorJitter(brightness=0.3, contrast=0.3)
            blur = GaussianBlur(kernel_size=3, sigma=(1.5, 2.0))
            img, cor = jit(panel.copy(), CORNERS.copy())
            img, cor = blur(img, cor)
            draw(ax, img, cor, title)
        else:
            img, cor = tf(panel.copy(), CORNERS.copy())
            draw(ax, img, cor, title)
    fig.suptitle("online transform: simple flip, small rotation, optical jitter", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "transform_online.png"), dpi=140)
    plt.close(fig)


def main():
    panel = make_panel()
    fig_offline(panel)
    fig_online(panel)
    print("saved transform figures to", OUT_DIR)


if __name__ == "__main__":
    main()
