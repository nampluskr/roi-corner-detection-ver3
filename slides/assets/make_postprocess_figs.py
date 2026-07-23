# slides/assets/make_postprocess_figs.py: generate schematic postprocess visualizations for slides

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
GRID = 224

TL = (0.16, 0.20)
TR = (0.86, 0.24)
BR = (0.82, 0.82)
BL = (0.20, 0.78)
CORNERS = np.array([TL, TR, BR, BL])
LABELS = ["TL", "TR", "BR", "BL"]


def to_px(pts):
    return pts * GRID


def base_scene(ax, title):
    ax.set_xlim(0, GRID)
    ax.set_ylim(GRID, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_gt_poly(ax, color="#2f6f4e", lw=1.5, ls="--"):
    poly = Polygon(to_px(CORNERS), closed=True, fill=False,
                   edgecolor=color, lw=lw, ls=ls, label="target")
    ax.add_patch(poly)


def draw_pred_corners(ax, pts, color="#c0392b"):
    px = to_px(pts)
    ax.add_patch(Polygon(px, closed=True, fill=False, edgecolor=color, lw=2.0))
    ax.scatter(px[:, 0], px[:, 1], c=color, s=45, zorder=5)
    for (x, y), lab in zip(px, LABELS):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(6, -6), fontsize=8, color=color)


def gaussian_map(centers, sigma):
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    acc = np.zeros((GRID, GRID))
    for cx, cy in centers:
        acc = np.maximum(acc, np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
    return acc


def ridge_map(sigma):
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    acc = np.zeros((GRID, GRID))
    px = to_px(CORNERS)
    for i in range(4):
        p1 = px[i]
        p2 = px[(i + 1) % 4]
        d = p2 - p1
        n = np.array([-d[1], d[0]])
        n = n / (np.linalg.norm(n) + 1e-9)
        dist = (xs - p1[0]) * n[0] + (ys - p1[1]) * n[1]
        acc = np.maximum(acc, np.exp(-(dist ** 2) / (2 * sigma ** 2)))
    return acc


def fig_seg():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    mask = np.zeros((GRID, GRID))
    from matplotlib.path import Path
    poly_path = Path(to_px(CORNERS))
    pts = np.vstack([xs.ravel(), ys.ravel()]).T
    mask = poly_path.contains_points(pts).reshape(GRID, GRID).astype(float)
    base_scene(axes[0], "seg: predicted mask")
    axes[0].imshow(mask, cmap="viridis", extent=[0, GRID, GRID, 0])
    base_scene(axes[1], "seg: extreme-point corners")
    axes[1].imshow(mask, cmap="gray", alpha=0.5, extent=[0, GRID, GRID, 0])
    draw_gt_poly(axes[1])
    draw_pred_corners(axes[1], CORNERS + 0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_seg.png"), dpi=140)
    plt.close(fig)


def fig_peak():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    heat = gaussian_map(to_px(CORNERS), sigma=8)
    base_scene(axes[0], "peak: 4-channel Gaussian heatmap")
    axes[0].imshow(heat, cmap="magma", extent=[0, GRID, GRID, 0])
    base_scene(axes[1], "peak: per-channel argmax corners")
    axes[1].imshow(heat, cmap="magma", alpha=0.6, extent=[0, GRID, GRID, 0])
    draw_gt_poly(axes[1], color="#cfe")
    draw_pred_corners(axes[1], CORNERS, color="#39d")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_peak.png"), dpi=140)
    plt.close(fig)


def fig_ridge_pcaline():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    rmap = ridge_map(sigma=6)
    base_scene(axes[0], "ridge: 4-channel edge Gaussian")
    axes[0].imshow(rmap, cmap="cividis", extent=[0, GRID, GRID, 0])
    base_scene(axes[1], "pcaline: PCA lines + intersection")
    axes[1].imshow(rmap, cmap="cividis", alpha=0.5, extent=[0, GRID, GRID, 0])
    px = to_px(CORNERS)
    for i in range(4):
        p1 = px[i]
        p2 = px[(i + 1) % 4]
        d = p2 - p1
        ext1 = p1 - d * 0.4
        ext2 = p2 + d * 0.4
        axes[1].plot([ext1[0], ext2[0]], [ext1[1], ext2[1]], color="#f5a", lw=1.2)
    draw_pred_corners(axes[1], CORNERS, color="#f5a")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_ridge_pcaline.png"), dpi=140)
    plt.close(fig)


def ridge_channel(i, sigma):
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    px = to_px(CORNERS)
    p1 = px[i]
    p2 = px[(i + 1) % 4]
    d = p2 - p1
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-9)
    dist = (xs - p1[0]) * n[0] + (ys - p1[1]) * n[1]
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))


def fig_ridge_peakprod():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    sigma = 6
    channels = [ridge_channel(i, sigma) for i in range(4)]
    rmap = np.maximum.reduce(channels)
    base_scene(axes[0], "ridge: 4-channel edge Gaussian")
    axes[0].imshow(rmap, cmap="cividis", extent=[0, GRID, GRID, 0])
    corner_maps = np.maximum.reduce([channels[(i - 1) % 4] * channels[i] for i in range(4)])
    base_scene(axes[1], "peakprod: adjacent-channel product + argmax")
    axes[1].imshow(corner_maps, cmap="magma", extent=[0, GRID, GRID, 0])
    draw_gt_poly(axes[1], color="#cfe")
    draw_pred_corners(axes[1], CORNERS, color="#39d")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_ridge_peakprod.png"), dpi=140)
    plt.close(fig)


def fig_det():
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    base_scene(ax, "det: grid cells + class assign -> box center")
    for g in range(0, GRID + 1, GRID // 7):
        ax.axhline(g, color="#ddd", lw=0.5)
        ax.axvline(g, color="#ddd", lw=0.5)
    draw_gt_poly(ax)
    px = to_px(CORNERS)
    colors = ["#e74c3c", "#27ae60", "#2980b9", "#8e44ad"]
    for (x, y), c, lab in zip(px, colors, LABELS):
        cell = GRID // 7
        cx = (x // cell) * cell
        cy = (y // cell) * cell
        ax.add_patch(plt.Rectangle((cx, cy), cell, cell, fill=True, alpha=0.3, color=c))
        ax.scatter([x], [y], c=c, s=45, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -6), fontsize=8, color=c)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_det.png"), dpi=140)
    plt.close(fig)


def fig_gcn():
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    base_scene(ax, "gcn: iterative refinement trajectory")
    draw_gt_poly(ax)
    init = CORNERS + np.array([[0.07, 0.05], [-0.06, 0.06], [-0.05, -0.06], [0.06, -0.05]])
    steps = [init]
    for _ in range(3):
        steps.append(steps[-1] + (CORNERS - steps[-1]) * 0.55)
    for s in range(4):
        px = to_px(steps[s])
        alpha = 0.25 + 0.25 * s
        ax.scatter(px[:, 0], px[:, 1], c="#d35400", s=30, alpha=alpha, zorder=4)
    for i in range(4):
        traj = to_px(np.array([steps[s][i] for s in range(4)]))
        ax.plot(traj[:, 0], traj[:, 1], color="#d35400", lw=1.0, alpha=0.6)
    draw_pred_corners(ax, steps[-1], color="#d35400")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_gcn.png"), dpi=140)
    plt.close(fig)


def fig_hybrid():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    from matplotlib.path import Path
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    pts = np.vstack([xs.ravel(), ys.ravel()]).T
    mask = Path(to_px(CORNERS)).contains_points(pts).reshape(GRID, GRID).astype(float)
    base_scene(axes[0], "hybrid: binary mask")
    axes[0].imshow(mask, cmap="gray", extent=[0, GRID, GRID, 0])
    edge = np.zeros((GRID, GRID))
    px = to_px(CORNERS)
    for i in range(4):
        p1 = px[i]
        p2 = px[(i + 1) % 4]
        for t in np.linspace(0, 1, 400):
            p = p1 + (p2 - p1) * t
            xi, yi = int(round(p[0])), int(round(p[1]))
            if 0 <= xi < GRID and 0 <= yi < GRID:
                edge[yi, xi] = 1
    base_scene(axes[1], "hybrid: Canny edges + Hough lines")
    axes[1].imshow(edge, cmap="gray", extent=[0, GRID, GRID, 0])
    for i in range(4):
        p1 = px[i]
        p2 = px[(i + 1) % 4]
        d = p2 - p1
        e1 = p1 - d * 0.2
        e2 = p2 + d * 0.2
        axes[1].plot([e1[0], e2[0]], [e1[1], e2[1]], color="#e67e22", lw=1.2)
    base_scene(axes[2], "hybrid: intersection corners")
    axes[2].imshow(mask, cmap="gray", alpha=0.4, extent=[0, GRID, GRID, 0])
    draw_gt_poly(axes[2])
    draw_pred_corners(axes[2], CORNERS, color="#e67e22")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_hybrid.png"), dpi=140)
    plt.close(fig)


def fig_reg_gap():
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    base_scene(ax, "reg gap: global pooled feature -> 8 logits")
    draw_gt_poly(ax)
    pred = CORNERS + np.array([[0.05, 0.04], [-0.05, 0.05], [-0.04, -0.05], [0.05, -0.04]])
    draw_pred_corners(ax, pred)
    ax.annotate("shifted polygon\n(weak spatial cue)", (0.5 * GRID, 0.5 * GRID),
                ha="center", va="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_reg_gap.png"), dpi=140)
    plt.close(fig)


def fig_reg_spatial():
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    base_scene(ax, "reg spatial: feature map -> strided conv -> 8 logits")
    draw_gt_poly(ax)
    pred = CORNERS + np.array([[0.015, 0.01], [-0.015, 0.012], [-0.01, -0.015], [0.012, -0.01]])
    draw_pred_corners(ax, pred)
    ax.annotate("tighter fit\n(position preserved)", (0.5 * GRID, 0.5 * GRID),
                ha="center", va="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_reg_spatial.png"), dpi=140)
    plt.close(fig)


def det_grid(ax):
    for g in range(0, GRID + 1, GRID // 7):
        ax.axhline(g, color="#ddd", lw=0.5)
        ax.axvline(g, color="#ddd", lw=0.5)


def fig_det_box():
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    base_scene(ax, "det box: cls cell + dx dy w h -> box center")
    det_grid(ax)
    draw_gt_poly(ax)
    px = to_px(CORNERS)
    colors = ["#e74c3c", "#27ae60", "#2980b9", "#8e44ad"]
    half = 0.1 * GRID / 2
    for (x, y), c, lab in zip(px, colors, LABELS):
        cell = GRID // 7
        cx = (x // cell) * cell
        cy = (y // cell) * cell
        ax.add_patch(plt.Rectangle((cx, cy), cell, cell, fill=True, alpha=0.25, color=c))
        ax.add_patch(plt.Rectangle((x - half, y - half), 2 * half, 2 * half,
                                   fill=False, edgecolor=c, lw=1.2))
        ax.scatter([x], [y], c=c, s=40, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -6), fontsize=8, color=c)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_det_box.png"), dpi=140)
    plt.close(fig)


def fig_det_point():
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    base_scene(ax, "det point: cls cell + dx dy -> box center")
    det_grid(ax)
    draw_gt_poly(ax)
    px = to_px(CORNERS)
    colors = ["#e74c3c", "#27ae60", "#2980b9", "#8e44ad"]
    for (x, y), c, lab in zip(px, colors, LABELS):
        cell = GRID // 7
        cx = (x // cell) * cell
        cy = (y // cell) * cell
        ax.add_patch(plt.Rectangle((cx, cy), cell, cell, fill=True, alpha=0.25, color=c))
        ax.scatter([x], [y], c=c, s=40, zorder=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -6), fontsize=8, color=c)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "postprocess_det_point.png"), dpi=140)
    plt.close(fig)


def main():
    fig_reg_gap()
    fig_reg_spatial()
    fig_seg()
    fig_peak()
    fig_ridge_pcaline()
    fig_ridge_peakprod()
    fig_det()
    fig_det_box()
    fig_det_point()
    fig_gcn()
    fig_hybrid()
    print("saved postprocess figures to", OUT_DIR)


if __name__ == "__main__":
    main()
