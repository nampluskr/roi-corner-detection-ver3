# slides/assets/make_synthetic_variation_figs.py: render synthetic generation variation examples for slides

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZE = 256
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def rounded_rect_mask(cx, cy, hw, hh, radius, rot_deg):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    rad = np.radians(rot_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx = xs - cx
    dy = ys - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    ax = np.abs(lx) - (hw - radius)
    ay = np.abs(ly) - (hh - radius)
    outside = np.sqrt(np.maximum(ax, 0) ** 2 + np.maximum(ay, 0) ** 2)
    inside = np.minimum(np.maximum(ax, ay), 0)
    dist = outside + inside - radius
    return dist <= 0, (lx, ly)


def corner_points(cx, cy, hw, hh, rot_deg, top_ratio):
    rad = np.radians(rot_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    tw = hw * top_ratio
    local = np.array([[-tw, -hh], [tw, -hh], [hw, hh], [-hw, hh]], dtype=float)
    out = np.empty_like(local)
    out[:, 0] = cos_a * local[:, 0] - sin_a * local[:, 1] + cx
    out[:, 1] = sin_a * local[:, 0] + cos_a * local[:, 1] + cy
    return out


def render_panel(cx=128, cy=128, hw=78, hh=54, radius=10, rot_deg=0.0, top_ratio=1.0,
                 freq=11, phase=0.0, amp=0.4, bg_mean=0.10, bg_gradient=0.0,
                 fringe_dir="vertical", bow=0.0, hole_center=None, hole_r=8, hole_dark=0.0,
                 holders=None):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    xn = xs / SIZE
    yn = ys / SIZE
    mask, (lx, ly) = rounded_rect_mask(cx, cy, hw, hh, radius, rot_deg)
    background = bg_mean + bg_gradient * (xn - 0.5) + 0.03 * np.sin(2 * np.pi * (xn + yn))
    axis = ly if fringe_dir == "horizontal" else lx
    warped = axis / SIZE + bow * ((ly / SIZE) ** 2)
    fringe = 0.5 + amp * np.cos(2 * np.pi * freq * warped + phase)
    img = np.where(mask, fringe, background)
    if holders:
        for hx, hy, hwid, hdep, hbright, side in holders:
            hmask = holder_mask(cx, cy, hw, hh, rot_deg, hx, hwid, hdep, side)
            img = np.where(hmask, hbright, img)
    if hole_center is not None:
        hcx, hcy = hole_center
        hole = (xs - hcx) ** 2 + (ys - hcy) ** 2 < hole_r ** 2
        img = np.where(hole & mask, hole_dark, img)
    return np.clip(img, 0, 1), corner_points(cx, cy, hw, hh, rot_deg, top_ratio)


def holder_mask(cx, cy, hw, hh, rot_deg, pos_frac, width_frac, depth_frac, side):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    rad = np.radians(rot_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx = xs - cx
    dy = ys - cy
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    if side == "left":
        along = ly
        edge = -hw
        half_len = hh * width_frac
        depth = hw * depth_frac
        center = (pos_frac - 0.5) * 2 * hh
        return (np.abs(along - center) < half_len) & (lx < edge) & (lx > edge - depth)
    if side == "right":
        along = ly
        edge = hw
        half_len = hh * width_frac
        depth = hw * depth_frac
        center = (pos_frac - 0.5) * 2 * hh
        return (np.abs(along - center) < half_len) & (lx > edge) & (lx < edge + depth)
    if side == "top":
        along = lx
        edge = -hh
        half_len = hw * width_frac
        depth = hh * depth_frac
        center = (pos_frac - 0.5) * 2 * hw
        return (np.abs(along - center) < half_len) & (ly < edge) & (ly > edge - depth)
    along = lx
    edge = hh
    half_len = hw * width_frac
    depth = hh * depth_frac
    center = (pos_frac - 0.5) * 2 * hw
    return (np.abs(along - center) < half_len) & (ly > edge) & (ly < edge + depth)


def draw(ax, img, corners, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    poly = np.vstack([corners, corners[0]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color="#c0392b", lw=1.4)
    ax.scatter(corners[:, 0], corners[:, 1], c="#c0392b", s=14, zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def grid(items, suptitle, outfile, ncol=4):
    n = len(items)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, (title, kwargs) in zip(axes, items):
        img, corners = render_panel(**kwargs)
        draw(ax, img, corners, title)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, outfile), dpi=140)
    plt.close(fig)


def fig_position():
    items = [
        ("center, no tilt", dict(rot_deg=0, top_ratio=1.0)),
        ("rotation +9deg", dict(rot_deg=9, top_ratio=0.98)),
        ("left shift + trapezoid", dict(cx=100, rot_deg=-6, top_ratio=0.86)),
        ("right shift + rotation", dict(cx=156, cy=120, rot_deg=7, top_ratio=0.90)),
    ]
    grid(items, "position and pose: shift, rotation, tilt trapezoid", "synth_position.png")


def fig_rounding():
    items = [
        ("radius 3%", dict(radius=4)),
        ("radius 5%", dict(radius=9)),
        ("radius 8%", dict(radius=15)),
        ("square panel, radius 6%", dict(hw=62, hh=62, radius=11)),
    ]
    grid(items, "corner rounding: rounded radius variation", "synth_rounding.png")


def fig_holder():
    items = [
        ("left-right, 1 each", dict(holders=[(0.5, 0, 0.22, 0.14, 0.25, "left"),
                                             (0.5, 0, 0.22, 0.14, 0.25, "right")])),
        ("top-bottom, 2 each", dict(holders=[(0.32, 0, 0.14, 0.16, 0.3, "top"),
                                             (0.68, 0, 0.14, 0.16, 0.3, "top"),
                                             (0.32, 0, 0.14, 0.16, 0.3, "bottom"),
                                             (0.68, 0, 0.14, 0.16, 0.3, "bottom")])),
        ("four sides", dict(holders=[(0.5, 0, 0.2, 0.13, 0.2, "left"),
                                     (0.5, 0, 0.2, 0.13, 0.2, "right"),
                                     (0.5, 0, 0.2, 0.13, 0.2, "top"),
                                     (0.5, 0, 0.2, 0.13, 0.2, "bottom")])),
        ("left-right, 3 each", dict(holders=[(0.25, 0, 0.1, 0.16, 0.28, "left"),
                                             (0.5, 0, 0.1, 0.16, 0.28, "left"),
                                             (0.75, 0, 0.1, 0.16, 0.28, "left"),
                                             (0.25, 0, 0.1, 0.16, 0.28, "right"),
                                             (0.5, 0, 0.1, 0.16, 0.28, "right"),
                                             (0.75, 0, 0.1, 0.16, 0.28, "right")])),
    ]
    grid(items, "holder jig: layout, count, size", "synth_holder.png")


def fig_camera_hole():
    items = [
        ("top-center, visible", dict(hole_center=(128, 74), hole_r=9, hole_dark=0.0, phase=0.0)),
        ("upper-left, visible", dict(hole_center=(96, 82), hole_r=9, hole_dark=0.0, phase=0.0)),
        ("top-center, partial", dict(hole_center=(128, 74), hole_r=9, hole_dark=0.35, phase=1.4)),
        ("top-center, hidden", dict(hole_center=(128, 74), hole_r=6, hole_dark=0.12, phase=3.14, freq=10)),
    ]
    grid(items, "camera hole: position and visibility", "synth_camera_hole.png")


def fig_background():
    items = [
        ("dark stage 0.05", dict(bg_mean=0.05)),
        ("medium stage 0.35", dict(bg_mean=0.35)),
        ("bright stage 0.75", dict(bg_mean=0.75)),
        ("illumination gradient", dict(bg_mean=0.4, bg_gradient=0.5)),
    ]
    grid(items, "background brightness and illumination", "synth_background.png")


def fig_fringe():
    items = [
        ("coarse freq 8", dict(freq=8, amp=0.32, fringe_dir="vertical")),
        ("dense freq 30", dict(freq=30, amp=0.3, fringe_dir="vertical")),
        ("horizontal fringe", dict(freq=16, amp=0.3, fringe_dir="horizontal")),
        ("global bow deformation", dict(freq=16, amp=0.32, bow=1.2, fringe_dir="vertical")),
    ]
    grid(items, "fringe distortion: frequency, direction, deformation", "synth_fringe.png")


def main():
    fig_position()
    fig_rounding()
    fig_holder()
    fig_camera_hole()
    fig_background()
    fig_fringe()
    print("saved synthetic variation figures to", OUT_DIR)


if __name__ == "__main__":
    main()
