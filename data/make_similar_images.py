# data/make_similar_images.py: procedurally generate measured-style fringe panel images.

import argparse
import json
import math
import os

import cv2
import numpy as np
from PIL import Image


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORY_ORDER = ["google", "h8", "q8", "oppo"]
CATEGORY_SEED_OFFSETS = {
    "google": 100_000,
    "h8": 200_000,
    "q8": 300_000,
    "oppo": 400_000,
}
CATEGORY_PROFILES = {
    "google": {
        "height_fraction": (0.70, 0.79),
        "aspect": (0.80, 0.92),
        "corner_radius": (0.060, 0.095),
        "fringe_cycles": (54.0, 68.0),
        "fringe_contrast": (72.0, 92.0),
        "panel_mean": (98.0, 118.0),
        "distortion_count": (2, 4),
        "distortion_amplitude": (7.0, 20.0),
        "rotation": (-3.0, 3.0),
        "top_scale": (0.95, 1.00),
        "ribbon_sides": (1, 2),
        "clamp_count": (5, 8),
    },
    "h8": {
        "height_fraction": (0.72, 0.82),
        "aspect": (0.70, 0.83),
        "corner_radius": (0.012, 0.035),
        "fringe_cycles": (58.0, 74.0),
        "fringe_contrast": (68.0, 90.0),
        "panel_mean": (88.0, 110.0),
        "distortion_count": (2, 4),
        "distortion_amplitude": (8.0, 24.0),
        "rotation": (-2.5, 2.5),
        "top_scale": (0.96, 1.00),
        "ribbon_sides": (1, 3),
        "clamp_count": (6, 10),
    },
    "q8": {
        "height_fraction": (0.68, 0.77),
        "aspect": (0.78, 0.90),
        "corner_radius": (0.018, 0.050),
        "fringe_cycles": (55.0, 70.0),
        "fringe_contrast": (75.0, 96.0),
        "panel_mean": (102.0, 124.0),
        "distortion_count": (1, 3),
        "distortion_amplitude": (6.0, 18.0),
        "rotation": (-2.5, 2.5),
        "top_scale": (0.96, 1.00),
        "ribbon_sides": (1, 2),
        "clamp_count": (5, 8),
    },
    "oppo": {
        "height_fraction": (0.69, 0.78),
        "aspect": (0.64, 0.76),
        "corner_radius": (0.105, 0.155),
        "fringe_cycles": (51.0, 65.0),
        "fringe_contrast": (67.0, 88.0),
        "panel_mean": (88.0, 108.0),
        "distortion_count": (2, 4),
        "distortion_amplitude": (9.0, 25.0),
        "rotation": (-3.0, 3.0),
        "top_scale": (0.94, 0.99),
        "ribbon_sides": (1, 3),
        "clamp_count": (5, 8),
    },
}


def multiscale_noise(width, height, rng):
    """Create smooth low-frequency grayscale variation."""
    result = np.zeros((height, width), dtype=np.float32)
    for grid, weight in [(18, 0.58), (55, 0.28), (150, 0.14)]:
        small_width = max(2, int(math.ceil(width / grid)))
        small_height = max(2, int(math.ceil(height / grid)))
        small = rng.normal(
            0.0,
            1.0,
            (small_height, small_width),
        ).astype(np.float32)
        layer = cv2.resize(
            small,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        result += weight * layer
    result -= float(np.mean(result))
    deviation = float(np.std(result))
    if deviation > 1e-6:
        result /= deviation
    return result


def add_perforated_plate(stage, rng):
    """Render a dark perforated metal stage without unique source objects."""
    height, width = stage.shape
    canvas = stage.copy()
    spacing = int(rng.integers(25, 34))
    radius = max(4, int(round(spacing * rng.uniform(0.25, 0.31))))
    offset_x = int(rng.integers(-spacing, spacing))
    offset_y = int(rng.integers(-spacing, spacing))
    plate_value = float(rng.uniform(18.0, 34.0))
    hole_value = float(rng.uniform(2.0, 9.0))
    rim_value = float(rng.uniform(28.0, 48.0))

    plate_mask = np.zeros((height, width), dtype=np.uint8)
    margin_x = int(width * rng.uniform(0.03, 0.08))
    cv2.rectangle(
        plate_mask,
        (margin_x, 0),
        (width - margin_x - 1, height - 1),
        255,
        -1,
    )
    if rng.random() < 0.7:
        cut_width = int(width * rng.uniform(0.09, 0.17))
        cut_side = int(rng.integers(0, 2))
        if cut_side == 0:
            plate_mask[:, :cut_width] = 0
        else:
            plate_mask[:, width - cut_width:] = 0
    canvas = np.where(
        plate_mask > 0,
        0.72 * canvas + 0.28 * plate_value,
        canvas,
    )

    for y in range(offset_y, height + spacing, spacing):
        for x in range(offset_x, width + spacing, spacing):
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if plate_mask[y, x] == 0:
                continue
            local_radius = max(3, radius + int(rng.integers(-1, 2)))
            cv2.circle(
                canvas,
                (x + 1, y + 2),
                local_radius + 2,
                rim_value,
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                (x, y),
                local_radius,
                hole_value,
                -1,
                cv2.LINE_AA,
            )

    seam_count = int(rng.integers(2, 5))
    for _ in range(seam_count):
        if rng.random() < 0.5:
            x = int(rng.uniform(0.08, 0.92) * width)
            cv2.line(
                canvas,
                (x, 0),
                (x, height - 1),
                float(rng.uniform(4.0, 18.0)),
                int(rng.integers(3, 9)),
                cv2.LINE_AA,
            )
        else:
            y = int(rng.uniform(0.08, 0.92) * height)
            cv2.line(
                canvas,
                (0, y),
                (width - 1, y),
                float(rng.uniform(4.0, 18.0)),
                int(rng.integers(3, 9)),
                cv2.LINE_AA,
            )
    return canvas


def add_stage_hardware(stage, rng):
    """Add generic rails, blocks and screw heads behind the panel."""
    height, width = stage.shape
    canvas = stage.copy()
    block_count = int(rng.integers(5, 10))
    for _ in range(block_count):
        block_width = int(rng.uniform(0.025, 0.075) * width)
        block_height = int(rng.uniform(0.035, 0.12) * height)
        x0 = int(rng.uniform(0.02, 0.98) * width - block_width * 0.5)
        y0 = int(rng.uniform(0.02, 0.98) * height - block_height * 0.5)
        x0 = int(np.clip(x0, 0, width - block_width - 1))
        y0 = int(np.clip(y0, 0, height - block_height - 1))
        x1 = x0 + block_width
        y1 = y0 + block_height
        value = float(rng.uniform(12.0, 50.0))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), value, -1)
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x1, y1),
            min(80.0, value + 14.0),
            2,
            cv2.LINE_AA,
        )
        if block_width > 24 and block_height > 24:
            screw_count = int(rng.integers(1, 3))
            for screw_index in range(screw_count):
                fraction = (screw_index + 1.0) / (screw_count + 1.0)
                center = (
                    int(round(x0 + fraction * block_width)),
                    int(round(y0 + 0.52 * block_height)),
                )
                radius = max(3, int(round(min(block_width, block_height) * 0.08)))
                cv2.circle(
                    canvas,
                    center,
                    radius + 2,
                    max(2.0, value - 10.0),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    canvas,
                    center,
                    radius,
                    min(110.0, value + 26.0),
                    1,
                    cv2.LINE_AA,
                )
    return canvas


def create_stage(width, height, rng):
    """Create a dark measured-style stage background."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = x / max(1, width - 1)
    yn = y / max(1, height - 1)
    base = float(rng.uniform(6.0, 17.0))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    gradient = (
        (xn - 0.5) * math.cos(angle)
        + (yn - 0.5) * math.sin(angle)
    )
    stage = base + float(rng.uniform(4.0, 14.0)) * gradient
    stage += float(rng.uniform(1.0, 3.2)) * multiscale_noise(width, height, rng)
    center_x = float(rng.uniform(0.30, 0.70))
    center_y = float(rng.uniform(0.30, 0.70))
    radius = np.sqrt((xn - center_x) ** 2 + (yn - center_y) ** 2)
    stage -= float(rng.uniform(5.0, 16.0)) * radius
    stage = np.clip(stage, 0.0, 70.0)
    stage = add_perforated_plate(stage, rng)
    stage = add_stage_hardware(stage, rng)
    stage += rng.normal(0.0, rng.uniform(0.5, 1.8), stage.shape).astype(np.float32)
    return np.clip(stage, 0.0, 255.0)


def rounded_mask(width, height, radius):
    """Create a soft rounded-rectangle mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(np.clip(radius, 1, min(width, height) // 2))
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
    centers = [
        (radius, radius),
        (width - radius - 1, radius),
        (width - radius - 1, height - radius - 1),
        (radius, height - radius - 1),
    ]
    for center in centers:
        cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=0.8).astype(np.float32) / 255.0


def create_fringe_panel(panel_width, panel_height, profile, rng):
    """Create a rounded grayscale panel with locally distorted horizontal fringe."""
    y, x = np.mgrid[0:panel_height, 0:panel_width].astype(np.float32)
    xn = x / max(1, panel_width - 1)
    yn = y / max(1, panel_height - 1)
    cycles = float(rng.uniform(*profile["fringe_cycles"]))
    displacement = np.zeros((panel_height, panel_width), dtype=np.float32)

    broad_bend = float(rng.uniform(-0.008, 0.008))
    displacement += broad_bend * (xn - 0.5) ** 2
    distortion_count = int(
        rng.integers(
            profile["distortion_count"][0],
            profile["distortion_count"][1] + 1,
        )
    )
    for _ in range(distortion_count):
        center_x = float(rng.uniform(0.15, 0.85))
        center_y = float(rng.uniform(0.12, 0.88))
        sigma_x = float(rng.uniform(0.10, 0.32))
        sigma_y = float(rng.uniform(0.025, 0.095))
        amplitude_pixels = float(rng.uniform(*profile["distortion_amplitude"]))
        amplitude = amplitude_pixels / max(1, panel_height)
        spatial_frequency = float(rng.uniform(0.45, 1.50))
        phase_offset = float(rng.uniform(0.0, 2.0 * math.pi))
        envelope = np.exp(
            -0.5 * ((xn - center_x) / sigma_x) ** 2
            -0.5 * ((yn - center_y) / sigma_y) ** 2
        )
        displacement += (
            amplitude
            * envelope
            * np.sin(
                2.0 * math.pi * spatial_frequency * (xn - center_x)
                + phase_offset
            )
        )

    edge_roll = float(rng.uniform(-0.005, 0.005))
    displacement += edge_roll * np.sin(math.pi * yn) * (xn - 0.5)
    phase = 2.0 * math.pi * cycles * (yn + displacement)
    sharpness = float(rng.uniform(1.15, 1.75))
    fringe = np.tanh(sharpness * np.sin(phase))

    panel_mean = float(rng.uniform(*profile["panel_mean"]))
    contrast = float(rng.uniform(*profile["fringe_contrast"]))
    panel = panel_mean + contrast * fringe
    illumination = (
        float(rng.uniform(-15.0, 15.0)) * (xn - 0.5)
        + float(rng.uniform(-10.0, 10.0)) * (yn - 0.5)
    )
    panel += illumination
    panel += float(rng.uniform(1.5, 4.0)) * multiscale_noise(
        panel_width,
        panel_height,
        rng,
    )
    panel += rng.normal(
        0.0,
        rng.uniform(0.8, 2.6),
        panel.shape,
    ).astype(np.float32)

    radius_fraction = float(rng.uniform(*profile["corner_radius"]))
    radius = int(round(radius_fraction * min(panel_width, panel_height)))
    outer_mask = rounded_mask(panel_width, panel_height, radius)
    short_side = min(panel_width, panel_height)
    bezel = max(6, int(round(short_side * rng.uniform(0.014, 0.026))))
    inner_width = panel_width - 2 * bezel
    inner_height = panel_height - 2 * bezel
    inner_radius = max(2, radius - bezel)
    active_mask = np.zeros_like(outer_mask)
    active_mask[
        bezel:bezel + inner_height,
        bezel:bezel + inner_width,
    ] = rounded_mask(inner_width, inner_height, inner_radius)
    bezel_value = float(rng.uniform(4.0, 16.0))
    bezel_texture = (
        bezel_value
        + float(rng.uniform(0.4, 1.8))
        * multiscale_noise(panel_width, panel_height, rng)
    )
    panel = bezel_texture * (1.0 - active_mask) + panel * active_mask

    edge_distance = cv2.distanceTransform(
        np.where(active_mask > 0.5, 255, 0).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    edge_highlight = np.exp(-edge_distance / max(2.0, panel_width * 0.009))
    panel += (
        float(rng.uniform(4.0, 12.0))
        * edge_highlight
        * active_mask
    )

    marker_count = int(rng.integers(1, 4))
    for marker_index in range(marker_count):
        marker_x = int(
            panel_width
            * (
                rng.uniform(0.07, 0.16)
                if marker_index % 2 == 0
                else rng.uniform(0.84, 0.93)
            )
        )
        marker_y = int(panel_height * rng.uniform(0.06, 0.94))
        marker_radius = max(3, int(round(min(panel_width, panel_height) * 0.008)))
        cv2.circle(
            panel,
            (marker_x, marker_y),
            marker_radius + 2,
            float(rng.uniform(180.0, 235.0)),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            panel,
            (marker_x, marker_y),
            marker_radius,
            float(rng.uniform(15.0, 55.0)),
            -1,
            cv2.LINE_AA,
        )
    outer_edge = cv2.Canny(
        np.where(outer_mask > 0.5, 255, 0).astype(np.uint8),
        80,
        160,
    ).astype(np.float32) / 255.0
    panel += outer_edge * float(rng.uniform(10.0, 24.0))
    panel *= outer_mask
    return np.clip(panel, 0.0, 255.0), outer_mask, cycles


def order_quad(points):
    """Order quadrilateral points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(np.sum(ordered, axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    edge_a = ordered[1] - ordered[0]
    edge_b = ordered[2] - ordered[1]
    cross = float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
    if cross < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def sample_panel_quad(width, height, profile, rng):
    """Sample one centered measured-style panel quadrilateral."""
    panel_height = float(rng.uniform(*profile["height_fraction"])) * height
    aspect = float(rng.uniform(*profile["aspect"]))
    panel_width = panel_height * aspect
    panel_width = min(panel_width, width * 0.46)
    center_x = width * (0.52 + float(rng.uniform(-0.035, 0.035)))
    center_y = height * (0.51 + float(rng.uniform(-0.025, 0.025)))
    top_scale = float(rng.uniform(*profile["top_scale"]))
    top_offset = float(rng.uniform(-0.018, 0.018)) * panel_width
    local = np.array([
        [-0.5 * panel_width * top_scale + top_offset, -0.5 * panel_height],
        [0.5 * panel_width * top_scale + top_offset, -0.5 * panel_height],
        [0.5 * panel_width, 0.5 * panel_height],
        [-0.5 * panel_width, 0.5 * panel_height],
    ], dtype=np.float32)
    angle = math.radians(float(rng.uniform(*profile["rotation"])))
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=np.float32)
    quad = local @ rotation.T
    quad[:, 0] += center_x
    quad[:, 1] += center_y
    jitter = rng.uniform(-1.0, 1.0, (4, 2)).astype(np.float32)
    jitter[:, 0] *= 0.004 * panel_width
    jitter[:, 1] *= 0.004 * panel_height
    quad += jitter
    quad = order_quad(quad)
    if np.any(quad[:, 0] < 8.0) or np.any(quad[:, 0] > width - 8.0):
        raise RuntimeError("sampled panel is outside the horizontal frame")
    if np.any(quad[:, 1] < 8.0) or np.any(quad[:, 1] > height - 8.0):
        raise RuntimeError("sampled panel is outside the vertical frame")
    canonical_width = max(320, int(round(panel_width)))
    canonical_height = max(420, int(round(panel_height)))
    return quad, canonical_width, canonical_height


def edge_geometry(quad, edge_index):
    """Return one panel edge, its tangent and outward normal."""
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    start_index, end_index = edge_pairs[edge_index]
    start = quad[start_index].astype(np.float64)
    end = quad[end_index].astype(np.float64)
    edge = end - start
    length = float(np.linalg.norm(edge))
    tangent = edge / max(length, 1e-6)
    midpoint = 0.5 * (start + end)
    center = np.mean(quad, axis=0).astype(np.float64)
    outward = midpoint - center
    outward /= max(float(np.linalg.norm(outward)), 1e-6)
    return start, end, tangent, outward, length


def fill_striped_polygon(canvas, polygon, rng):
    """Fill one fringe ribbon polygon with measured horizontal stripes."""
    height, width = canvas.shape
    polygon_int = np.round(polygon).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon_int, 255, cv2.LINE_AA)
    y0 = max(0, int(np.min(polygon[:, 1])) - 2)
    y1 = min(height, int(np.max(polygon[:, 1])) + 3)
    x0 = max(0, int(np.min(polygon[:, 0])) - 2)
    x1 = min(width, int(np.max(polygon[:, 0])) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    yy = np.arange(y0, y1, dtype=np.float32).reshape(-1, 1)
    frequency = float(rng.uniform(0.12, 0.19))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    values = (
        float(rng.uniform(90.0, 120.0))
        + float(rng.uniform(55.0, 85.0))
        * np.tanh(1.35 * np.sin(2.0 * math.pi * frequency * yy + phase))
    )
    values = np.repeat(values, x1 - x0, axis=1)
    alpha = mask[y0:y1, x0:x1].astype(np.float32) / 255.0
    crop = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = crop * (1.0 - alpha) + values * alpha
    cv2.polylines(
        canvas,
        [polygon_int],
        True,
        float(rng.uniform(105.0, 175.0)),
        1,
        cv2.LINE_AA,
    )


def add_ribbon_tabs(canvas, quad, profile, rng):
    """Add generic fringe extensions around the panel perimeter."""
    side_count = int(
        rng.integers(
            profile["ribbon_sides"][0],
            profile["ribbon_sides"][1] + 1,
        )
    )
    edge_indices = list(rng.choice(4, size=min(4, side_count + 1), replace=False))
    for edge_index in edge_indices:
        start, _, tangent, outward, length = edge_geometry(quad, edge_index)
        tab_count = int(rng.integers(1, side_count + 1))
        positions = np.linspace(0.22, 0.78, tab_count)
        positions += rng.uniform(-0.055, 0.055, tab_count)
        for position in positions:
            tab_length = length * float(rng.uniform(0.08, 0.19))
            depth = min(canvas.shape) * float(rng.uniform(0.025, 0.075))
            contact = start + float(position) * (
                edge_geometry(quad, edge_index)[1] - start
            )
            inner0 = contact - tangent * tab_length * 0.5
            inner1 = contact + tangent * tab_length * 0.5
            outer_center = contact + outward * depth
            outer0 = outer_center - tangent * tab_length * rng.uniform(0.42, 0.58)
            outer1 = outer_center + tangent * tab_length * rng.uniform(0.42, 0.58)
            polygon = np.array(
                [inner0, inner1, outer1, outer0],
                dtype=np.float32,
            )
            fill_striped_polygon(canvas, polygon, rng)


def add_clamps(canvas, quad, profile, rng):
    """Add generic metallic clamp blocks around the panel."""
    height, width = canvas.shape
    clamp_count = int(
        rng.integers(
            profile["clamp_count"][0],
            profile["clamp_count"][1] + 1,
        )
    )
    for clamp_index in range(clamp_count):
        edge_index = clamp_index % 4
        start, end, tangent, outward, length = edge_geometry(quad, edge_index)
        position = float(rng.uniform(0.16, 0.84))
        contact = start + position * (end - start)
        clamp_length = length * float(rng.uniform(0.055, 0.13))
        depth = min(width, height) * float(rng.uniform(0.025, 0.065))
        inner0 = contact - tangent * clamp_length * 0.5
        inner1 = contact + tangent * clamp_length * 0.5
        outer_center = contact + outward * depth
        taper = float(rng.uniform(0.72, 1.02))
        outer0 = outer_center - tangent * clamp_length * taper * 0.5
        outer1 = outer_center + tangent * clamp_length * taper * 0.5
        polygon = np.array([inner0, inner1, outer1, outer0], dtype=np.float32)
        polygon_int = np.round(polygon).astype(np.int32)

        shadow = polygon_int + np.array([4, 6], dtype=np.int32)
        shadow_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(shadow_mask, shadow, 255, cv2.LINE_AA)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=5.0)
        canvas *= 1.0 - 0.24 * shadow_mask.astype(np.float32) / 255.0

        value = float(rng.uniform(18.0, 82.0))
        cv2.fillConvexPoly(canvas, polygon_int, value, cv2.LINE_AA)
        cv2.polylines(
            canvas,
            [polygon_int],
            True,
            min(120.0, value + 24.0),
            2,
            cv2.LINE_AA,
        )
        if rng.random() < 0.72:
            screw_center = contact + outward * depth * float(rng.uniform(0.45, 0.72))
            screw_radius = max(2, int(round(depth * 0.10)))
            screw_point = tuple(np.round(screw_center).astype(int))
            cv2.circle(
                canvas,
                screw_point,
                screw_radius + 2,
                max(4.0, value - 12.0),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                screw_point,
                screw_radius,
                min(150.0, value + 42.0),
                1,
                cv2.LINE_AA,
            )


def warp_panel(stage, panel, panel_mask, quad):
    """Warp one rounded panel into the stage and add a contact shadow."""
    height, width = stage.shape
    panel_height, panel_width = panel.shape
    source = np.array([
        [0.0, 0.0],
        [panel_width - 1.0, 0.0],
        [panel_width - 1.0, panel_height - 1.0],
        [0.0, panel_height - 1.0],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    warped_panel = cv2.warpPerspective(
        panel,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = cv2.warpPerspective(
        panel_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shadow_matrix = matrix.copy()
    shadow_matrix[0, 2] += 7.0
    shadow_matrix[1, 2] += 10.0
    shadow_mask = cv2.warpPerspective(
        panel_mask,
        shadow_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=9.0)
    composed = stage * (1.0 - 0.36 * shadow_mask)
    alpha = np.clip(warped_mask, 0.0, 1.0)
    composed = composed * (1.0 - alpha) + warped_panel * alpha
    return composed


def add_camera_response(image, rng):
    """Apply restrained optical blur, banding and grayscale sensor noise."""
    height, width = image.shape
    sigma = float(rng.uniform(0.30, 0.95))
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    y = np.arange(height, dtype=np.float32).reshape(-1, 1)
    banding = (
        float(rng.uniform(0.0, 1.8))
        * np.sin(
            2.0
            * math.pi
            * float(rng.uniform(2.0, 8.0))
            * y
            / height
            + float(rng.uniform(0.0, 2.0 * math.pi))
        )
    )
    image += banding
    image += rng.normal(
        0.0,
        rng.uniform(0.7, 2.8),
        image.shape,
    ).astype(np.float32)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def polygon_area(points):
    """Return the absolute area of one polygon."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def labelme_payload(image_name, width, height, quad):
    """Build one LabelMe four-point ROI annotation."""
    points = [
        [round(float(point[0]), 6), round(float(point[1]), 6)]
        for point in quad
    ]
    return {
        "version": "6.2.0",
        "flags": {},
        "shapes": [
            {
                "label": "roi",
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def quad_is_valid(quad, width, height):
    """Return whether one ROI is convex and fully inside the image."""
    margin = 2.0
    if np.any(quad[:, 0] <= margin) or np.any(quad[:, 0] >= width - margin):
        return False
    if np.any(quad[:, 1] <= margin) or np.any(quad[:, 1] >= height - margin):
        return False
    contour = quad.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False
    area_ratio = polygon_area(quad) / float(width * height)
    return 0.14 <= area_ratio <= 0.40


def save_pair(output_dir, category, index, image, quad):
    """Save one TIFF image and its LabelMe JSON."""
    stem = "{}_{:04d}".format(category, index)
    image_name = stem + ".tif"
    image_path = os.path.join(output_dir, image_name)
    json_path = os.path.join(output_dir, stem + ".json")
    Image.fromarray(image, mode="L").save(
        image_path,
        format="TIFF",
        compression="tiff_lzw",
    )
    payload = labelme_payload(image_name, image.shape[1], image.shape[0], quad)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return image_path, json_path


def validate_pair(image_path, json_path, width, height):
    """Validate one generated TIFF and LabelMe JSON pair."""
    with Image.open(image_path) as image:
        if image.size != (width, height):
            raise RuntimeError("invalid image size: " + image_path)
        if image.mode != "L":
            raise RuntimeError("image is not grayscale: " + image_path)
    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload["imagePath"] != os.path.basename(image_path):
        raise RuntimeError("LabelMe imagePath mismatch: " + json_path)
    if payload["imageWidth"] != width or payload["imageHeight"] != height:
        raise RuntimeError("LabelMe image dimensions mismatch: " + json_path)
    if len(payload["shapes"]) != 1:
        raise RuntimeError("LabelMe file must contain one shape: " + json_path)
    shape = payload["shapes"][0]
    if shape["label"] != "roi" or shape["shape_type"] != "polygon":
        raise RuntimeError("invalid LabelMe ROI shape: " + json_path)
    quad = np.asarray(shape["points"], dtype=np.float32)
    if quad.shape != (4, 2) or not quad_is_valid(quad, width, height):
        raise RuntimeError("invalid LabelMe ROI geometry: " + json_path)


def prepare_output_dir(output_dir):
    """Create one empty output directory without overwriting data."""
    if os.path.exists(output_dir):
        if os.path.isdir(output_dir) and not os.listdir(output_dir):
            return
        raise FileExistsError(
            "output directory already exists and is not empty: " + output_dir
        )
    os.makedirs(output_dir)


def validate_output(output_dir, category, count, width, height):
    """Validate the exact output file set for one category."""
    expected = []
    for index in range(1, count + 1):
        stem = "{}_{:04d}".format(category, index)
        expected.extend([stem + ".json", stem + ".tif"])
    names = sorted(os.listdir(output_dir))
    if names != sorted(expected):
        raise RuntimeError("output directory contains an unexpected file set")
    for index in range(1, count + 1):
        stem = "{}_{:04d}".format(category, index)
        validate_pair(
            os.path.join(output_dir, stem + ".tif"),
            os.path.join(output_dir, stem + ".json"),
            width,
            height,
        )


def generate_sample(category, index, args):
    """Generate one source-independent procedural sample."""
    profile = CATEGORY_PROFILES[category]
    seed = args.seed + CATEGORY_SEED_OFFSETS[category] + index * 1009
    rng = np.random.default_rng(seed)
    stage = create_stage(args.width, args.height, rng)
    quad, panel_width, panel_height = sample_panel_quad(
        args.width,
        args.height,
        profile,
        rng,
    )
    add_ribbon_tabs(stage, quad, profile, rng)
    add_clamps(stage, quad, profile, rng)
    panel, panel_mask, cycles = create_fringe_panel(
        panel_width,
        panel_height,
        profile,
        rng,
    )
    composed = warp_panel(stage, panel, panel_mask, quad)
    image = add_camera_response(composed, rng)
    return image, quad, cycles


def generate_category(category, args):
    """Generate and validate one category dataset."""
    output_dir = os.path.join(
        args.output_root,
        "{}_{}".format(category, args.count),
    )
    prepare_output_dir(output_dir)
    print("category=" + category)
    print("output=" + output_dir)
    for index in range(1, args.count + 1):
        image, quad, cycles = generate_sample(category, index, args)
        image_path, json_path = save_pair(
            output_dir,
            category,
            index,
            image,
            quad,
        )
        validate_pair(image_path, json_path, args.width, args.height)
        print(
            "{:04d} cycles={:.1f} area={:.3f}".format(
                index,
                cycles,
                polygon_area(quad) / float(args.width * args.height),
            )
        )
    validate_output(
        output_dir,
        category,
        args.count,
        args.width,
        args.height,
    )
    print("validated_pairs={}".format(args.count))


def create_preview(args):
    """Create a compact overlay preview for the generated categories."""
    selected = [category for category in CATEGORY_ORDER if category in args.categories]
    sample_count = min(args.count, 10)
    columns = 5
    rows_per_category = int(math.ceil(sample_count / columns))
    thumb_width = 320
    thumb_height = 180
    header_height = 26
    preview = np.full(
        (
            len(selected) * rows_per_category * (thumb_height + header_height),
            columns * thumb_width,
            3,
        ),
        22,
        dtype=np.uint8,
    )
    for category_index, category in enumerate(selected):
        output_dir = os.path.join(
            args.output_root,
            "{}_{}".format(category, args.count),
        )
        for sample_index in range(1, sample_count + 1):
            stem = "{}_{:04d}".format(category, sample_index)
            image_path = os.path.join(output_dir, stem + ".tif")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            thumb = cv2.resize(
                image,
                (thumb_width, thumb_height),
                interpolation=cv2.INTER_AREA,
            )
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            row = category_index * rows_per_category + (sample_index - 1) // columns
            column = (sample_index - 1) % columns
            y0 = row * (thumb_height + header_height)
            x0 = column * thumb_width
            cv2.putText(
                preview,
                "{} {}".format(category, sample_index),
                (x0 + 8, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            preview[
                y0 + header_height:y0 + header_height + thumb_height,
                x0:x0 + thumb_width,
            ] = thumb
    preview_path = os.path.join(
        args.output_root,
        "preview_{}.png".format(args.count),
    )
    cv2.imwrite(preview_path, preview)
    print("preview=" + preview_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        default=os.path.join(DATA_DIR, "similar"),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORY_ORDER,
        default=CATEGORY_ORDER,
    )
    return parser.parse_args()


def validate_args(args):
    """Validate generator arguments."""
    if args.count < 1 or args.count > 9999:
        raise ValueError("count must be between 1 and 9999")
    if args.width < 320 or args.height < 320:
        raise ValueError("width and height must be at least 320")
    if not args.categories:
        raise ValueError("at least one category is required")


def main():
    """Generate selected source-independent similar image datasets."""
    args = parse_args()
    validate_args(args)
    for category in CATEGORY_ORDER:
        if category in args.categories:
            generate_category(category, args)
    create_preview(args)


if __name__ == "__main__":
    main()
