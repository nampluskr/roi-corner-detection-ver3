# data/make_augmented_images.py: generate measured-style grayscale images with LabelMe ROI polygons.

import argparse
import json
import math
import os

import cv2
import numpy as np
from PIL import Image


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE_ROOT = r"E:\fringe_data\training_all"
CATEGORY_SOURCES = [
    ("google", "google"),
    ("oppo", "oppo"),
    ("h8", "H8"),
    ("q8", "Q8"),
]
CATEGORY_SEED_OFFSETS = {
    "google": 100_000,
    "oppo": 200_000,
    "h8": 300_000,
    "q8": 400_000,
}
CATEGORY_ROI_INSETS = {
    "google": 0.035,
    "oppo": 0.105,
    "h8": 0.030,
    "q8": 0.035,
}
CATEGORY_PROFILES = {
    "google": {
        "gain": (0.94, 1.07),
        "bias": (-5.0, 6.0),
        "gamma": (0.92, 1.09),
        "gradient": (2.0, 10.0),
        "noise": (0.6, 2.6),
        "blur": (0.15, 0.75),
        "banding": (0.0, 2.8),
    },
    "oppo": {
        "gain": (0.94, 1.08),
        "bias": (-4.0, 7.0),
        "gamma": (0.91, 1.08),
        "gradient": (2.0, 9.0),
        "noise": (0.7, 2.8),
        "blur": (0.20, 0.85),
        "banding": (0.0, 2.5),
    },
    "h8": {
        "gain": (0.93, 1.07),
        "bias": (-5.0, 6.0),
        "gamma": (0.92, 1.10),
        "gradient": (2.0, 9.0),
        "noise": (0.6, 2.7),
        "blur": (0.15, 0.80),
        "banding": (0.0, 2.5),
    },
    "q8": {
        "gain": (0.94, 1.06),
        "bias": (-6.0, 5.0),
        "gamma": (0.93, 1.08),
        "gradient": (2.0, 10.0),
        "noise": (0.6, 2.6),
        "blur": (0.15, 0.75),
        "banding": (0.0, 2.8),
    },
}


def list_source_images(source_dir):
    """Return the stable TIFF source list for one measured category."""
    names = []
    for name in sorted(os.listdir(source_dir)):
        path = os.path.join(source_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in {".tif", ".tiff"}:
            names.append(path)
    if not names:
        raise FileNotFoundError("no TIFF source images found: " + source_dir)
    return names


def load_grayscale(path):
    """Load one source image as an 8-bit grayscale array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def resize_letterbox_gray(image, width, height):
    """Resize one grayscale image without changing its aspect ratio."""
    source_height, source_width = image.shape
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    edge_samples = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1],
    ])
    fill_value = int(np.percentile(edge_samples, 25))
    canvas = np.full((height, width), fill_value, dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
    return canvas


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


def polygon_area(points):
    """Return the absolute polygon area."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def candidate_quads(mask, width, height):
    """Return plausible quadrilateral candidates from one binary panel mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    frame_area = float(width * height)
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < 0.035 * frame_area:
            continue
        rectangle = cv2.minAreaRect(contour)
        rect_width, rect_height = rectangle[1]
        rect_area = float(rect_width * rect_height)
        if rect_area < 0.07 * frame_area or rect_area > 0.62 * frame_area:
            continue
        short_side = min(rect_width, rect_height)
        long_side = max(rect_width, rect_height)
        if short_side < 0.16 * min(width, height):
            continue
        if long_side / max(short_side, 1e-6) > 2.4:
            continue
        quad = order_quad(cv2.boxPoints(rectangle))
        candidates.append((quad, contour_area, rect_area))
    return candidates


def score_quad(image, directional_energy, quad, contour_area, rect_area):
    """Score one detected panel candidate using geometry and fringe energy."""
    height, width = image.shape
    frame_area = float(width * height)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    inside = mask > 0
    outside = ~inside
    inside_energy = float(np.mean(directional_energy[inside]))
    outside_energy = float(np.mean(directional_energy[outside]))
    energy_ratio = inside_energy / max(outside_energy, 1e-6)
    center = np.mean(quad, axis=0)
    center_distance = math.hypot(
        (float(center[0]) - width * 0.5) / width,
        (float(center[1]) - height * 0.5) / height,
    )
    area_ratio = rect_area / frame_area
    fill_ratio = contour_area / max(rect_area, 1e-6)
    area_preference = 1.0 - min(1.0, abs(area_ratio - 0.24) / 0.24)
    return (
        2.6 * min(energy_ratio, 4.0)
        + 1.3 * area_preference
        + 0.8 * min(fill_ratio, 1.0)
        - 2.2 * center_distance
    )


def shrink_quad(quad, fraction):
    """Move a quadrilateral slightly toward its center."""
    center = np.mean(quad, axis=0)
    return center + (quad - center) * (1.0 - fraction)


def detect_roi_quad(image, inset_fraction):
    """Detect the central fringe panel as a four-point quadrilateral."""
    height, width = image.shape
    smooth = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    gradient_x = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    directional = np.maximum(gradient_y - 0.35 * gradient_x, 0.0)
    directional = cv2.GaussianBlur(directional, (0, 0), sigmaX=7.0, sigmaY=4.0)

    gradient_threshold = float(np.percentile(directional, 76.0))
    gradient_mask = np.where(
        directional >= max(gradient_threshold, 1.0),
        255,
        0,
    ).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (39, 27))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_CLOSE, close_kernel)
    gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_OPEN, open_kernel)

    _, brightness_mask = cv2.threshold(
        smooth,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    brightness_mask = cv2.morphologyEx(
        brightness_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
    )
    brightness_mask = cv2.morphologyEx(
        brightness_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    candidates = candidate_quads(gradient_mask, width, height)
    candidates.extend(candidate_quads(brightness_mask, width, height))
    if not candidates:
        raise RuntimeError("could not detect a plausible fringe panel")

    scored = [
        (
            score_quad(image, directional, quad, contour_area, rect_area),
            quad,
        )
        for quad, contour_area, rect_area in candidates
    ]
    score, quad = max(scored, key=lambda item: item[0])
    if score < 3.0:
        raise RuntimeError("fringe panel confidence is too low: {:.3f}".format(score))
    quad = order_quad(shrink_quad(quad, inset_fraction))
    return quad, score


def sample_homography(width, height, rng):
    """Sample one conservative projective camera variation."""
    source = np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0],
    ], dtype=np.float32)
    center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
    angle = math.radians(float(rng.uniform(-2.2, 2.2)))
    scale = float(rng.uniform(0.975, 1.025))
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=np.float32)
    destination = (source - center) @ rotation.T
    destination *= scale
    destination += center
    destination[:, 0] += float(rng.uniform(-0.018, 0.018)) * width
    destination[:, 1] += float(rng.uniform(-0.018, 0.018)) * height
    perspective = rng.uniform(-1.0, 1.0, (4, 2)).astype(np.float32)
    perspective[:, 0] *= 0.0045 * width
    perspective[:, 1] *= 0.0045 * height
    destination += perspective
    return cv2.getPerspectiveTransform(source, destination.astype(np.float32))


def transform_quad(quad, matrix):
    """Transform one quadrilateral with a homography."""
    points = cv2.perspectiveTransform(
        quad.reshape(1, 4, 2).astype(np.float32),
        matrix,
    )[0]
    return order_quad(points)


def quad_is_valid(quad, width, height):
    """Return whether a transformed ROI remains a valid in-frame polygon."""
    margin = 2.0
    if np.any(quad[:, 0] <= margin) or np.any(quad[:, 0] >= width - margin):
        return False
    if np.any(quad[:, 1] <= margin) or np.any(quad[:, 1] >= height - margin):
        return False
    contour = quad.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False
    area_ratio = polygon_area(quad) / float(width * height)
    return 0.06 <= area_ratio <= 0.62


def apply_geometry(image, quad, rng):
    """Apply conservative camera geometry to an image and its ROI."""
    height, width = image.shape
    for _ in range(50):
        matrix = sample_homography(width, height, rng)
        transformed_quad = transform_quad(quad, matrix)
        if not quad_is_valid(transformed_quad, width, height):
            continue
        edge_values = np.concatenate([
            image[0, :],
            image[-1, :],
            image[:, 0],
            image[:, -1],
        ])
        border_value = float(np.percentile(edge_values, 25))
        transformed = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return transformed, transformed_quad
    raise RuntimeError("could not sample an in-frame geometric variation")


def apply_photometric_variation(image, profile, rng):
    """Apply measured-style illumination, fringe and sensor variation."""
    height, width = image.shape
    values = image.astype(np.float32)
    normalized = np.clip(values / 255.0, 0.0, 1.0)
    gamma = float(rng.uniform(*profile["gamma"]))
    values = 255.0 * np.power(normalized, gamma)
    values = values * float(rng.uniform(*profile["gain"]))
    values += float(rng.uniform(*profile["bias"]))

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    xn = x / max(1, width - 1) - 0.5
    yn = y / max(1, height - 1) - 0.5
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    gradient = xn * math.cos(angle) + yn * math.sin(angle)
    values += gradient * float(rng.uniform(*profile["gradient"]))

    banding_amplitude = float(rng.uniform(*profile["banding"]))
    banding_frequency = float(rng.uniform(7.0, 28.0))
    banding_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    banding = np.sin(2.0 * math.pi * banding_frequency * y / height + banding_phase)
    values += banding_amplitude * banding

    sigma = float(rng.uniform(*profile["blur"]))
    if sigma >= 0.25:
        values = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma)
    noise_sigma = float(rng.uniform(*profile["noise"]))
    values += rng.normal(0.0, noise_sigma, values.shape).astype(np.float32)
    return np.clip(values, 0, 255).astype(np.uint8)


def labelme_payload(image_name, width, height, quad):
    """Build one LabelMe polygon annotation."""
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


def prepare_output_dir(output_dir):
    """Create one empty output directory without overwriting data."""
    if os.path.exists(output_dir):
        if os.path.isdir(output_dir) and not os.listdir(output_dir):
            return
        raise FileExistsError(
            "output directory already exists and is not empty: " + output_dir
        )
    os.makedirs(output_dir)


def save_pair(output_dir, category, index, image, quad):
    """Save one grayscale TIFF and matching LabelMe JSON."""
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
    """Validate one generated image and LabelMe annotation."""
    with Image.open(image_path) as image:
        if image.size != (width, height):
            raise RuntimeError("invalid image dimensions: " + image_path)
        if image.mode != "L":
            raise RuntimeError("generated image is not grayscale: " + image_path)
    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload["imagePath"] != os.path.basename(image_path):
        raise RuntimeError("LabelMe imagePath mismatch: " + json_path)
    if payload["imageWidth"] != width or payload["imageHeight"] != height:
        raise RuntimeError("LabelMe dimensions mismatch: " + json_path)
    if len(payload["shapes"]) != 1:
        raise RuntimeError("LabelMe file must contain one shape: " + json_path)
    shape = payload["shapes"][0]
    if shape["label"] != "roi" or shape["shape_type"] != "polygon":
        raise RuntimeError("invalid LabelMe ROI shape: " + json_path)
    quad = np.asarray(shape["points"], dtype=np.float32)
    if quad.shape != (4, 2):
        raise RuntimeError("ROI must contain four points: " + json_path)
    if not quad_is_valid(quad, width, height):
        raise RuntimeError("invalid ROI geometry: " + json_path)


def validate_output(output_dir, category, count, width, height):
    """Validate the exact generated file set for one category."""
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


def generate_category(category, source_name, args):
    """Generate one measured-style category dataset."""
    source_dir = os.path.join(args.source_root, source_name, "images")
    source_paths = list_source_images(source_dir)
    output_dir = os.path.join(
        args.output_root,
        "{}_{}".format(category, args.count),
    )
    prepare_output_dir(output_dir)
    print("category=" + category)
    print("sources={}".format(len(source_paths)))
    print("output=" + output_dir)
    profile = CATEGORY_PROFILES[category]
    category_seed = args.seed + CATEGORY_SEED_OFFSETS[category]

    source_cache = {}
    roi_cache = {}
    for index in range(1, args.count + 1):
        source_path = source_paths[(index - 1) % len(source_paths)]
        if source_path not in source_cache:
            source = load_grayscale(source_path)
            source_cache[source_path] = resize_letterbox_gray(
                source,
                args.width,
                args.height,
            )
            quad, confidence = detect_roi_quad(
                source_cache[source_path],
                CATEGORY_ROI_INSETS[category],
            )
            roi_cache[source_path] = (quad, confidence)
        base_image = source_cache[source_path]
        base_quad, confidence = roi_cache[source_path]
        rng = np.random.default_rng(category_seed + index * 1009)
        image, quad = apply_geometry(base_image, base_quad, rng)
        image = apply_photometric_variation(image, profile, rng)
        image_path, json_path = save_pair(
            output_dir,
            category,
            index,
            image,
            quad,
        )
        validate_pair(image_path, json_path, args.width, args.height)
        print(
            "{:04d} source={} confidence={:.3f} area={:.3f}".format(
                index,
                os.path.basename(source_path),
                confidence,
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--output-root",
        default=os.path.join(DATA_DIR, "augmented"),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[category for category, _ in CATEGORY_SOURCES],
        default=[category for category, _ in CATEGORY_SOURCES],
    )
    return parser.parse_args()


def validate_args(args):
    """Validate generator arguments and source paths."""
    if args.count < 1 or args.count > 9999:
        raise ValueError("count must be between 1 and 9999")
    if args.width < 320 or args.height < 320:
        raise ValueError("width and height must be at least 320")
    if not os.path.isdir(args.source_root):
        raise FileNotFoundError("source root does not exist: " + args.source_root)


def main():
    """Generate measured-style preview datasets for selected categories."""
    args = parse_args()
    validate_args(args)
    selected = set(args.categories)
    for category, source_name in CATEGORY_SOURCES:
        if category in selected:
            generate_category(category, source_name, args)


if __name__ == "__main__":
    main()
