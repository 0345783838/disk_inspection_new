"""Normalize factory frames for operator review without modifying raw images."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_roi(value):
    if not value:
        return None
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("ROI must be empty or x,y,width,height")
    if min(x, y) < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI coordinates must be non-negative and size must be positive")
    return x, y, width, height


def normalize_frame(image, camera_id, roi=None):
    if image is None or image.size == 0:
        raise ValueError("Cannot normalize an empty image")
    if camera_id == 1:
        normalized = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif camera_id == 2:
        normalized = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("camera_id must be 1 or 2")

    if roi is None:
        return normalized
    x, y, width, height = roi
    if x + width > normalized.shape[1] or y + height > normalized.shape[0]:
        raise ValueError(
            f"ROI {roi} is outside normalized frame "
            f"{normalized.shape[1]}x{normalized.shape[0]}")
    return normalized[y:y + height, x:x + width].copy()


def fit_preview(image, width=480, height=360):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def contact_sheet(samples):
    rows = []
    for camera_id, path, raw, normalized in samples:
        raw_preview = fit_preview(raw)
        normalized_preview = fit_preview(normalized)
        cv2.putText(raw_preview, f"Cam{camera_id} RAW: {path.name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(normalized_preview, f"Cam{camera_id} NORMALIZED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack((raw_preview, normalized_preview)))
    return np.vstack(rows) if rows else None


def process_camera(camera_id, source_dir, output_dir, roi, sample_limit):
    source_dir = Path(source_dir).resolve()
    target_root = Path(output_dir).resolve() / f"Cam{camera_id}"
    paths = sorted(path for path in source_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    samples = []
    records = []
    for path in paths:
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            records.append({"source": str(path), "status": "unreadable"})
            continue
        normalized = normalize_frame(raw, camera_id, roi)
        relative = path.relative_to(source_dir)
        target = (target_root / relative).with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), normalized):
            raise OSError(f"Cannot write {target}")
        records.append({
            "source": str(path),
            "normalized": str(target),
            "raw_shape": list(raw.shape),
            "normalized_shape": list(normalized.shape),
            "status": "ok",
        })
        if len(samples) < sample_limit:
            samples.append((camera_id, path, raw, normalized))
    return records, samples


def main():
    parser = argparse.ArgumentParser(
        description="Rotate Cam1 CW and Cam2 CCW, optionally crop ROI, and render a review sheet.")
    parser.add_argument("--cam1-dir", help="Directory containing Camera 1 raw frames")
    parser.add_argument("--cam2-dir", help="Directory containing Camera 2 raw frames")
    parser.add_argument("--output", required=True, help="New output directory; raw files are never modified")
    parser.add_argument("--cam1-roi", type=parse_roi, help="Cam1 ROI after rotation: x,y,width,height")
    parser.add_argument("--cam2-roi", type=parse_roi, help="Cam2 ROI after rotation: x,y,width,height")
    parser.add_argument("--samples-per-camera", type=int, default=5)
    args = parser.parse_args()
    if not args.cam1_dir and not args.cam2_dir:
        parser.error("at least one of --cam1-dir or --cam2-dir is required")
    if args.samples_per_camera < 1:
        parser.error("--samples-per-camera must be positive")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, samples = [], []
    for camera_id, source, roi in (
            (1, args.cam1_dir, args.cam1_roi), (2, args.cam2_dir, args.cam2_roi)):
        if source:
            camera_records, camera_samples = process_camera(
                camera_id, source, output, roi, args.samples_per_camera)
            records.extend(camera_records)
            samples.extend(camera_samples)

    sheet = contact_sheet(samples)
    if sheet is not None and not cv2.imwrite(str(output / "normalization_contact_sheet.jpg"), sheet):
        raise OSError("Cannot write normalization contact sheet")
    manifest = {
        "preprocessing": {
            "cam1_rotation": "Clockwise90",
            "cam2_rotation": "CounterClockwise90",
            "cam1_roi": args.cam1_roi,
            "cam2_roi": args.cam2_roi,
        },
        "images": records,
    }
    (output / "normalization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Normalized {sum(item['status'] == 'ok' for item in records)} images into {output}")


if __name__ == "__main__":
    main()
