"""Build a reviewed, mixed old/factory point detector dataset.

Factory labels use manually selected 25-slot row geometry on reference frames,
then image registration to transfer it to the other frames of the same camera.
The old detector's predictions are never copied into training labels.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


# Canonical coordinates after Cam1 clockwise / Cam2 counter-clockwise rotation.
# The first/last centers and row slopes were inspected on the factory White frames.
ROW_GEOMETRY = {
    "Cam1": (
        (206.0, 1705.0, 218.0, 198.0),
        (214.0, 1718.0, 1136.0, 1122.0),
        (232.0, 1718.0, 2043.0, 2047.0),
    ),
    "Cam2": (
        (190.0, 1660.0, 59.0, 64.0),
        (205.0, 1682.0, 972.0, 956.0),
        (212.0, 1706.0, 1901.0, 1861.0),
    ),
}
BOX_SIZE = (60.0, 100.0)
VAL_INDICES = {2, 7}
FACTORY_CROP_BOTTOM = 2200  # canonical pixels; excludes the other camera's overlap row


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def rotate(image, camera):
    code = cv2.ROTATE_90_CLOCKWISE if camera == "Cam1" else cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(image, code)


def crop_owned_rows(image):
    if image.shape[:2] != (2592, 1944):
        raise ValueError(f"Unexpected canonical factory frame size: {image.shape}")
    return image[:FACTORY_CROP_BOTTOM].copy()


def registration_shift(reference, image):
    if reference.shape != image.shape:
        raise ValueError("Factory frames within one camera have different shapes")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = (reference.shape[1] // 4, reference.shape[0] // 4)
    a = cv2.resize(reference_gray, size).astype(np.float32)
    b = cv2.resize(image_gray, size).astype(np.float32)
    (shift_x, shift_y), response = cv2.phaseCorrelate(a, b)
    shift_x *= 4
    shift_y *= 4
    if response < 0.15 or max(abs(shift_x), abs(shift_y)) > 35:
        raise ValueError(f"Image registration uncertain: shift={(shift_x, shift_y)}, response={response}")
    return shift_x, shift_y, response


def make_boxes(camera, image_shape, shift_x, shift_y):
    height, width = image_shape[:2]
    box_width, box_height = BOX_SIZE
    boxes = []
    for row_number, (x_start, x_end, y_start, y_end) in enumerate(ROW_GEOMETRY[camera], 1):
        for slot in range(25):
            t = slot / 24
            center_x = x_start + (x_end - x_start) * t + shift_x
            center_y = y_start + (y_end - y_start) * t + shift_y
            x1 = max(0.0, center_x - box_width / 2)
            y1 = max(0.0, center_y - box_height / 2)
            x2 = min(float(width), center_x + box_width / 2)
            y2 = min(float(height), center_y + box_height / 2)
            if x2 - x1 < box_width * .8 or y2 - y1 < box_height * .6:
                raise ValueError(f"Visible point too clipped: {camera}, row {row_number}, slot {slot + 1}")
            boxes.append((row_number, slot + 1, x1, y1, x2, y2))
    return boxes


def yolo_line(box, image_shape):
    _, _, x1, y1, x2, y2 = box
    height, width = image_shape[:2]
    return f"0 {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} {(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"


def draw_review(image, boxes):
    review = image.copy()
    for row, slot, x1, y1, x2, y2 in boxes:
        color = ((0, 255, 0), (255, 100, 0), (0, 120, 255))[row - 1]
        cv2.rectangle(review, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.putText(review, f"{row}:{slot}", (round(x1), max(14, round(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    return review


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize_old_label_line(line, path):
    values = [float(value) for value in line.split()]
    if not values or values[0] != 0:
        raise ValueError(f"Unexpected class in {path}: {line}")
    if len(values) == 5:
        box = values[1:]
    elif len(values) >= 7 and len(values) % 2 == 1:
        points = np.array(values[1:], dtype=float).reshape(-1, 2)
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        box = [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin]
    else:
        raise ValueError(f"Unsupported old label format in {path}: {line}")
    if any(value < 0 or value > 1 for value in box) or box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"Invalid old box in {path}: {line}")
    return "0 " + " ".join(f"{value:.8f}" for value in box)


def copy_old_dataset(old_root, output):
    records = []
    seen_hashes = set()
    for split in ("train", "val"):
        images_dir = old_root / "images" / split
        labels_dir = old_root / "labels" / split
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg"}:
                continue
            label_path = labels_dir / (image_path.stem + ".txt")
            if not label_path.is_file():
                raise ValueError(f"Missing old label: {label_path}")
            digest = sha256(image_path)
            if digest in seen_hashes:
                print("Skip duplicate old image across splits:", image_path, flush=True)
                continue
            seen_hashes.add(digest)
            target_name = "old_" + image_path.name
            target_image = output / "images" / split / target_name
            target_label = output / "labels" / split / ("old_" + image_path.stem + ".txt")
            shutil.copy2(image_path, target_image)
            lines = label_path.read_text(encoding="utf-8").splitlines()
            normalized = [normalize_old_label_line(line, label_path)
                          for line in lines if line.strip()]
            target_label.write_text("\n".join(normalized) + ("\n" if normalized else ""),
                                    encoding="utf-8")
            records.append({
                "origin": "old", "camera": "", "split": split,
                "source": str(image_path), "source_sha256": digest,
                "image": str(target_image), "label": str(target_label),
                "points": len(normalized),
            })
    return records


def add_factory_images(factory_root, output):
    records = []
    for camera in ("Cam1", "Cam2"):
        paths = sorted((factory_root / camera).glob("*.bmp"))
        if len(paths) != 10:
            raise ValueError(f"Expected 10 White frames in {camera}, found {len(paths)}")
        reference = crop_owned_rows(rotate(read_image(paths[0]), camera))
        for index, path in enumerate(paths):
            image = crop_owned_rows(rotate(read_image(path), camera))
            shift_x, shift_y, response = registration_shift(reference, image)
            boxes = make_boxes(camera, image.shape, shift_x, shift_y)
            split = "val" if index in VAL_INDICES else "train"
            stem = f"factory_{camera}_{path.stem}"
            target_image = output / "images" / split / (stem + ".png")
            target_label = output / "labels" / split / (stem + ".txt")
            review_path = output / "review" / camera / (stem + ".jpg")
            review_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target_image), image):
                raise OSError(f"Cannot write {target_image}")
            target_label.write_text(
                "\n".join(yolo_line(box, image.shape) for box in boxes) + "\n",
                encoding="utf-8",
            )
            if not cv2.imwrite(str(review_path), draw_review(image, boxes)):
                raise OSError(f"Cannot write {review_path}")
            records.append({
                "origin": "factory", "camera": camera, "split": split,
                "source": str(path), "source_sha256": sha256(path),
                "image": str(target_image), "label": str(target_label),
                "points": len(boxes), "shift_x": round(shift_x, 3),
                "shift_y": round(shift_y, 3), "registration_response": round(response, 3),
                "review": str(review_path),
            })
            print(camera, path.name, split, "shift", round(shift_x, 1), round(shift_y, 1),
                  "response", round(response, 3), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--factory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    records = copy_old_dataset(args.old_dataset.resolve(), output)
    records.extend(add_factory_images(args.factory.resolve(), output))
    (output / "disk.yaml").write_text(
        f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: point\n",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps({
            "label_policy": "Manual 25-slot grid per camera; registration per frame; old detector boxes not copied",
            "factory_rotation": {"Cam1": "clockwise90", "Cam2": "counterclockwise90"},
            "factory_crop_canonical_xyxy": [0, 0, 1944, FACTORY_CROP_BOTTOM],
            "factory_rows": ROW_GEOMETRY, "factory_box_size": BOX_SIZE,
            "factory_validation_indices": sorted(VAL_INDICES),
            "records": records,
        }, indent=2), encoding="utf-8")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["origin", "camera", "split", "source", "source_sha256", "image",
                  "label", "points", "shift_x", "shift_y", "registration_response", "review"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Dataset: {len(records)} images, {sum(r['points'] for r in records)} point boxes")


if __name__ == "__main__":
    main()
