"""Combine a source-grouped old subset with curated factory edge masks."""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def stable_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def image_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_old_label(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        values = [float(value) for value in line.split()]
        if len(values) < 7 or len(values) % 2 != 1 or values[0] != 0:
            raise ValueError(f"Invalid polygon in {path}")
        if any(value < 0 or value > 1 for value in values[1:]):
            raise ValueError(f"Out-of-range polygon in {path}")
    return len(lines)


def copy_old(old_root, output, per_source):
    groups = defaultdict(list)
    for original_split in ("train", "val"):
        for source in sorted((old_root / "images" / original_split).iterdir()):
            if source.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg"}:
                continue
            label = old_root / "labels" / original_split / (source.stem + ".txt")
            if not label.exists():
                raise FileNotFoundError(label)
            group = source.stem.split("_crop_")[0]
            groups[group].append((source, label, original_split))
    records = []
    for group, candidates in sorted(groups.items()):
        split = "val" if int(stable_hash(group)[:8], 16) % 5 == 0 else "train"
        seen = set()
        selected = []
        for source, label, original_split in sorted(candidates, key=lambda c: stable_hash(c[0])):
            digest = image_hash(source)
            if digest in seen:
                continue
            seen.add(digest)
            selected.append((source, label, original_split, digest))
            if len(selected) >= per_source:
                break
        for source, label, original_split, digest in selected:
            name = f"old_{original_split}_{source.stem}{source.suffix.lower()}"
            target_image = output / "images" / split / name
            target_label = output / "labels" / split / (Path(name).stem + ".txt")
            shutil.copy2(source, target_image)
            shutil.copy2(label, target_label)
            records.append({"origin": "old", "split": split, "source_group": group,
                            "source": str(source), "sha256": digest,
                            "image": str(target_image), "label": str(target_label),
                            "objects": validate_old_label(label)})
    return records


def polygons_from_mask(mask, min_area=15):
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        _, _, box_width, box_height = cv2.boundingRect(contour)
        if box_height < 12 or box_width < 2:
            continue
        contour = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        if len(contour) < 3:
            continue
        points = np.clip(contour / np.array([width, height], dtype=np.float32), 0, 1)
        lines.append("0 " + " ".join(f"{value:.6f}" for value in points.reshape(-1)))
    return lines


def augment_factory(image, variant, seed):
    """Photometric-only changes; edge geometry and polygon masks stay valid."""
    if variant == 0:
        return image, "original"
    rgb = image.astype(np.float32) / 255.0
    if variant == 1:
        changed = np.power(rgb, 1.12)
        name = "low_light_gamma_1_12"
    elif variant == 2:
        mean = rgb.mean(axis=(0, 1), keepdims=True)
        changed = np.power(np.clip(mean + .82 * (rgb - mean), 0, 1), .92)
        name = "reduced_edge_contrast"
    elif variant == 3:
        changed = cv2.GaussianBlur(rgb, (3, 3), .7)
        rng = np.random.default_rng(seed)
        changed = changed + rng.normal(0, 1.5 / 255.0, changed.shape)
        name = "mild_blur_sensor_noise"
    else:
        raise ValueError("Only four physically mild factory variants are supported")
    return np.uint8(np.clip(changed * 255, 0, 255)), name


def add_factory(crops_root, curated_root, output, train_repeats):
    crops = json.loads((crops_root / "manifest.json").read_text(encoding="utf-8"))
    masks = json.loads((curated_root / "manifest.json").read_text(encoding="utf-8"))
    mask_lookup = {(record["source"], record["band"]): record for record in masks}
    records = []
    for crop in crops:
        source = crop["source"]
        split = crop["point_split"]
        mask_record = mask_lookup[(source, crop["band"])]
        full_mask = cv2.imread(mask_record["curated_mask"], cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = crop["rectified_band_xyxy"]
        mask_tile = full_mask[:, x1:x2]
        image = cv2.imread(crop["image"])
        if image is None or image.shape[:2] != mask_tile.shape:
            raise ValueError(f"Mask/image shape mismatch: {crop['image']}")
        lines = polygons_from_mask(mask_tile)
        repeats = train_repeats if split == "train" else 1
        for repeat in range(repeats):
            stem = f"factory_{Path(crop['image']).stem}_r{repeat}"
            target_image = output / "images" / split / (stem + ".png")
            target_label = output / "labels" / split / (stem + ".txt")
            augmented, augmentation = augment_factory(
                image, repeat, int(stable_hash(stem)[:8], 16))
            if not cv2.imwrite(str(target_image), augmented):
                raise OSError(target_image)
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            records.append({"origin": "factory", "split": split,
                            "source_group": f"{crop['camera']}_{Path(source).stem}",
                            "camera": crop["camera"], "band": crop["band"],
                            "tile": crop["tile"], "source": source,
                            "image": str(target_image), "label": str(target_label),
                            "objects": len(lines), "repeat": repeat,
                            "augmentation": augmentation,
                            "label_status": mask_record.get(
                                "label_status", "semi_automatic_provisional"),
                            "band_review_ids": [
                                review["id"] for review in mask_record.get("human_reviews", [])],
                            "band_review_scope": "selected_edges_only"})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-per-source", type=int, default=3)
    parser.add_argument("--factory-train-repeats", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.factory_train_repeats <= 4:
        parser.error("--factory-train-repeats must be between 1 and 4")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    records = copy_old(args.old.resolve(), output, args.old_per_source)
    records += add_factory(args.crops.resolve(), args.curated.resolve(), output,
                           args.factory_train_repeats)
    for split in ("train", "val"):
        (output / f"disk_{split}.yaml").write_text(
            f"path: {output.as_posix()}\ntrain: images/train\nval: images/{split}\nnames:\n  0: disk\n",
            encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    for origin in ("old", "factory"):
        for split in ("train", "val"):
            selected = [record for record in records if record["origin"] == origin and record["split"] == split]
            print(origin, split, "images", len(selected), "polygons",
                  sum(record["objects"] for record in selected))


if __name__ == "__main__":
    main()
