"""Create controlled disk-edge scenarios from real HA crops with exact masks.

Image-generation edits are intentionally not used for labeled training: they
can redraw every edge. This script removes or repositions a measured edge
residual, so each changed edge has a corresponding transformed instance mask.
Synthetic images are train-only and kept separate from source validation.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from build_segment_dataset import polygons_from_mask


def select_edge(mask):
    count, components, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if h < height * .75 or w > 90 or area < 80:
            continue
        if x < 26 or x + w > width - 26:
            continue
        candidates.append((abs(centers[index, 0] - width / 2), index))
    if not candidates:
        return None
    return np.uint8(components == min(candidates)[1]) * 255


def transform(image, instance, scenario, seed):
    h, w = instance.shape
    rng = np.random.default_rng(seed)
    erase = cv2.dilate(instance, np.ones((5, 5), np.uint8), iterations=2)
    background = cv2.inpaint(image, erase, 5, cv2.INPAINT_TELEA)
    residual = image.astype(np.float32) - background.astype(np.float32)
    if scenario == "empty_slot":
        return background, None
    y, x = np.mgrid[:h, :w].astype(np.float32)
    if scenario == "near_overlap":
        distance = int(rng.integers(9, 21)) * int(rng.choice([-1, 1]))
        shift = np.full((h, 1), distance, np.float32)
    elif scenario == "crossing":
        top = int(rng.integers(16, 31))
        bottom = int(rng.integers(16, 31))
        direction = int(rng.choice([-1, 1]))
        shift = np.linspace(-direction * top, direction * bottom, h,
                            dtype=np.float32)[:, None]
    else:
        raise ValueError(scenario)
    map_x = x - shift
    shifted_mask = cv2.remap(instance, map_x, y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shifted_residual = cv2.remap(residual, map_x, y, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Preserve the original edge; only blend a camera-derived residual.
    strength = float(rng.uniform(.65, .9))
    synthetic = np.uint8(np.clip(image.astype(np.float32) +
                                 strength * shifted_residual, 0, 255))
    return synthetic, shifted_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--curated", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    shutil.copytree(args.dataset, args.output)
    records = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    for record in records:
        record["image"] = str(args.output / "images" / record["split"] /
                              Path(record["image"]).name)
        record["label"] = str(args.output / "labels" / record["split"] /
                              Path(record["label"]).name)
    crops = json.loads((args.crops / "manifest.json").read_text(encoding="utf-8"))
    curated = json.loads((args.curated / "manifest.json").read_text(encoding="utf-8"))
    crop_lookup = {(r["source"], r["band"], r["tile"]): r for r in crops}
    mask_lookup = {(r["source"], r["band"]): r for r in curated}
    scenarios = ("empty_slot", "near_overlap", "crossing")
    added = []
    previews = []
    originals = [r for r in records if r["origin"] == "factory" and
                 r["split"] == "train" and r["repeat"] == 0]
    for index, record in enumerate(originals):
        scenario = scenarios[index % len(scenarios)]
        image = cv2.imread(record["image"])
        crop = crop_lookup[(record["source"], record["band"], record["tile"])]
        mask = cv2.imread(mask_lookup[(record["source"], record["band"])][
            "curated_mask"], cv2.IMREAD_GRAYSCALE)
        x1, _, x2, _ = crop["rectified_band_xyxy"]
        mask = mask[:, x1:x2]
        instance = select_edge(mask)
        if instance is None:
            continue
        synthetic, added_mask = transform(image, instance, scenario, index + 930)
        if scenario == "empty_slot":
            label_mask = cv2.bitwise_and(mask, cv2.bitwise_not(instance))
            lines = polygons_from_mask(label_mask)
        else:
            lines = Path(record["label"]).read_text(encoding="utf-8").splitlines()
            new_lines = polygons_from_mask(added_mask)
            if len(new_lines) != 1:
                continue
            lines += new_lines
        stem = Path(record["image"]).stem + "_" + scenario
        image_path = args.output / "images" / "train" / (stem + ".png")
        label_path = args.output / "labels" / "train" / (stem + ".txt")
        cv2.imwrite(str(image_path), synthetic)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")
        added.append({**record, "origin": "synthetic", "image": str(image_path),
                      "label": str(label_path), "objects": len(lines),
                      "scenario": scenario, "label_status": "geometry_exact_synthetic"})
        if len(previews) < 18:
            preview = np.concatenate([image, synthetic], axis=1)
            cv2.putText(preview, scenario, (image.shape[1] + 5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
            previews.append(preview)
    (args.output / "manifest.json").write_text(
        json.dumps(records + added, indent=2), encoding="utf-8")
    for split in ("train", "val"):
        (args.output / f"disk_{split}.yaml").write_text(
            f"path: {args.output.resolve().as_posix()}\n"
            f"train: images/train\nval: images/{split}\n"
            "names:\n  0: disk\n", encoding="utf-8")
    if previews:
        width = max(p.shape[1] for p in previews)
        grid = np.concatenate([cv2.copyMakeBorder(
            preview, 0, 0, 0, width - preview.shape[1],
            cv2.BORDER_CONSTANT) for preview in previews], axis=0)
        cv2.imwrite(str(args.output / "scenario_review.jpg"), grid)
    for scenario in scenarios:
        print(scenario, sum(r["scenario"] == scenario for r in added))
    print("total_synthetic", len(added), "skipped", len(originals) - len(added))


if __name__ == "__main__":
    main()
