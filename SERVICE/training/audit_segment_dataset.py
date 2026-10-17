"""Check generated YOLO polygons against their factory source masks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rasterize(path, shape):
    h, w = shape
    result = np.zeros(shape, np.uint8)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        values = np.array([float(value) for value in line.split()], np.float32)
        if len(values) < 7 or len(values) % 2 != 1 or values[0] != 0:
            raise ValueError(f"Invalid polygon: {path}")
        points = values[1:].reshape(-1, 2)
        if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError(f"Invalid coordinates: {path}")
        points *= [w, h]
        cv2.fillPoly(result, [np.rint(points).astype(np.int32)], 255)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("dataset", "crops", "curated"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    dataset = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    crops = json.loads((args.crops / "manifest.json").read_text(encoding="utf-8"))
    curated = json.loads((args.curated / "manifest.json").read_text(encoding="utf-8"))
    crop_lookup = {(r["source"], r["band"], r["tile"]): r for r in crops}
    mask_lookup = {(r["source"], r["band"]): r for r in curated}
    source_split = {}
    recall_values = []
    iou_values = []
    for record in dataset:
        family = "old" if record["origin"] == "old" else "factory"
        key = (family, record["source_group"])
        if key in source_split and source_split[key] != record["split"]:
            raise ValueError(f"Source leakage: {key}")
        source_split[key] = record["split"]
        image = cv2.imread(record["image"])
        if image is None:
            raise ValueError(record["image"])
        label = rasterize(record["label"], image.shape[:2])
        if record["origin"] not in {"factory", "hard_real", "faint_real"} or record["repeat"] != 0:
            continue
        crop = crop_lookup[(record["source"], record["band"], record["tile"])]
        mask = cv2.imread(mask_lookup[(record["source"], record["band"])]["curated_mask"], 0)
        x1, _, x2, _ = crop["rectified_band_xyxy"]
        if record["origin"] in {"hard_real", "faint_real"}:
            x1 = record["shifted_band_x1"]
            x2 = x1 + image.shape[1]
        target = mask[:, x1:x2] > 0
        observed = label > 0
        intersection = np.count_nonzero(target & observed)
        union = np.count_nonzero(target | observed)
        recall = intersection / np.count_nonzero(target) if target.any() else 1.0
        iou = intersection / union if union else 1.0
        recall_values.append(recall)
        iou_values.append(iou)
        if recall < .9:
            print("LOW RECALL", round(recall, 4), record["image"])
    print("factory/real shifted tiles", len(recall_values), "min_recall", min(recall_values),
          "mean_recall", float(np.mean(recall_values)), "mean_iou", float(np.mean(iou_values)))
    if min(recall_values) < .9:
        raise ValueError("Polygon conversion lost over 10% of some masks")


if __name__ == "__main__":
    main()
