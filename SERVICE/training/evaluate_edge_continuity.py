"""Measure held-out HA edge recall and near-boundary continuity per tile."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def read_instances(path, shape):
    h, w = shape
    instances = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        values = np.asarray([float(value) for value in line.split()], np.float32)
        points = values[1:].reshape(-1, 2)
        points[:, 0] *= w
        points[:, 1] *= h
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        instances.append(mask.astype(bool))
    return instances


def metrics_for_tile(record, predicted):
    image = cv2.imread(record["image"])
    h, w = image.shape[:2]
    targets = read_instances(record["label"], (h, w))
    results = []
    for target in targets:
        area = np.count_nonzero(target)
        if not area:
            continue
        best_iou, best = 0.0, None
        for candidate in predicted:
            intersection = np.count_nonzero(target & candidate)
            union = area + np.count_nonzero(candidate) - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou, best = iou, candidate
        ys = np.flatnonzero(target.any(axis=1))
        top = ys[:min(12, len(ys))]
        bottom = ys[-min(12, len(ys)):]
        full = bool(best is not None and best_iou >= .3 and
                    np.any((best & target)[top]) and
                    np.any((best & target)[bottom]))
        results.append({"iou": best_iou, "found": best_iou >= .3,
                        "full": full})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=.25)
    args = parser.parse_args()
    records = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in records if r["origin"] == "factory" and
               r["split"] == "val" and r["repeat"] == 0]
    model = YOLO(str(args.model.resolve()))
    grouped = defaultdict(list)
    paths = [r["image"] for r in records]
    predictions = model.predict(source=paths, stream=True, batch=4,
                                imgsz=args.imgsz, conf=args.conf, iou=.8, device=0,
                                retina_masks=True, verbose=False)
    for record, result in zip(records, predictions):
        image = cv2.imread(record["image"])
        shape = image.shape[:2]
        masks = []
        if result.masks is not None:
            for value in result.masks.data.cpu().numpy():
                if value.shape != shape:
                    value = cv2.resize(value, shape[::-1],
                                       interpolation=cv2.INTER_NEAREST)
                masks.append(value > .5)
        grouped[f'{record["camera"]}_{record["band"]}'].extend(
            metrics_for_tile(record, masks))
    report = {}
    for group, edges in grouped.items():
        report[group] = {
            "gt_edges_in_tiles": len(edges),
            "edge_recall_iou30": float(np.mean([e["found"] for e in edges])),
            "full_length_recall_iou30": float(np.mean([e["full"] for e in edges])),
            "mean_best_iou": float(np.mean([e["iou"] for e in edges])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
