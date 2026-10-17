"""Evaluate candidate and production ONNX via the SERVICE inference wrapper."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference.yolo_detect.onnx_detector import OnnxDetector  # noqa: E402


def ground_truth(label_path, width, height):
    boxes = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        values = [float(v) for v in line.split()]
        _, x, y, w, h = values
        boxes.append(((x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def match_boxes(truth, predictions, threshold=0.3):
    if not len(truth) or not len(predictions):
        return 0, [0] * len(truth)
    matches = []
    for truth_idx, a in enumerate(truth):
        for pred_idx, b in enumerate(predictions):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            iou = intersection / max(1, area_a + area_b - intersection)
            if iou >= threshold:
                matches.append((iou, truth_idx, pred_idx))
    matched_truth, matched_pred = set(), set()
    for _, truth_idx, pred_idx in sorted(matches, reverse=True):
        if truth_idx not in matched_truth and pred_idx not in matched_pred:
            matched_truth.add(truth_idx)
            matched_pred.add(pred_idx)
    return len(matched_pred), [int(idx in matched_truth) for idx in range(len(truth))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.1)
    parser.add_argument("--all-factory", action="store_true")
    args = parser.parse_args()
    records = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))["records"]
    val_records = [record for record in records
                   if record["split"] == "val" or (args.all_factory and record["origin"] == "factory")]
    results = {}
    for model_name, model_path in (("production", args.production), ("candidate", args.candidate)):
        detector = OnnxDetector(str(model_path.resolve()), ["point"],
                                conf_thres=args.conf, iou_thres=args.iou)
        results[model_name] = []
        for record in val_records:
            image = cv2.imread(record["image"])
            if image is None:
                raise ValueError(record["image"])
            height, width = image.shape[:2]
            truth = ground_truth(record["label"], width, height)
            boxes, scores, _ = detector(image)
            predictions = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            matched, truth_matches = match_boxes(truth, predictions)
            result = {
                "source": record["source"], "origin": record["origin"],
                "camera": record["camera"], "truth": len(truth),
                "predictions": len(predictions), "matched_iou_0_3": matched,
                "false_positives": len(predictions) - matched,
            }
            if record["origin"] == "factory":
                result["row_matches"] = [sum(truth_matches[i:i + 25]) for i in (0, 25, 50)]
                result["three_rows_complete"] = result["row_matches"] == [25, 25, 25]
            results[model_name].append(result)
        print(model_name, flush=True)
        groups = defaultdict(list)
        for result in results[model_name]:
            groups[result["camera"] or "old"].append(result)
        for group, rows in groups.items():
            total_truth = sum(row["truth"] for row in rows)
            total_predictions = sum(row["predictions"] for row in rows)
            total_matched = sum(row["matched_iou_0_3"] for row in rows)
            complete = sum(row.get("three_rows_complete", False) for row in rows)
            print(f"  {group}: frames={len(rows)} recall={total_matched}/{total_truth} "
                  f"predictions={total_predictions} FP={total_predictions - total_matched} "
                  f"complete_3x25={f'{complete}/{len(rows)}' if group != 'old' else 'n/a'}",
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
