"""Evaluate a candidate ONNX with the exact six-tile service decoder on HA val."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from curate_segment_masks import band_image, complete_component, ridge_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference.yolo_segment.onnx_segmentor import OnnxSegmentor  # noqa: E402


def complete_predicted_edges(mask, image):
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    ridge = ridge_image(image)
    result = mask.copy()
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        height = stats[index, cv2.CC_STAT_HEIGHT]
        width = stats[index, cv2.CC_STAT_WIDTH]
        if area < 100 or height < image.shape[0] * .35 or width > 180:
            continue
        repaired, _, _ = complete_component(components == index, ridge)
        result = np.maximum(result, repaired)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point-dataset", type=Path, required=True)
    p.add_argument("--curated", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conf", type=float, default=.5)
    p.add_argument("--preprocess", choices=("stretch", "letterbox"),
                   default="stretch")
    p.add_argument("--complete", action="store_true")
    p.add_argument("--mask-decode", choices=("legacy", "native"), default="legacy")
    p.add_argument("--tile-merge", choices=("max", "core"), default="max")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points = json.loads((args.point_dataset / "manifest.json").read_text(encoding="utf-8"))["records"]
    point_lookup = {r["source"]: r for r in points if r["origin"] == "factory"}
    labels = json.loads((args.curated / "manifest.json").read_text(encoding="utf-8"))
    labels = [r for r in labels if r["point_split"] == "val"]
    model = OnnxSegmentor(str(args.model.resolve()), ["disk"],
                          conf_thres=args.conf, iou_thres=.8,
                          preprocess_mode=args.preprocess, mask_decode=args.mask_decode,
                          tile_merge=args.tile_merge)
    report = {}
    for color_order in ("bgr_service_current", "rgb_candidate"):
        values = []
        for record in labels:
            band = band_image(point_lookup[record["source"]], record["band"])
            source = band if color_order == "bgr_service_current" else cv2.cvtColor(
                band, cv2.COLOR_BGR2RGB)
            predicted, _ = model.segment_large_image_debug(source, args.conf, .8)
            if args.complete:
                predicted = complete_predicted_edges(predicted, band)
            target = cv2.imread(record["curated_mask"], 0) > 0
            predicted = predicted > 0
            intersection = int(np.count_nonzero(predicted & target))
            union = int(np.count_nonzero(predicted | target))
            values.append({
                "camera": record["camera"], "band": record["band"],
                "source": record["source"],
                "iou": intersection / union if union else 1.0,
                "recall": intersection / np.count_nonzero(target) if target.any() else 1.0,
                "precision": intersection / np.count_nonzero(predicted) if predicted.any() else 0.0,
                "predicted_pixels": int(np.count_nonzero(predicted)),
                "target_pixels": int(np.count_nonzero(target)),
            })
            display = band.copy()
            display[predicted] = (0, 0, 255)
            cv2.imwrite(str(args.output / (
                f"{color_order}_{record['camera']}_{Path(record['source']).stem}_"
                f"{record['band']}.jpg")), cv2.addWeighted(band, .6, display, .4, 0))
        report[color_order] = {
            "mean_iou": float(np.mean([r["iou"] for r in values])),
            "mean_recall": float(np.mean([r["recall"] for r in values])),
            "mean_precision": float(np.mean([r["precision"] for r in values])),
            "bands": values,
        }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    for order, value in report.items():
        print(order, {k: v for k, v in value.items() if k != "bands"})


if __name__ == "__main__":
    main()
