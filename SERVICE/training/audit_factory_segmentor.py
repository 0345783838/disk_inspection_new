"""Audit old segmentor proposals on new bands; these are not training labels."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from build_segment_crops import band_box, read_point_boxes, rectify

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference.yolo_segment.onnx_segmentor import OnnxSegmentor  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=.05)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "mask").mkdir(parents=True)
    (output / "overlay").mkdir()
    model = OnnxSegmentor(str(args.model.resolve()), ["disk"],
                          conf_thres=args.conf, iou_thres=.8)
    source = json.loads((args.point_dataset / "manifest.json").read_text(encoding="utf-8"))["records"]
    aggregates = defaultdict(list)
    rows = []
    for record in source:
        if record["origin"] != "factory":
            continue
        image = cv2.imread(record["image"])
        points = read_point_boxes(record["label"], image.shape[1], image.shape[0])
        warped, _, _, transformed = rectify(image, points)
        for band_name, upper, lower, direction in (
                ("band12", transformed[0], transformed[1], "bottom"),
                ("band23", transformed[1], transformed[2], "top")):
            x1, y1, x2, y2 = band_box(upper, lower, warped.shape, direction)
            band = warped[y1:y2, x1:x2]
            mask, _ = model.segment_large_image_debug(band, args.conf, .8)
            stem = f"{record['camera']}_{Path(record['source']).stem}_{band_name}"
            mask_path = output / "mask" / (stem + ".png")
            overlay_path = output / "overlay" / (stem + ".jpg")
            cv2.imwrite(str(mask_path), mask)
            display = band.copy()
            display[mask > 0] = (0, 0, 255)
            cv2.imwrite(str(overlay_path), cv2.addWeighted(band, .65, display, .35, 0))
            aggregates[(record["camera"], band_name)].append(mask > 0)
            rows.append({"camera": record["camera"], "band": band_name,
                         "source": record["source"], "point_split": record["split"],
                         "mask_path": str(mask_path), "overlay_path": str(overlay_path),
                         "mask_area": int(np.count_nonzero(mask))})
            print(stem, "area", rows[-1]["mask_area"], flush=True)
    for (camera, band_name), masks in aggregates.items():
        average = np.mean(np.stack(masks), axis=0)
        heatmap = cv2.applyColorMap(np.round(average * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(output / f"{camera}_{band_name}_occupancy.png"), heatmap)
    (output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
