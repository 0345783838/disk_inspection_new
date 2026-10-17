"""Inspect old point detector predictions on rotated factory White frames."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def rotate(image, camera):
    return cv2.rotate(
        image,
        cv2.ROTATE_90_CLOCKWISE if camera == "Cam1"
        else cv2.ROTATE_90_COUNTERCLOCKWISE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    rows = []
    detections = []
    for camera in ("Cam1", "Cam2"):
        for path in sorted((args.factory / camera).glob("*.bmp")):
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                raise ValueError(f"Cannot read {path}")
            image = rotate(raw, camera)
            result = model.predict(
                image, conf=args.conf, iou=0.1, imgsz=args.imgsz,
                max_det=300, device=0, verbose=False,
            )[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                detections.append({
                    "camera": camera, "image": path.name,
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "score": float(score),
                })
            centers = (boxes[:, 1] + boxes[:, 3]) / 2 if len(boxes) else np.array([])
            if camera == "Cam1":
                bands = ((100, 330), (1030, 1250), (1930, 2160), (2200, image.shape[0]))
            else:
                bands = ((0, 180), (850, 1100), (1760, 2000), (2050, image.shape[0]))
            counts = [int(np.count_nonzero((centers >= low) & (centers < high)))
                      for low, high in bands]
            canvas = image.copy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(canvas, f"{score:.2f}", (x1, max(14, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1)
            for low, high in bands[:3]:
                cv2.line(canvas, (0, low), (canvas.shape[1], low), (0, 0, 255), 2)
                cv2.line(canvas, (0, high), (canvas.shape[1], high), (0, 0, 255), 2)
            target = args.output / camera / (path.stem + ".jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), canvas):
                raise OSError(f"Cannot write {target}")
            rows.append({
                "camera": camera, "source": str(path), "total": len(boxes),
                "row1": counts[0], "row2": counts[1], "row3": counts[2],
                "overlap": counts[3], "min_conf": min(scores) if len(scores) else "",
                "overlay": str(target),
            })
            print(camera, path.name, counts, flush=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "boxes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detections[0].keys())
        writer.writeheader()
        writer.writerows(detections)


if __name__ == "__main__":
    main()
