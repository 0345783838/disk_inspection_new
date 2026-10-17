"""Train a non-deployed disk-edge segmentor candidate on HA crops."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--lr0", type=float, default=.001)
    args = parser.parse_args()
    model = YOLO(str(args.pretrained.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device=0,
        seed=930,
        deterministic=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        weight_decay=0.0005,
        patience=12,
        amp=True,
        overlap_mask=False,
        mask_ratio=2,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.02,
        scale=0.05,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.005,
        hsv_s=0.10,
        hsv_v=0.12,
        bgr=0.0,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        plots=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
