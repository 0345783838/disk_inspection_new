"""Fine-tune YOLO11n point detector without replacing the production ONNX."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not (dataset / "disk.yaml").is_file():
        raise FileNotFoundError(dataset / "disk.yaml")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "data": str(dataset / "disk.yaml"),
        "model": str(checkpoint),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": 640,
        "device": 0,
        "workers": 2,
        "seed": 42,
        "patience": 20,
        "lr0": 0.001,
        "lrf": 0.1,
        "optimizer": "AdamW",
        "degrees": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "translate": 0.03,
        "scale": 0.15,
        "mosaic": 0.5,
        "close_mosaic": 10,
        "hsv_h": 0.005,
        "hsv_s": 0.25,
        "hsv_v": 0.2,
    }
    (output / "training_request.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = YOLO(str(checkpoint))
    model.train(
        data=config["data"], epochs=config["epochs"], batch=config["batch"],
        imgsz=config["imgsz"], device=config["device"], workers=config["workers"],
        seed=config["seed"], patience=config["patience"], lr0=config["lr0"],
        lrf=config["lrf"], optimizer=config["optimizer"],
        degrees=config["degrees"], flipud=config["flipud"], fliplr=config["fliplr"],
        translate=config["translate"], scale=config["scale"], mosaic=config["mosaic"],
        close_mosaic=config["close_mosaic"], hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"], hsv_v=config["hsv_v"],
        project=str(output), name="finetune", exist_ok=True, plots=True,
        save=True, val=True, verbose=True,
    )


if __name__ == "__main__":
    main()
