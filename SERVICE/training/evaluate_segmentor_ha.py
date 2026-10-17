"""Report old-source and held-out HA segmentation metrics separately."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    model = YOLO(str(args.model.resolve()))
    results = {}
    for origin in ("factory", "old"):
        paths = [r["image"] for r in records if r["origin"] == origin and
                 r["split"] == "val"]
        if not paths:
            raise ValueError(f"No validation images for {origin}")
        file_list = args.output / f"{origin}_val.txt"
        file_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
        yaml = args.output / f"{origin}_val.yaml"
        yaml.write_text(
            f"path: {args.dataset.resolve().as_posix()}\n"
            f"train: images/train\nval: {file_list.resolve().as_posix()}\n"
            "names:\n  0: disk\n", encoding="utf-8")
        metrics = model.val(data=str(yaml.resolve()), split="val", imgsz=args.imgsz,
                            batch=4, device=0, workers=2, plots=False,
                            project=str(args.output.resolve()), name=origin,
                            exist_ok=True, verbose=False)
        results[origin] = {
            "images": len(paths),
            "mask_precision": float(metrics.seg.mp),
            "mask_recall": float(metrics.seg.mr),
            "mask_map50": float(metrics.seg.map50),
            "mask_map50_95": float(metrics.seg.map),
            "box_map50": float(metrics.box.map50),
        }
        print(origin, results[origin], flush=True)
    (args.output / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
