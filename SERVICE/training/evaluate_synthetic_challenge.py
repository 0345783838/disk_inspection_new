"""Evaluate model separately on held-out synthetic anomaly categories."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--challenge", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.challenge / "manifest.json").read_text(encoding="utf-8"))
    model = YOLO(str(args.model.resolve()))
    report = {}
    for scenario in ("empty_slot", "near_overlap", "crossing"):
        paths = [r["image"] for r in records if r["scenario"] == scenario]
        listing = args.output / f"{scenario}.txt"
        listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
        yaml = args.output / f"{scenario}.yaml"
        yaml.write_text(
            f"path: {args.challenge.resolve().as_posix()}\n"
            f"train: images/val\nval: {listing.resolve().as_posix()}\n"
            "names:\n  0: disk\n", encoding="utf-8")
        metrics = model.val(data=str(yaml.resolve()), split="val", imgsz=args.imgsz,
                            batch=4, device=0, workers=2, plots=False,
                            project=str(args.output.resolve()), name=scenario,
                            exist_ok=True, verbose=False)
        report[scenario] = {
            "images": len(paths),
            "mask_precision": float(metrics.seg.mp),
            "mask_recall": float(metrics.seg.mr),
            "mask_map50": float(metrics.seg.map50),
            "mask_map50_95": float(metrics.seg.map),
        }
        print(scenario, report[scenario], flush=True)
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
