"""Make held-out HA-derived anomaly scenarios; never include these in train."""
import argparse
import json
from pathlib import Path

import cv2

from build_segment_dataset import polygons_from_mask
from synthesize_segment_scenarios import select_edge, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    (args.output / "images" / "val").mkdir(parents=True)
    (args.output / "labels" / "val").mkdir(parents=True)
    records = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    crops = json.loads((args.crops / "manifest.json").read_text(encoding="utf-8"))
    curated = json.loads((args.curated / "manifest.json").read_text(encoding="utf-8"))
    crop_lookup = {(r["source"], r["band"], r["tile"]): r for r in crops}
    mask_lookup = {(r["source"], r["band"]): r for r in curated}
    selected = [r for r in records if r["origin"] == "factory" and
                r["split"] == "val" and r["repeat"] == 0]
    report = []
    for index, record in enumerate(selected):
        image = cv2.imread(record["image"])
        crop = crop_lookup[(record["source"], record["band"], record["tile"])]
        mask = cv2.imread(mask_lookup[(record["source"], record["band"])][
            "curated_mask"], cv2.IMREAD_GRAYSCALE)
        x1, _, x2, _ = crop["rectified_band_xyxy"]
        mask = mask[:, x1:x2]
        instance = select_edge(mask)
        if instance is None:
            continue
        for variant, scenario in enumerate(("empty_slot", "near_overlap",
                                            "crossing")):
            synthetic, added_mask = transform(image, instance, scenario,
                                               2000 + index * 3 + variant)
            if scenario == "empty_slot":
                lines = polygons_from_mask(cv2.bitwise_and(
                    mask, cv2.bitwise_not(instance)))
            else:
                lines = Path(record["label"]).read_text(
                    encoding="utf-8").splitlines()
                extra = polygons_from_mask(added_mask)
                if len(extra) != 1:
                    continue
                lines += extra
            stem = Path(record["image"]).stem + "_" + scenario
            target_image = args.output / "images" / "val" / (stem + ".png")
            target_label = args.output / "labels" / "val" / (stem + ".txt")
            cv2.imwrite(str(target_image), synthetic)
            target_label.write_text("\n".join(lines) +
                                    ("\n" if lines else ""), encoding="utf-8")
            report.append({**record, "origin": "synthetic_challenge",
                           "scenario": scenario, "image": str(target_image),
                           "label": str(target_label), "objects": len(lines)})
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "challenge.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/val\nval: images/val\n"
        "names:\n  0: disk\n", encoding="utf-8")
    for scenario in ("empty_slot", "near_overlap", "crossing"):
        print(scenario, sum(r["scenario"] == scenario for r in report))


if __name__ == "__main__":
    main()
