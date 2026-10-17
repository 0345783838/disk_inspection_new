"""Make zoom sheets for weak, long model-assisted edge extensions."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from curate_segment_masks import band_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point-dataset", type=Path, required=True)
    p.add_argument("--curated", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points = json.loads((args.point_dataset / "manifest.json").read_text(encoding="utf-8"))["records"]
    lookup = {r["source"]: r for r in points if r["origin"] == "factory"}
    records = json.loads((args.curated / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for record in records:
        risky = [edge for edge in record["edges"] if edge["added_rows"] > 40 and
                 edge["median_extension_ridge"] is not None and
                 edge["median_extension_ridge"] < 10]
        if not risky:
            continue
        image = band_image(lookup[record["source"]], record["band"])
        review = cv2.imread(record["curated_review"])
        for edge in risky:
            x = int(round(edge["x"]))
            x1, x2 = max(0, x - 65), min(image.shape[1], x + 65)
            original = image[:, x1:x2]
            overlay = review[:, x1:x2]
            panel = np.concatenate([original, overlay], axis=1)
            panel = cv2.copyMakeBorder(panel, 25, 0, 0, 0, cv2.BORDER_CONSTANT)
            title = (f"{record['camera']} {record['band']} "
                     f"{Path(record['source']).stem[-8:]} x={x} "
                     f"add={edge['added_rows']} ridge={edge['median_extension_ridge']}")
            cv2.putText(panel, title, (3, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        .38, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append({"source": record["source"], "camera": record["camera"],
                         "band": record["band"], **edge,
                         "review": record["curated_review"],
                         "panel": panel})
    rows.sort(key=lambda r: (r["median_extension_ridge"], -r["added_rows"]))
    report = [{k: v for k, v in r.items() if k != "panel"} for r in rows]
    (args.output / "queue.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    for start in range(0, len(rows), 7):
        batch = rows[start:start + 7]
        width = max(row["panel"].shape[1] for row in batch)
        sheet = np.concatenate([
            cv2.copyMakeBorder(row["panel"], 0, 0, 0,
                               width - row["panel"].shape[1],
                               cv2.BORDER_CONSTANT)
            for row in batch], axis=0)
        cv2.imwrite(str(args.output / f"weak_extension_{start // 7 + 1}.jpg"), sheet)
    print("weak_long_extensions", len(rows), "sheets", (len(rows) + 6) // 7)


if __name__ == "__main__":
    main()
