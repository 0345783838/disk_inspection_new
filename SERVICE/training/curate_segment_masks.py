"""Repair truncated segmentor proposals against the new factory images.

The old ONNX output is a proposal, not ground truth. Each accepted long edge is
extended along a local image ridge; short/noisy components are excluded and
every repair is recorded for visual QA. No fixed edge count is imposed.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_segment_crops import band_box, read_point_boxes, rectify


def band_image(point_record, band_name):
    image = cv2.imread(point_record["image"])
    rows = read_point_boxes(point_record["label"], image.shape[1], image.shape[0])
    warped, _, _, transformed = rectify(image, rows)
    upper, lower, direction = ((transformed[0], transformed[1], "bottom")
                               if band_name == "band12" else
                               (transformed[1], transformed[2], "top"))
    x1, y1, x2, y2 = band_box(upper, lower, warped.shape, direction)
    return warped[y1:y2, x1:x2]


def ridge_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=1)
    return cv2.GaussianBlur(np.abs(gray - blur), (3, 3), 0)


def complete_component(component, ridge):
    height, width = component.shape
    ys, xs = np.where(component)
    observed = np.unique(ys)
    centers = np.array([np.median(xs[ys == y]) for y in observed], dtype=np.float32)
    widths = np.array([np.count_nonzero(component[y]) for y in observed], dtype=np.float32)
    thickness = int(np.clip(np.median(widths), 3, 12))
    all_y = np.arange(height)
    path = np.interp(all_y, observed, centers).astype(np.float32)
    for side in ("top", "bottom"):
        take = min(60, len(observed))
        fit_y = observed[:take] if side == "top" else observed[-take:]
        fit_x = centers[:take] if side == "top" else centers[-take:]
        slope = float(np.polyfit(fit_y, fit_x, 1)[0]) if len(fit_y) > 1 else 0.0
        slope = float(np.clip(slope, -0.8, 0.8))
        missing_y = all_y[all_y < observed[0]] if side == "top" else all_y[all_y > observed[-1]]
        anchor_y = observed[0] if side == "top" else observed[-1]
        anchor_x = centers[0] if side == "top" else centers[-1]
        path[missing_y] = anchor_x + slope * (missing_y - anchor_y)
    missing = np.ones(height, dtype=bool)
    missing[observed] = False
    evidence = []
    for y in np.flatnonzero(missing):
        center = path[y]
        x1 = max(0, int(round(center)) - 4)
        x2 = min(width, int(round(center)) + 5)
        if x2 <= x1:
            continue
        values = ridge[y, x1:x2] - .8 * (np.arange(x1, x2) - center) ** 2
        best_x = x1 + int(np.argmax(values))
        path[y] = best_x
        evidence.append(float(ridge[y, best_x]))
    path = cv2.GaussianBlur(path[:, None], (1, 21), 0)[:, 0]
    repaired = component.astype(np.uint8) * 255
    added = np.zeros_like(repaired)
    for y in np.flatnonzero(missing):
        x = int(round(float(path[y])))
        if 0 <= x < width:
            cv2.line(added, (x, y), (x, y), 255, thickness)
    repaired = np.maximum(repaired, added)
    return repaired, added, {
        "first_y": int(observed[0]), "last_y": int(observed[-1]),
        "added_rows": int(np.count_nonzero(missing)),
        "median_extension_ridge": round(float(np.median(evidence)), 2) if evidence else None,
        "thickness": thickness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "mask").mkdir(parents=True)
    (output / "review").mkdir()
    point_records = json.loads((args.point_dataset / "manifest.json").read_text(encoding="utf-8"))["records"]
    point_lookup = {record["source"]: record for record in point_records if record["origin"] == "factory"}
    proposal_records = json.loads((args.audit / "manifest.json").read_text(encoding="utf-8"))
    report = []
    for record in proposal_records:
        image = band_image(point_lookup[record["source"]], record["band"])
        proposal = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
        if proposal is None or proposal.shape != image.shape[:2]:
            raise ValueError(record["mask_path"])
        ridge = ridge_image(image)
        count, components, stats, centroids = cv2.connectedComponentsWithStats(
            (proposal > 0).astype(np.uint8), 8)
        result_mask = np.zeros_like(proposal)
        added_mask = np.zeros_like(proposal)
        details = []
        rejected = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
            component_width = int(stats[index, cv2.CC_STAT_WIDTH])
            if area < 100 or component_height < image.shape[0] * .35 or component_width > 180:
                if area >= 100:
                    rejected.append({"x": round(float(centroids[index, 0]), 1),
                                     "area": area, "height": component_height,
                                     "width": component_width})
                continue
            component = components == index
            repaired, added, detail = complete_component(component, ridge)
            detail.update({"x": round(float(centroids[index, 0]), 1), "area": area})
            details.append(detail)
            result_mask = np.maximum(result_mask, repaired)
            added_mask = np.maximum(added_mask, added)
        stem = f"{record['camera']}_{Path(record['source']).stem}_{record['band']}"
        mask_path = output / "mask" / (stem + ".png")
        review_path = output / "review" / (stem + ".jpg")
        cv2.imwrite(str(mask_path), result_mask)
        review = image.copy()
        review[result_mask > 0] = (0, 0, 255)
        review[added_mask > 0] = (0, 255, 0)
        cv2.imwrite(str(review_path), cv2.addWeighted(image, .6, review, .4, 0))
        report.append({**record, "curated_mask": str(mask_path),
                       "curated_review": str(review_path), "accepted_edges": len(details),
                       "edges": details, "rejected_components": rejected})
        print(stem, "accepted", len(details), "rejected", len(rejected),
              "extended", sum(d["added_rows"] > 0 for d in details), flush=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
