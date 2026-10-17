"""Apply source-bound human decisions to a new mask version with provenance."""
import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from curate_segment_masks import band_image, complete_component, ridge_image


def human_path(points, shape, thickness):
    height, width = shape
    ordered = sorted(points, key=lambda p: p["y"])
    y = np.array([p["y"] for p in ordered], np.float64)
    x = np.array([p["x"] for p in ordered], np.float64)
    if len(y) < 3 or np.any(np.diff(y) <= 0) or not np.isfinite([x, y]).all():
        raise ValueError("Correction needs at least three points with distinct y")
    if y[0] > 15 or y[-1] < height - 16:
        raise ValueError("Partial-span correction needs explicit manual handling")
    rows = np.arange(height)
    center = PchipInterpolator(y, x, extrapolate=False)(np.clip(rows, y[0], y[-1]))
    for side, anchor, slope in (
            (rows < y[0], 0, (x[1] - x[0]) / (y[1] - y[0])),
            (rows > y[-1], -1, (x[-1] - x[-2]) / (y[-1] - y[-2]))):
        center[side] = x[anchor] + np.clip(slope, -.8, .8) * (rows[side] - y[anchor])
    result = np.zeros(shape, np.uint8)
    for row, value in enumerate(center):
        left = int(round(value)) - thickness // 2
        right = left + thickness
        result[row, max(0, left):min(width, right)] = 255
    return result, center


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point-dataset", type=Path, required=True)
    p.add_argument("--curated", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    queue_bytes = args.queue.read_bytes()
    decisions_bytes = args.decisions.read_bytes()
    review = json.loads(decisions_bytes)
    if review["queue_sha256"] != hashlib.sha256(queue_bytes).hexdigest():
        raise ValueError("Review does not match queue")
    queue = json.loads(queue_bytes)
    decisions = review["decisions"]
    expected_ids = {f"R{i:02d}" for i in range(1, len(queue) + 1)}
    if set(decisions) != expected_ids:
        raise ValueError("Incomplete or unknown review IDs")
    bound = {}
    for index, target in enumerate(queue, 1):
        identifier = f"R{index:02d}"
        decision = decisions[identifier]
        if (decision["source"] != target["source"] or
                decision["band"] != target["band"] or
                abs(decision["target_x"] - target["x"]) > .01):
            raise ValueError(f"Review target mismatch: {identifier}")
        if decision["status"] not in {"accept", "fix", "reject", "uncertain"}:
            raise ValueError(f"Unreviewed status: {identifier}")
        bound.setdefault((target["source"], target["band"]), []).append(
            (identifier, target, decision))
    args.output.mkdir(parents=True)
    (args.output / "mask").mkdir()
    (args.output / "review").mkdir()
    points = json.loads((args.point_dataset / "manifest.json").read_text())["records"]
    point_lookup = {r["source"]: r for r in points if r["origin"] == "factory"}
    records = json.loads((args.curated / "manifest.json").read_text())
    report = []
    output_records = []
    for original in records:
        record = copy.deepcopy(original)
        key = (record["source"], record["band"])
        local_reviews = bound.get(key, [])
        old_mask = cv2.imread(record["curated_mask"], 0)
        band = band_image(point_lookup[record["source"]], record["band"])
        edited = False
        review_meta = []
        if local_reviews:
            proposal = cv2.imread(record["mask_path"], 0)
            count, labels, stats, centers = cv2.connectedComponentsWithStats(
                (proposal > 0).astype(np.uint8), 8)
            instances = []
            ridge = ridge_image(band)
            for edge in record["edges"]:
                selected = min(range(1, count), key=lambda i: (
                    abs(float(centers[i, 0]) - edge["x"]) +
                    abs(int(stats[i, cv2.CC_STAT_AREA]) - edge["area"]) /
                    max(edge["area"], 1)))
                if abs(float(centers[selected, 0]) - edge["x"]) > .2:
                    raise ValueError("Cannot reconstruct original component")
                observed = labels == selected
                repaired, _, _ = complete_component(observed, ridge)
                instances.append({"edge": copy.deepcopy(edge), "observed": observed,
                                  "mask": repaired, "reviews": []})
            reconstructed = np.maximum.reduce([instance["mask"] for instance in instances])
            if not np.array_equal(reconstructed, old_mask):
                raise ValueError("Original mask cannot be reproduced exactly")
            for identifier, target, decision in local_reviews:
                nearest = min(range(len(instances)), key=lambda i: (
                    abs(instances[i]["edge"]["x"] - target["x"]) +
                    abs(instances[i]["edge"]["area"] - target["area"]) /
                    max(target["area"], 1)))
                instances[nearest]["reviews"].append(identifier)
                instances[nearest]["edge"]["human_review"] = decision["status"]
                review_meta.append({"id": identifier, **decision})
                if decision["status"] == "reject":
                    instances[nearest]["mask"][:] = 0
                    edited = True
            for identifier, target, decision in local_reviews:
                if decision["status"] != "fix":
                    continue
                nearest = min(range(len(instances)), key=lambda i: (
                    abs(instances[i]["edge"]["x"] - target["x"]) +
                    abs(instances[i]["edge"]["area"] - target["area"]) /
                    max(target["area"], 1)))
                _, path = human_path(decision["points"], old_mask.shape, 1)
                selected = [nearest]
                observed_y = instances[nearest]["observed"].any(axis=1)
                # Merge a second reviewed proposal only if it is a disjoint
                # fragment on the same user-drawn path. Never merge by count.
                for i, candidate in enumerate(instances):
                    if i == nearest or not candidate["reviews"]:
                        continue
                    ys, xs = np.where(candidate["observed"])
                    if not len(ys):
                        continue
                    median_error = float(np.median(np.abs(xs - path[ys])))
                    overlap = np.count_nonzero(observed_y & candidate["observed"].any(axis=1))
                    if median_error <= 5 and overlap == 0:
                        selected.append(i)
                thickness = int(np.ceil(np.median(
                    [instances[i]["edge"]["thickness"] for i in selected])))
                replacement, _ = human_path(decision["points"], old_mask.shape, thickness)
                replaced = np.maximum.reduce([instances[i]["mask"] for i in selected])
                rest = [instance["mask"] for i, instance in enumerate(instances)
                        if i not in selected]
                unaffected = np.maximum.reduce(rest) if rest else np.zeros_like(old_mask)
                new_mask = np.maximum(unaffected, replacement)
                altered = old_mask != new_mask
                allowed = (replaced > 0) | (replacement > 0)
                if np.any(altered & ~allowed):
                    raise ValueError("Correction altered unrelated pixels")
                before = band.copy()
                before[old_mask > 0] = (0, 0, 255)
                after = band.copy()
                after[new_mask > 0] = (0, 210, 255)
                xs = np.flatnonzero(allowed.any(axis=0))
                x1, x2 = max(0, int(xs.min()) - 65), min(band.shape[1], int(xs.max()) + 66)
                comparison = np.concatenate(
                    [band[:, x1:x2], cv2.addWeighted(band, .5, before, .5, 0)[:, x1:x2],
                     cv2.addWeighted(band, .5, after, .5, 0)[:, x1:x2]], axis=1)
                cv2.imwrite(str(args.output / f"{identifier}_raw_before_after.png"), comparison)
                for i in selected:
                    instances[i]["mask"][:] = 0
                    instances[i]["edge"]["superseded_by"] = identifier
                instances.append({
                    "mask": replacement, "observed": replacement > 0, "reviews": [identifier],
                    "edge": {"x": round(float(np.mean(path)), 1),
                             "thickness": thickness, "human_review": "corrected",
                             "review_ids": sorted({r for i in selected for r in instances[i]["reviews"]}),
                             "centerline_points": decision["points"],
                             "merged_proposal_count": len(selected),
                             "area": int(np.count_nonzero(replacement)),
                             "first_y": 0, "last_y": old_mask.shape[0] - 1,
                             "added_rows": 0, "median_extension_ridge": None},
                })
                edited = True
                report.append({"id": identifier, "source": record["source"],
                               "band": record["band"], "split": record["point_split"],
                               "merged_proposals": len(selected),
                               "review_ids": instances[-1]["edge"]["review_ids"],
                               "changed_pixels": int(np.count_nonzero(altered)),
                               "thickness": thickness})
            result = np.maximum.reduce([instance["mask"] for instance in instances])
            record["edges"] = [instance["edge"] for instance in instances
                               if np.any(instance["mask"])]
            record["accepted_edges"] = len(record["edges"])
        else:
            result = old_mask
        stem = Path(original["curated_mask"]).stem
        mask_path = args.output / "mask" / f"{stem}.png"
        review_path = args.output / "review" / f"{stem}.jpg"
        if edited:
            cv2.imwrite(str(mask_path), result)
            display = band.copy()
            display[result > 0] = (0, 210, 255)
            cv2.imwrite(str(review_path), cv2.addWeighted(band, .6, display, .4, 0))
        else:
            shutil.copy2(original["curated_mask"], mask_path)
            shutil.copy2(original["curated_review"], review_path)
        record.update({
            "curated_mask": str(mask_path.resolve()), "curated_review": str(review_path.resolve()),
            "human_reviews": review_meta,
            "label_status": ("model_assisted_with_partial_human_review" if local_reviews
                             else "semi_automatic_provisional"),
        })
        output_records.append(record)
    (args.output / "manifest.json").write_text(json.dumps(output_records, indent=2), encoding="utf-8")
    (args.output / "human_decisions_snapshot.json").write_bytes(decisions_bytes)
    summary = {
        "queue_sha256": review["queue_sha256"],
        "decisions_sha256": hashlib.sha256(decisions_bytes).hexdigest(),
        "status_counts": dict(Counter(d["status"] for d in decisions.values())),
        "corrections": report, "changed_train_bands": sum(r["split"] == "train" for r in report),
        "changed_val_bands": sum(r["split"] == "val" for r in report),
    }
    (args.output / "application_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
