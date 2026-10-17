"""Build factory segmentation tiles using the SERVICE rectification/band geometry."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_point_boxes(path, width, height):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) != 75:
        raise ValueError(f"Expected 75 factory point boxes: {path}, got {len(lines)}")
    boxes = []
    for line in lines:
        cls, x, y, w, h = [float(value) for value in line.split()]
        if cls != 0:
            raise ValueError(f"Unexpected point class in {path}")
        boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height])
    return np.asarray(boxes, dtype=np.float32).reshape(3, 25, 4)


def order_quad_pts(pts):
    pts = np.asarray(pts, dtype=np.float32)
    sums = pts.sum(axis=1)
    differences = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(sums)], pts[np.argmin(differences)],
                     pts[np.argmax(sums)], pts[np.argmax(differences)]], dtype=np.float32)


def rectify(image, rows):
    row1, _, row3 = rows
    left_top = row1[np.argmin(row1[:, 0])]
    right_top = row1[np.argmax(row1[:, 2])]
    left_bottom = row3[np.argmin(row3[:, 0])]
    right_bottom = row3[np.argmax(row3[:, 2])]
    quad = order_quad_pts([[left_top[0], left_top[1]],
                           [right_top[2], right_top[1]],
                           [right_bottom[2], right_bottom[3]],
                           [left_bottom[0], left_bottom[3]]])
    center = quad.mean(axis=0)
    quad = center + (quad - center) * np.array([1.2, 1.1], dtype=np.float32)
    quad[:, 0] = np.clip(quad[:, 0], 0, image.shape[1] - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, image.shape[0] - 1)
    tl, tr, br, bl = quad
    width = max(1, int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))))
    height = max(1, int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))))
    destination = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, destination)
    warped = cv2.warpPerspective(image, matrix, (width, height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    transformed_rows = []
    for row in rows:
        corners = np.stack((row[:, [0, 1]], row[:, [2, 1]],
                            row[:, [2, 3]], row[:, [0, 3]]), axis=1)
        points = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 4, 2)
        transformed_rows.append(np.column_stack((points.min(axis=1), points.max(axis=1))))
    return warped, matrix, quad, transformed_rows


def band_box(upper_row, lower_row, image_shape, direction):
    upper_bottom = int(round(float(np.median(upper_row[:, 3]))))
    lower_top = int(round(float(np.median(lower_row[:, 1]))))
    gap = lower_top - upper_bottom
    if gap <= 0:
        raise ValueError("Point rows overlap after rectification")
    crop_height = max(1, int(round(gap * .35)))
    if direction == "bottom":
        y2 = lower_top
        y1 = y2 - crop_height
    else:
        y1 = upper_bottom
        y2 = y1 + crop_height
    y1 = max(0, y1)
    y2 = min(image_shape[0], y2)
    if y2 <= y1:
        raise ValueError("Invalid segmentation band")
    return [0, y1, image_shape[1], y2]


def tile_boxes(width, num_splits=6, overlap_ratio=.05):
    step = width // num_splits
    if step <= 0:
        raise ValueError("Rectified band too narrow")
    overlap = int(step * overlap_ratio)
    boxes = []
    x = 0
    while x < width and len(boxes) < num_splits:
        boxes.append([max(0, x - overlap), min(width, x + step + overlap)])
        x += step
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "images").mkdir(parents=True)
    (output / "review").mkdir()
    source_manifest = json.loads((args.point_dataset / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest["factory_crop_canonical_xyxy"] != [0, 0, 1944, 2200]:
        raise ValueError("Unexpected point dataset crop contract")
    records = []
    for record in source_manifest["records"]:
        if record["origin"] != "factory":
            continue
        image = cv2.imread(record["image"])
        if image is None:
            raise ValueError(record["image"])
        rows = read_point_boxes(record["label"], image.shape[1], image.shape[0])
        warped, matrix, quad, transformed = rectify(image, rows)
        band_specs = (("band12", transformed[0], transformed[1], "bottom"),
                      ("band23", transformed[1], transformed[2], "top"))
        review = cv2.resize(warped, (warped.shape[1] // 2, warped.shape[0] // 2))
        for band_name, upper, lower, direction in band_specs:
            x1, y1, x2, y2 = band_box(upper, lower, warped.shape, direction)
            cv2.rectangle(review, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 0), 2)
            band = warped[y1:y2, x1:x2]
            for tile_index, (tile_x1, tile_x2) in enumerate(tile_boxes(band.shape[1])):
                tile = band[:, tile_x1:tile_x2]
                stem = f"{record['camera']}_{Path(record['source']).stem}_{band_name}_tile{tile_index}"
                path = output / "images" / f"{stem}.png"
                if not cv2.imwrite(str(path), tile):
                    raise OSError(path)
                records.append({
                    "camera": record["camera"], "source": record["source"],
                    "source_sha256": record["source_sha256"], "point_split": record["split"],
                    "band": band_name, "tile": tile_index, "image": str(path),
                    "shape_hw": list(tile.shape[:2]),
                    "rectified_band_xyxy": [tile_x1, y1, tile_x2, y2],
                    "rectified_shape_hw": list(warped.shape[:2]),
                    "source_quad": quad.tolist(), "homography": matrix.tolist(),
                })
        preview_path = output / "review" / f"{record['camera']}_{Path(record['source']).stem}.jpg"
        cv2.imwrite(str(preview_path), review)
        print(record["camera"], Path(record["source"]).name,
              "rectified", warped.shape[:2], flush=True)
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Factory segmentation tiles:", len(records))


if __name__ == "__main__":
    main()
