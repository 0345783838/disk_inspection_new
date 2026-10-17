"""Export all HA White predictions, without label/ridge-based mask correction."""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from build_segment_crops import band_box, read_point_boxes, rectify

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference.yolo_segment.onnx_segmentor import OnnxSegmentor  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def overlay(image, mask):
    result = image.copy()
    selected = mask > 0
    result[selected] = np.rint(.45 * image[selected] +
                               .55 * np.array([0, 210, 255])).astype(np.uint8)
    return result


def titled(image, title):
    strip = np.full((34, image.shape[1], 3), 24, np.uint8)
    cv2.putText(strip, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .62, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([strip, image], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point-dataset", type=Path, required=True)
    p.add_argument("--ha", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conf", type=float, default=.2)
    p.add_argument("--preprocess", choices=["letterbox", "stretch"], default="letterbox")
    p.add_argument("--mask-decode", choices=["legacy", "native"], default="legacy")
    p.add_argument("--rgb", action="store_true")
    p.add_argument("--tile-merge", choices=["max", "core"], default="max")
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((args.point_dataset / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["origin"] == "factory"]
    lookup = {Path(r["source"]).resolve(): r for r in records}
    white = sorted(path.resolve() for camera in ("Cam1", "Cam2")
                   for path in (args.ha / camera).glob("*.bmp"))
    if set(white) != set(lookup):
        raise ValueError("HA White sources do not match the crop geometry manifest")
    for folder in ("raw_overlay", "band_original", "band_overlay", "band_compare", "masks"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    model = OnnxSegmentor(str(args.model.resolve()), ["disk"],
                          conf_thres=args.conf, iou_thres=.8,
                          preprocess_mode=args.preprocess, rgb_input=args.rgb,
                          mask_decode=args.mask_decode, tile_merge=args.tile_merge)
    outputs = []
    gallery = []
    x1, y1, x2, y2 = manifest["factory_crop_canonical_xyxy"]
    for source in white:
        record = lookup[source]
        if digest(source) != record["source_sha256"]:
            raise ValueError(f"Source changed: {source}")
        raw = cv2.imread(str(source))
        camera = record["camera"]
        rotation = manifest["factory_rotation"][camera]
        forward = (cv2.ROTATE_90_CLOCKWISE if rotation == "clockwise90"
                   else cv2.ROTATE_90_COUNTERCLOCKWISE)
        inverse = (cv2.ROTATE_90_COUNTERCLOCKWISE if rotation == "clockwise90"
                   else cv2.ROTATE_90_CLOCKWISE)
        canonical = cv2.rotate(raw, forward)
        image = canonical[y1:y2, x1:x2]
        owned = cv2.imread(record["image"])
        if not np.array_equal(image, owned):
            raise ValueError("Canonical raw crop differs from prepared image")
        rows = read_point_boxes(record["label"], image.shape[1], image.shape[0])
        warped, matrix, _, transformed = rectify(image, rows)
        warped_mask = np.zeros(warped.shape[:2], np.uint8)
        stem = f"{camera}_{source.stem}"
        band_outputs = []
        for band_name, upper, lower, direction in (
                ("band12", transformed[0], transformed[1], "bottom"),
                ("band23", transformed[1], transformed[2], "top")):
            bx1, by1, bx2, by2 = band_box(upper, lower, warped.shape, direction)
            band = warped[by1:by2, bx1:bx2]
            predicted, _ = model.segment_large_image_debug(band, args.conf, .8)
            painted = overlay(band, predicted)
            name = f"{stem}_{band_name}.png"
            write_image(out / "band_original" / name, band)
            write_image(out / "band_overlay" / name, painted)
            comparison = np.concatenate([
                titled(band, f"{stem} / {band_name} / original / {record['split']}"),
                titled(painted, f"prediction / conf={args.conf} / {args.preprocess} / {args.mask_decode} / {args.tile_merge} / {'RGB' if args.rgb else 'BGR'} / NO edge completion")], axis=0)
            write_image(out / "band_compare" / name, comparison)
            write_image(out / "masks" / name, predicted)
            warped_mask[by1:by2, bx1:bx2] = predicted
            band_outputs.append({"band": band_name, "rectified_box": [bx1, by1, bx2, by2],
                                 "mask_pixels": int(np.count_nonzero(predicted)),
                                 "comparison": f"band_compare/{name}"})
            gallery.append(f'<section><h2>{html.escape(stem)} / {band_name} / {record["split"]}</h2>'
                           f'<a href="band_compare/{name}"><img loading="lazy" src="band_compare/{name}"></a></section>')
        crop_mask = cv2.warpPerspective(warped_mask, np.linalg.inv(matrix),
                                       (image.shape[1], image.shape[0]),
                                       flags=cv2.INTER_NEAREST)
        canonical_mask = np.zeros(canonical.shape[:2], np.uint8)
        canonical_mask[y1:y2, x1:x2] = crop_mask
        raw_mask = cv2.rotate(canonical_mask, inverse)
        assert raw_mask.shape == raw.shape[:2]
        write_image(out / "raw_overlay" / f"{stem}.png", overlay(raw, raw_mask))
        outputs.append({"source": str(source), "source_sha256": record["source_sha256"],
                        "camera": camera, "split": record["split"], "bands": band_outputs,
                        "raw_overlay": f"raw_overlay/{stem}.png"})
        print(stem, record["split"], "done", flush=True)
    skipped = sorted(str(path.resolve()) for path in args.ha.rglob("*.bmp")
                     if path.resolve() not in lookup)
    report = {"model": str(args.model.resolve()), "model_sha256": digest(args.model),
              "conf": args.conf, "iou": .8, "preprocess": args.preprocess,
              "color_order": "RGB" if args.rgb else "BGR", "mask_decode": args.mask_decode,
              "tile_merge": args.tile_merge,
              "edge_completion": False,
              "geometry_source": "prepared point annotations, NOT end-to-end detector inference",
              "frame_count": len(outputs), "band_count": len(gallery),
              "skipped_non_white": skipped, "records": outputs}
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    page = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>HA segmentation inference</title>'
            '<style>body{background:#17191d;color:#eee;font:16px sans-serif;margin:24px}img{width:100%;height:auto}'
            'section{margin:32px 0}h2{font-size:18px}a{color:#7edaff}</style>'
            f'<h1>HA: {len(outputs)} White frames / {len(gallery)} bands</h1>'
            f'<p>ONNX candidate, {args.preprocess}, {report["color_order"]}, {args.mask_decode}, merge={args.tile_merge}, confidence {args.conf}. '
            'Yellow = predicted mask; top = original, bottom = prediction. '
            'No automatic edge extension. Click an image for full resolution.</p>'
            '<p>Crop geometry uses prepared point annotations, not detector predictions. '
            'Train and validation sources are both shown. UV1/UV2 are not in the White segmentation branch.</p>'
            + ''.join(gallery) + '</html>')
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "README.txt").write_text(
        "Mo index.html de xem 40 band (goc tren / prediction duoi).\n"
        "raw_overlay/: 20 anh raw voi mask chieu nguoc ve toa do camera.\n"
        "band_compare/: doi chieu tung band, band_original/: anh crop goc, "
        "band_overlay/: mask vang, masks/: mask nhi phan.\n"
        f"Model: {args.model.resolve()}\nConf={args.conf}, preprocess={args.preprocess}, {report['color_order']}, {args.mask_decode}, merge={args.tile_merge}.\n"
        "Khong tu noi canh. Geometry tu nhan point da chuan bi, khong phai end-to-end detect.\n"
        "UV1/UV2 khong thuoc nhanh segmentation White.\n", encoding="utf-8")
    print(out, f"{len(outputs)} frames / {len(gallery)} bands", flush=True)


if __name__ == "__main__":
    main()
