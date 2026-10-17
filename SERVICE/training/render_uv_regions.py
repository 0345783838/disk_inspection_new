"""Project the EXACT borrowed White UV bands/scanlines back onto full UV frames.

This is a deterministic geometry visualization, not a new detector or generated scene.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.tools.normalize_factory_frames import normalize_frame


COLORS = {'crop': (255, 230, 30), 'band1': (60, 235, 65),
          'band2': (235, 70, 235), 'scan': (0, 240, 255)}


def ordered_quad(points):
    p = np.asarray(points, np.float32)
    s, d = p.sum(axis=1), np.diff(p, axis=1).ravel()
    return p[[np.argmin(s), np.argmin(d), np.argmax(s), np.argmax(d)]]


def inverse_crop_transform(points):
    # Identical output dimensions and corner convention to service.crop_by_4pts().
    quad = ordered_quad(points)
    w = int(max(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[2]-quad[3])))
    h = int(max(np.linalg.norm(quad[3]-quad[0]), np.linalg.norm(quad[2]-quad[1])))
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
    forward = cv2.getPerspectiveTransform(quad, dst)
    return np.linalg.inv(forward), (w, h), quad


def project(points, transform):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(1, -1, 2), transform)[0]


def normalized_to_raw(points, camera_id, raw_shape):
    p = np.asarray(points, np.float32)
    h, w = raw_shape[:2]
    if camera_id == 1:  # inverse CW90
        return np.column_stack((p[:, 1], h-1-p[:, 0])).astype(np.float32)
    if camera_id == 2:  # inverse CCW90
        return np.column_stack((w-1-p[:, 1], p[:, 0])).astype(np.float32)
    raise ValueError('camera_id must be 1 or 2')


def label(image, text, position, color, scale=.8):
    x, y = map(int, position)
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(4, min(x, image.shape[1]-w-8))
    y = max(h+6, min(y, image.shape[0]-baseline-4))
    cv2.rectangle(image, (x-4, y-h-4), (x+w+4, y+baseline+3), (12, 12, 12), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw(image, crop, bands, scans, camera, reference, orientation):
    out = image.copy()
    shade = image.copy()
    for i, band in enumerate(bands, 1):
        cv2.fillPoly(shade, [np.rint(band).astype(np.int32)], COLORS[f'band{i}'])
    out = cv2.addWeighted(out, .92, shade, .08, 0)
    cv2.polylines(out, [np.rint(crop).astype(np.int32)], True, COLORS['crop'], 4, cv2.LINE_AA)
    for i, (band, scan) in enumerate(zip(bands, scans), 1):
        color = COLORS[f'band{i}']
        cv2.polylines(out, [np.rint(band).astype(np.int32)], True, color, 5, cv2.LINE_AA)
        cv2.line(out, tuple(np.rint(scan[0]).astype(int)), tuple(np.rint(scan[1]).astype(int)), COLORS['scan'], 3, cv2.LINE_AA)
        center = np.mean(band, axis=0)
        label(out, f'BAND {i}: HSV + CALIPER', center + (12, -18), color)
    for row, (text, color) in enumerate([
        (f'CAM{camera} UV / {orientation} / White reference: {reference}', (255, 255, 255)),
        ('CYAN: White CropBox (perspective crop)', COLORS['crop']),
        ('GREEN: UV Band1  |  MAGENTA: UV Band2', COLORS['band1']),
        ('YELLOW: actual center scanline (95% width)', COLORS['scan']),
        ('MIXING confirmed; only 2 bands are checked, not the whole crop', (255, 255, 255))]):
        label(out, text, (20, 38+row*34), color)
    return out


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(str(path))


def render_reference(root, folder, name):
    report = json.loads((root / folder / 'report.json').read_text(encoding='utf-8'))
    records = []
    for record in report['cameras']:
        camera, geometry = record['camera'], record['reference']
        raw_path = Path(record['uv_path'])
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record['uv_sha256']:
            raise ValueError('UV source changed since evaluation')
        raw = cv2.imread(str(raw_path))
        normalized = normalize_frame(raw, camera)
        cropped = cv2.imread(str(root / folder / f'Cam{camera}' / 'uv_rectified.png'))
        inverse, (w, h), crop = inverse_crop_transform(ast.literal_eval(geometry['CropBox']))
        if cropped.shape[:2] != (h, w):
            raise ValueError('Crop transform does not match actual cached UV warp')
        band_rects, scan_lines, bounds = [], [], []
        for i in (1, 2):
            # Match int32 cast + exclusive numpy slicing used by the actual UV service.
            box = np.asarray(ast.literal_eval(geometry[f'UvBox{i}']), np.int32)
            x0, y0 = box[0]
            x1, y1 = min(int(box[2, 0]), w), min(int(box[2, 1]), h)
            band = np.array([[x0, y0], [x1-1, y0], [x1-1, y1-1], [x0, y1-1]], np.float32)
            cached = cv2.imread(str(root / folder / f'Cam{camera}' / f'band{i}_raw.png'))
            if not np.array_equal(cropped[y0:y1, x0:x1], cached):
                raise ValueError('Band rectangle does not match evaluated pixels')
            bw, bh = x1-x0, y1-y0
            length = int(.95*bw)
            cx, cy = bw//2, bh//2
            half = (length-1)/2
            scan = np.array([[x0+cx-half, y0+cy], [x0+cx+half, y0+cy]], np.float32)
            band_rects.append(band)
            scan_lines.append(scan)
            bounds.append({'band': i, 'rectified_slice_xyxy_exclusive': [int(x0), int(y0), x1, y1],
                           'scan_rectified': scan.tolist(), 'band_normalized': project(band, inverse).tolist()})
        normalized_bands = [project(band, inverse) for band in band_rects]
        normalized_scans = [project(scan, inverse) for scan in scan_lines]
        raw_map = lambda points: normalized_to_raw(points, camera, raw.shape)
        output = root / 'regions' / name / f'Cam{camera}'
        save(output / 'uv_full_normalized_regions.png', draw(normalized, crop, normalized_bands, normalized_scans, camera, name, 'NORMALIZED'))
        save(output / 'uv_full_raw_regions.png', draw(raw, raw_map(crop), [raw_map(b) for b in normalized_bands],
                                                    [raw_map(s) for s in normalized_scans], camera, name, 'RAW'))
        full_rect = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
        save(output / 'uv_full_rectified_regions.png', draw(cropped, full_rect, band_rects, scan_lines, camera, name, 'RECTIFIED'))
        records.append({'camera': camera, 'uv_path': str(raw_path), 'white_path': geometry['white_path'],
                        'raw_shape': list(raw.shape), 'normalized_shape': list(normalized.shape),
                        'rectified_size_wh': [w, h], 'bands': bounds,
                        'verified_actual_cached_band_pixels': True})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    report = {'first': render_reference(a.root, 'diagnostics_first', 'first'),
              'last': render_reference(a.root, 'v_candidates_last', 'last'),
              'user_confirmed_both_uv_images_are_mixing': True,
              'per_band_uv_counts_not_confirmed': True}
    (a.root / 'regions' / 'geometry_overlay.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Rendered 12 full-frame region overlays; exact cached band pixels verified.')


if __name__ == '__main__':
    main()
