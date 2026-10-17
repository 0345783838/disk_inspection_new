"""Offline HSV/caliper diagnostics, using actual but unpaired White geometry. Run from SERVICE."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from src.tools.normalize_factory_frames import normalize_frame

CANDIDATES = {
    'legacy': ((107, 35, 40), (119, 150, 250)),
    'v255_only': ((107, 35, 40), (119, 150, 255)),
    's255_only': ((107, 35, 40), (119, 255, 250)),
    'sv255': ((107, 35, 40), (119, 255, 255)),
    's20_v30': ((107, 20, 30), (119, 255, 255)),
    'h105_125_s20_v30': ((105, 20, 30), (125, 255, 255)),
    'h100_130_s20_v30': ((100, 20, 30), (130, 255, 255)),
    'h107_119_s15_v35': ((107, 15, 35), (119, 255, 255)),
    'h105_125_s35_v40': ((105, 35, 40), (125, 255, 255)),
    'h105_125_s35_v60': ((105, 35, 60), (125, 255, 255)),
    'h105_125_s35_v80': ((105, 35, 80), (125, 255, 255)),
    'h105_125_s35_v100': ((105, 35, 100), (125, 255, 255)),
    'h105_125_s50_v60': ((105, 50, 60), (125, 255, 255)),
    'h105_125_s50_v80': ((105, 50, 80), (125, 255, 255)),
    'h107_119_s35_v80': ((107, 35, 80), (119, 255, 255)),
    'h107_119_s35_v55': ((107, 35, 55), (119, 150, 255)),
    'h105_125_s35_v55': ((105, 35, 55), (125, 150, 255)),
    'h105_125_s45_v55': ((105, 45, 55), (125, 150, 255)),
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(str(path))

def decode(value):
    return cv2.imdecode(np.frombuffer(base64.b64decode(value), np.uint8), cv2.IMREAD_UNCHANGED)

def band_details(service, band, lower, upper, band_index):
    raw = service.preprocess_uv_image_hsv(band, tuple(lower), tuple(upper))
    mask = service.remove_mask_noise_uv(raw, service.uv_min_disk_area)
    kwargs = dict(length_rate=.95, max_edge_distance=50, min_edge_distance=5, thickness_list=[1, 3]) if band_index == 1 else dict(
        length_rate=service.caliper.length_rate, max_edge_distance=service.caliper.pair_max_gap,
        min_edge_distance=service.caliper.min_edge_distance, thickness_list=service.caliper.thickness_list)
    center = (band.shape[1] // 2, band.shape[0] // 2)
    result = service.get_caliper_result_debug(mask, center=center, **kwargs)
    overlay = band.copy()
    active = mask > 0
    overlay[active] = (overlay[active] * .5 + np.array([0, 230, 0]) * .5).astype(np.uint8)
    cv2.line(overlay, (0, center[1]), (band.shape[1]-1, center[1]), (0, 255, 255), 1)
    mids = []
    for index, pair in enumerate(result['pairs'], 1):
        mid = (np.array(pair['e1']['point']) + np.array(pair['e2']['point'])) / 2
        mids.append(mid.tolist())
        cv2.circle(overlay, tuple(np.rint(mid).astype(int)), 4, (0, 0, 255), -1)
        cv2.putText(overlay, str(index), (int(mid[0])-5, max(15, center[1]-9)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 100) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 15) & (hsv[..., 2] >= 30)
    n = max(1, int(blue.sum()))
    stats = {'raw_pairs': len(result['pairs']), 'midpoints': mids,
             'pair_widths': [float(p['distance']) for p in result['pairs']],
             'mask_pixels': int(active.sum()), 'raw_mask_pixels': int(np.count_nonzero(raw)),
             'mask_center_pixels': int(np.count_nonzero(mask[center[1]])), 'caliper': kwargs, 'shape': list(band.shape),
             'exploratory_blue_pixels': int(blue.sum()),
             'legacy_exclusion_among_exploratory_blue': {
                 's_above150_fraction': float(np.count_nonzero(blue & (hsv[..., 1] > 150)) / n),
                 's_below35_fraction': float(np.count_nonzero(blue & (hsv[..., 1] < 35)) / n),
                 'v_above250_fraction': float(np.count_nonzero(blue & (hsv[..., 2] > 250)) / n),
                 'v_below40_fraction': float(np.count_nonzero(blue & (hsv[..., 2] < 40)) / n)},
             'hsv_blue_quantiles': [np.percentile(hsv[..., d][blue], [1, 10, 50, 90, 99]).tolist()
                                    if blue.any() else [] for d in range(3)]}
    # Diagnostics ONLY. These UV-specific ablations do not mutate the shared caliper or runtime.
    ablations = {}
    for name, override in [('full_scan', {'length_rate': 1.0}),
                           ('thin_edges', {'min_edge_distance': 1}),
                           ('full_thin_scan', {'length_rate': 1.0, 'min_edge_distance': 1})]:
        settings = {**kwargs, **override}
        alternate = service.get_caliper_result_debug(mask, center=center, **settings)
        ablations[name] = {'raw_pairs': len(alternate['pairs']), 'settings': settings}
    stats['caliper_ablations_not_deployed'] = ablations
    return stats, mask, overlay

def title(image, label):
    canvas = np.zeros((image.shape[0] + 30, image.shape[1], 3), np.uint8)
    canvas[30:] = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    return canvas

def main():
    from src.service.check_disk_service_yolo import DiskCheckingService
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ha', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--white-reference', choices=['first', 'last'], default='last')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    service = DiskCheckingService()
    records = []
    for camera in (1, 2):
        whites = sorted((args.ha / f'Cam{camera}').glob('*.bmp'))
        uv_path = next((args.ha / f'UV{camera}').glob('*.bmp'))
        white_path = whites[0 if args.white_reference == 'first' else -1]
        white = normalize_frame(cv2.imread(str(white_path)), camera)
        uv = normalize_frame(cv2.imread(str(uv_path)), camera)
        if white.shape != uv.shape:
            raise ValueError('White and UV shapes differ')
        camera_output = args.output / f'Cam{camera}'
        camera_output.mkdir(exist_ok=True)
        cache = camera_output / 'white_geometry.json'
        if cache.exists():
            geometry = json.loads(cache.read_text(encoding='utf-8'))
            if geometry['white_sha256'] != digest(white_path):
                raise ValueError('Cached White differs; use fresh output')
        else:
            response = service.check_disk_white(white)
            geometry = {'white_path': str(white_path), 'white_sha256': digest(white_path), 'white_result': response.Result,
                        **{key: getattr(response, key) for key in ('CropBox', 'UvBox1', 'UvBox2', 'Mid1', 'Mid2')}}
            if not geometry['CropBox']:
                raise ValueError('White did not return geometry')
            cache.write_text(json.dumps(geometry, indent=2), encoding='utf-8')
            save(camera_output / 'white_result.png', decode(response.ResImg))
        save(camera_output / 'uv_normalized.png', uv)
        cropped = service.crop_by_4pts(uv, np.array(ast.literal_eval(geometry['CropBox']), np.float32))
        save(camera_output / 'uv_rectified.png', cropped)
        bands = []
        for index in (1, 2):
            box = np.array(ast.literal_eval(geometry[f'UvBox{index}']), np.int32)
            band = cropped[box[0, 1]:box[2, 1], box[0, 0]:box[2, 0]].copy()
            bands.append(band)
            save(camera_output / f'band{index}_raw.png', band)
        record = {'camera': camera, 'uv_path': str(uv_path), 'uv_sha256': digest(uv_path), 'reference': geometry,
                  'geometry_is_unpaired': True, 'white_counts': [len(ast.literal_eval(geometry[f'Mid{i}'])) for i in (1, 2)], 'candidates': {}}
        for name, (lower, upper) in CANDIDATES.items():
            started = time.perf_counter()
            stats, sheets = [], []
            for index, band in enumerate(bands, 1):
                detail, mask, overlay = band_details(service, band, lower, upper, index)
                stats.append(detail)
                save(camera_output / f'{name}_band{index}_mask.png', mask)
                save(camera_output / f'{name}_band{index}_overlay.png', overlay)
                sheets += [title(band, f'CAM{camera} band{index} RAW'), title(mask, f'{name} HSV={lower}..{upper}'),
                           title(overlay, f'current caliper raw count={detail["raw_pairs"]} (not CountUvDisk)')]
            response = service._check_uv(uv.copy(), geometry['CropBox'], geometry['UvBox1'], geometry['UvBox2'],
                                         geometry['Mid1'], geometry['Mid2'], lower, upper, service.uv_min_disk_area)
            save(camera_output / f'{name}_production_result.png', decode(response.ResImg))
            save(camera_output / f'{name}_bands_comparison.png', np.vstack(sheets))
            record['candidates'][name] = {'lower': lower, 'upper': upper, 'bands': stats,
                                          'passed': response.Result, 'reported_count': response.CountUvDisk,
                                          'seconds': time.perf_counter() - started}
            print(f'CAM{camera} {name}: raw={[s["raw_pairs"] for s in stats]}, reported={response.CountUvDisk}, pass={response.Result}', flush=True)
        records.append(record)
    report = {'scope': '2 single UV captures, borrowed White geometry; NO paired tray or NG/negative ground truth',
              'raw_count_not_reported_count': True,
              'warning': 'CountUvDisk returns 0 for accepted homogeneous tray; zero is not evidence of no masks',
              'config_sha256': digest(Path('config/config.env')), 'service_sha256': digest(Path('src/service/check_disk_service_yolo.py')),
              'caliper_sha256': digest(Path('src/tools/caliper_advanced_new.py')), 'cameras': records}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
