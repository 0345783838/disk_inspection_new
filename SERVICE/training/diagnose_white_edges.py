"""Read-only HA White diagnostics with actual detector crop/model/caliper flow.

Production configuration is loaded explicitly; no config, weights or backend is
modified. Threshold/caliper ablations are offline tests, not proposed truth counts.
"""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from decouple import Config, RepositoryEnv
import decouple

from training.smoke_local_white import normalize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def decode(encoded):
    return cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)


def overlay(image, mask, measurements=None, length_rate=.95):
    out = image.copy()
    select = mask > 0
    out[select] = (out[select]*.45+np.array((0, 215, 255))*.55).astype(np.uint8)
    for measurement in measurements or []:
        y = measurement['center'][1]
        length = int(length_rate*out.shape[1])
        cx = measurement['center'][0]
        half = (length-1)/2
        cv2.line(out, (round(cx-half), y), (round(cx+half), y), (255, 200, 0), 1)
        for i, pair in enumerate(measurement['pairs'], 1):
            point = np.rint((np.array(pair['e1']['point'])+np.array(pair['e2']['point']))/2).astype(int)
            cv2.circle(out, tuple(point), 3, (0, 0, 255), -1)
            cv2.putText(out, str(i), (int(point[0])-4, y-7), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ha', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--app-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    ha, runtime, app_config, output = (path.resolve() for path in (args.ha,args.runtime,args.app_config,args.output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config_path = runtime/'config/config.env'
    decouple.config = Config(RepositoryEnv(str(config_path)))
    # Model paths in the production env are relative to its runtime directory.
    os.chdir(runtime)
    from src.service.check_disk_service_yolo import DiskCheckingService
    service = DiskCheckingService()
    params = service._default_white_params()
    model = service.disk_segmentor_yolo
    segment = model.segment_large_image_debug
    clean = service.clean_mask
    measure = service.caliper.measure_with_params
    snapshots = []

    def capture_segment(image, confidence, iou):
        mask, auxiliary = segment(image, confidence, iou)
        snapshots.append({'image': image.copy(), 'raw': mask.copy(), 'measurements': []})
        return mask, auxiliary

    def capture_clean(mask, area):
        result = clean(mask, area)
        next(snapshot for snapshot in snapshots if 'clean' not in snapshot)['clean'] = result.copy()
        return result

    def capture_measure(mask, center, min_edge_distance, max_edge_distance, length_rate, thickness_list, debug_coordinates=False):
        result = measure(mask, center, min_edge_distance, max_edge_distance, length_rate, thickness_list, debug_coordinates)
        # Production measures band1 twice, followed by band2 twice.
        index = sum(len(snapshot['measurements']) for snapshot in snapshots)//2
        snapshots[index]['measurements'].append({'center': list(center), 'pairs': result['pairs']})
        return result

    model.segment_large_image_debug = capture_segment
    service.clean_mask = capture_clean
    service.caliper.measure_with_params = capture_measure
    records = []
    for camera in (1, 2):
        for source in sorted((ha/f'Cam{camera}').glob('*.bmp')):
            start = time.perf_counter()
            snapshots.clear()
            image = normalize(cv2.imread(str(source)), camera, app_config)
            # APP sends JPEG. This uses OpenCV's encoder, not byte-identical Emgu.
            ok, encoded = cv2.imencode('.jpg', image)
            if not ok: raise ValueError('JPEG encoding failed')
            api_input = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            response = service.check_disk_white(api_input.copy())
            stem = f'Cam{camera}_{source.stem}'
            folder = output/stem
            save(folder/'origin_rotated.png', api_input)
            save(folder/'production_result.png', decode(response.ResImg))
            bands = []
            for index, snapshot in enumerate(snapshots, 1):
                band_image, raw, filtered = snapshot['image'], snapshot['raw'], snapshot['clean']
                low_mask, _ = segment(band_image, .1, params.segment_iou)
                low_filtered = clean(low_mask.copy(), params.disk_min_area)
                removed = cv2.subtract(raw, filtered)
                save(folder/f'band{index}_original.png', band_image)
                for key, mask in [('raw', raw), ('filtered', filtered), ('removed', removed),
                                  ('conf01_raw', low_mask), ('conf01_filtered', low_filtered)]:
                    save(folder/f'band{index}_{key}_mask.png', mask)
                save(folder/f'band{index}_raw_overlay.png', overlay(band_image, raw))
                save(folder/f'band{index}_filtered_overlay.png', overlay(band_image, filtered, snapshot['measurements']))
                low_measurements, full_measurements = [], []
                counts = []
                for production in snapshot['measurements']:
                    center = tuple(production['center'])
                    def count(mask, minimum=params.caliper_min_edge_distance, length=params.caliper_length_rate):
                        return measure(mask, center, minimum, params.caliper_max_edge_distance,
                                       length, params.caliper_thickness_list)
                    alternative = count(low_filtered)
                    full = count(filtered, length=1.)
                    low_measurements.append({'center': list(center), 'pairs': alternative['pairs']})
                    full_measurements.append({'center': list(center), 'pairs': full['pairs']})
                    counts.append({'center': list(center), 'production': len(production['pairs']),
                                   'no_area_filter': len(count(raw)['pairs']),
                                   'min_edge1': len(count(filtered, minimum=1)['pairs']),
                                   'full_scan': len(full['pairs']),
                                   'conf01': len(alternative['pairs'])})
                save(folder/f'band{index}_conf01_overlay.png', overlay(band_image, low_filtered, low_measurements))
                save(folder/f'band{index}_fullscan_overlay.png', overlay(band_image, filtered, full_measurements, length_rate=1.))
                contours, _ = cv2.findContours(removed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                removed_components = [{'box': list(cv2.boundingRect(c)), 'contour_area': float(cv2.contourArea(c))} for c in contours]
                bands.append({'band': index, 'shape': list(raw.shape), 'raw_pixels': int(np.count_nonzero(raw)),
                              'filtered_pixels': int(np.count_nonzero(filtered)), 'removed_pixels': int(np.count_nonzero(removed)),
                              'conf01_pixels': int(np.count_nonzero(low_mask)), 'counts': counts,
                              'removed_components': removed_components})
            record = {'camera': camera, 'source': str(source), 'folder': stem, 'source_sha256': digest(source),
                      'result': response.model_dump(exclude={'ResImg'}), 'bands': bands, 'seconds': time.perf_counter()-start}
            records.append(record)
            (folder/'diagnostics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(stem, 'counts', [c['production'] for b in bands for c in b['counts']],
                  'conf01', [c['conf01'] for b in bands for c in b['counts']],
                  'removed pixels', sum(b['removed_pixels'] for b in bands), flush=True)
    report = {'scope': '20 HA White frames; actual source production geometry/models, loaded with APP runtime config; not ground-truth accuracy',
              'config_path': str(config_path), 'config_sha256': digest(config_path), 'params': params.model_dump(),
              'segment_model_sha256': digest(runtime/decouple.config('DISK_SEGMENT_MODEL_PATH')),
              'packaged_backend_sha256': digest(runtime/'main.exe'), 'records': records,
              'hardware_calls': False, 'production_files_modified': False,
              'jpeg_warning': 'OpenCV JPEG, not claimed byte-identical to Emgu APP encoding'}
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    esc = html.escape
    sections = ['''<!doctype html><meta charset="utf-8"><title>White HA — diagnostic</title>
<style>body{background:#161c24;color:#e6edf3;font:16px system-ui;margin:24px}a{color:#87cbff}img{width:100%;height:auto}figure{margin:12px 0;background:#26323f;padding:12px}table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #59677b;padding:8px}summary{padding:12px;cursor:pointer}.note{background:#392e20;padding:16px}</style>
<h1>White HA: mask model / area filter / caliper</h1>
<div class="note">Baseline uses APP plugin config, not SERVICE config. No training, config edit, model/backend build or hardware calls.
Counts are diagnostics, NOT exact disk ground truth. Conf0.1/min-edge1/full-scan are offline ablations, not deployed settings.
APP draws mask only in strips around two scanlines; lack of paint outside those strips is not proof of absent mask.
JPEG encoding here is OpenCV, not claimed byte-identical to APP Emgu.</div>''']
    sections.append('<p>Actual baseline parameters: '+esc(str(params.model_dump()))+'</p>')
    for record in records:
        stem = record['folder']
        sections.append(f'<h2>{esc(stem)}</h2><p><a href="{stem}/origin_rotated.png">Origin rotated</a> · <a href="{stem}/production_result.png">Production-rendered result</a> · <a href="{stem}/diagnostics.json">Details</a></p>')
        for band in record['bands']:
            index = band['band']
            sections.append(f'<h3>Band{index}: area filter removes {band["removed_pixels"]} pixels</h3><table><tr><th>Scanline y</th><th>Baseline</th><th>No area filter</th><th>Min edge1</th><th>Full scan</th><th>Conf0.1</th></tr>')
            for counts in band['counts']:
                sections.append('<tr>'+''.join(f'<td>{value}</td>' for value in [counts['center'][1],counts['production'],counts['no_area_filter'],counts['min_edge1'],counts['full_scan'],counts['conf01']])+'</tr>')
            sections.append('</table>')
            for key, title in [('original','Original band'),('raw_overlay','Model mask before filtering'),
                               ('filtered_overlay','After area filter + actual caliper marks'),('conf01_overlay','Offline conf0.1, same area/caliper'),
                               ('fullscan_overlay','Offline full width100% (not deployed)')]:
                path=f'{stem}/band{index}_{key}.png'
                sections.append(f'<figure><figcaption>{title}</figcaption><a href="{path}" target="_blank"><img loading="lazy" src="{path}"></a></figure>')
            sections.append(f'<p><a href="{stem}/band{index}_raw_mask.png">Raw mask</a> · <a href="{stem}/band{index}_filtered_mask.png">Filtered mask</a> · <a href="{stem}/band{index}_removed_mask.png">Removed pixels</a></p>')
    sections.append('<p><a href="report.json">Full diagnostics JSON</a></p>')
    (output/'index.html').write_text('\n'.join(sections), encoding='utf-8')
    print('Report:', output/'index.html', flush=True)


if __name__ == '__main__':
    main()
