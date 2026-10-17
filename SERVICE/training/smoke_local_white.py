"""Smoke-test the packaged White endpoint with raw factory frames; never call PLC APIs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def normalize(raw, camera_id, app_config):
    settings = {item.get('key'): item.get('value')
                for item in ET.parse(app_config).findall('./appSettings/add')}
    rotation = settings.get(f'Cam{camera_id}FrameRotation',
                            'Clockwise90' if camera_id == 1 else 'CounterClockwise90')
    codes = {'Clockwise90': cv2.ROTATE_90_CLOCKWISE,
             'CounterClockwise90': cv2.ROTATE_90_COUNTERCLOCKWISE,
             'Rotate180': cv2.ROTATE_180}
    if rotation == 'None':
        image = raw.copy()
    else:
        image = cv2.rotate(raw, codes[rotation])
    roi = settings.get(f'Cam{camera_id}NormalizedRoi', '').strip()
    if roi:
        x, y, w, h = map(int, roi.split(','))
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            raise ValueError('Configured ROI is outside normalized image')
        image = image[y:y+h, x:x+w].copy()
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ha', type=Path, required=True)
    parser.add_argument('--app-config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8386/ai/check_disk_white')
    parser.add_argument('--per-camera', type=int, default=1)
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.path != '/ai/check_disk_white':
        parser.error('Only the loopback production White endpoint is permitted')
    if args.per_camera < 1:
        parser.error('--per-camera must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for camera_id in (1, 2):
        sources = sorted(p for p in (args.ha / f'Cam{camera_id}').rglob('*')
                         if p.suffix.lower() in {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'})
        if not sources:
            raise ValueError(f'No Camera {camera_id} raw frames')
        for source in sources[:args.per_camera]:
            raw = cv2.imread(str(source))
            if raw is None:
                raise ValueError(f'Cannot decode {source}')
            normalized = normalize(raw, camera_id, args.app_config)
            # APP transmits JPEG too; OpenCV's encoder is not claimed byte-identical to Emgu.
            success, encoded = cv2.imencode('.jpg', normalized)
            if not success:
                raise ValueError('JPEG encode failed')
            boundary = 'local-white-smoke-boundary'
            payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image.jpg"\r\n'
                       'Content-Type: image/jpeg\r\n\r\n').encode() + encoded.tobytes() + f'\r\n--{boundary}--\r\n'.encode()
            request = urllib.request.Request(args.url, payload,
                                             {'Content-Type': f'multipart/form-data; boundary={boundary}'})
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            if result.get('Result') not in (0, 1, 2) or not result.get('ResImg'):
                raise ValueError('Invalid production White response')
            rendered = cv2.imdecode(np.frombuffer(base64.b64decode(result.pop('ResImg')), np.uint8), cv2.IMREAD_COLOR)
            if rendered is None:
                raise ValueError('Invalid result image')
            name = f'Cam{camera_id}_{source.stem}_White_result.png'
            if not cv2.imwrite(str(args.output / name), rendered):
                raise OSError('Cannot save White overlay')
            records.append({'camera': camera_id, 'source': str(source), 'raw_shape': list(raw.shape),
                            'normalized_shape': list(normalized.shape), 'result_image': name,
                            'seconds': time.perf_counter() - start, 'response': result})
            print(json.dumps(records[-1]), flush=True)
    report = {'endpoint': args.url, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'scope': 'Production White API only, no UV, no PLC; integration smoke, not an accuracy evaluation',
              'images': records}
    (args.output / 'smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
