"""Verify packaged White/UV geometry and rendered results against an offline source run.

Only inspection HTTP routes are used; no PLC or camera routes are called.
The caller supplies an isolated runtime with its production config/model copies.
"""
import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import cv2
import numpy as np

from src.tools.normalize_factory_frames import normalize_frame


def post_image(route, image, fields=None, port=8387):
    boundary = 'uv-middle-smoke-boundary'
    data = bytearray()
    for name, value in (fields or {}).items():
        data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise ValueError('Image encoding failed')
    data.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="smoke.png"\r\nContent-Type: image/png\r\n\r\n'.encode())
    data.extend(encoded.tobytes())
    data.extend(f'\r\n--{boundary}--\r\n'.encode())
    request = urllib.request.Request(f'http://127.0.0.1:{port}/ai/{route}', bytes(data),
                                     {'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def assert_image(encoded, expected_path, actual_path, max_changed_pixels=0):
    actual = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    expected = cv2.imread(str(expected_path))
    if actual is not None:
        cv2.imwrite(str(actual_path), actual)
    if actual is None or expected is None or actual.shape != expected.shape:
        raise AssertionError(f'Rendered dimensions differ: {expected_path}')
    difference = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
    changed = int(np.any(difference, axis=2).sum())
    if changed > max_changed_pixels:
        raise AssertionError(f'Rendered pixels differ: {expected_path}; changed pixels={changed}')
    return {'sha256': hashlib.sha256(actual.tobytes()).hexdigest(), 'changed_pixels': changed,
            'max_allowed_changed_pixels': max_changed_pixels, 'max_channel_difference': int(difference.max())}


def assert_fields(actual, expected):
    """Compare all non-image API outputs, allowing only numerical roundoff."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('API field set changed')
        for key in expected:
            assert_fields(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError('API list length changed')
        for left, right in zip(actual, expected):
            assert_fields(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-4)
    elif actual != expected:
        raise AssertionError(f'API output changed: {actual!r} != {expected!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--report-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8387)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError(f'Port {args.port} already in use; do not interrupt another backend')
    report = json.loads((args.report_root / 'diagnostics_first/report.json').read_text(encoding='utf-8'))
    from src.service.check_disk_service_yolo import DiskCheckingService
    source = DiskCheckingService()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with (args.output.parent / 'backend_smoke.log').open('wb') as log:
        process = subprocess.Popen([str(runtime / 'main.exe')], cwd=runtime, stdout=log, stderr=log,
                                   env={**os.environ, 'DISK_API_PORT': str(args.port)},
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for _ in range(80):
                if process.poll() is not None:
                    raise RuntimeError('Backend exited; inspect backend_smoke.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/ai/check_service_status', timeout=1) as health:
                        if health.status == 200:
                            break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError('Backend did not become healthy')
            for record in report['cameras']:
                camera, reference = record['camera'], record['reference']
                folder = args.report_root / 'diagnostics_first' / f'Cam{camera}'
                white = normalize_frame(cv2.imread(reference['white_path']), camera)
                result = post_image('check_disk_white', white, port=args.port)
                fresh_white = source.check_disk_white(white.copy()).model_dump(exclude={'ResImg'})
                assert_fields({key: value for key, value in result.items() if key != 'ResImg'}, fresh_white)
                if result['Result'] != reference['white_result']:
                    raise AssertionError('White verdict changed')
                fields = ('CropBox', 'UvBox1', 'UvBox2', 'Mid1', 'Mid2')
                for field in fields:
                    np.testing.assert_allclose(ast.literal_eval(result[field]), ast.literal_eval(reference[field]), atol=1e-4)
                # A tiny rasterization difference is recorded, never hidden. All
                # inspection outputs above must match; UV rendering remains exact.
                white_render = assert_image(result['ResImg'], folder / 'white_result.png',
                                             args.output.parent / f'Cam{camera}_packaged_white.png', max_changed_pixels=16)
                geometry = dict(zip(('crop_box', 'uv_box_1', 'uv_box_2', 'mid_1', 'mid_2'),
                                    (result[field] for field in fields)))
                uv = cv2.imread(str(folder / 'uv_normalized.png'))
                uv_result = post_image('check_disk_uv', uv, {'uv_box': json.dumps(geometry)}, port=args.port)
                fresh_uv = source.check_disk_uv(uv.copy(), *geometry.values()).model_dump(exclude={'ResImg'})
                assert_fields({key: value for key, value in uv_result.items() if key != 'ResImg'}, fresh_uv)
                expected = record['candidates']['legacy']
                if uv_result['Result'] != expected['passed'] or uv_result['CountUvDisk'] != expected['reported_count']:
                    raise AssertionError('Packaged UV result differs from source')
                uv_render = assert_image(uv_result['ResImg'], folder / 'legacy_production_result.png', args.output.parent / f'Cam{camera}_packaged_uv.png')
                records.append({'camera': camera, 'white_result': result['Result'], 'geometry_matches_source': True,
                                'all_non_image_outputs_match_source': True,
                                'white_render_comparison': white_render, 'uv_render_comparison': uv_render,
                                'uv_result': uv_result['Result'], 'uv_reported_count': uv_result['CountUvDisk']})
                print(f'Cam{camera}: all White/UV outputs match; changed render pixels White={white_render["changed_pixels"]}, UV={uv_render["changed_pixels"]}', flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    assets = {str(path.relative_to(runtime)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((runtime / 'config').rglob('*')) if path.is_file()}
    args.output.write_text(json.dumps({'runtime': str(runtime), 'port': args.port, 'camera_or_plc_calls': False,
                                      'main_exe_sha256': hashlib.sha256((runtime / 'main.exe').read_bytes()).hexdigest(),
                                      'config_asset_sha256': assets,
                                      'cameras': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
