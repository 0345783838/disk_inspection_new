"""Compare packaged API against source on normalized HA frames, no PLC calls."""
import json
import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'SERVICE'))
from src.service.check_disk_service_yolo import DiskCheckingService
from src.tools.normalize_factory_frames import normalize_frame


def request_debug(image,params):
    boundary = 'disk-deployment-smoke-boundary'
    ok,png = cv2.imencode('.png',image)
    assert ok
    payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="params_json"\r\n\r\n'.encode()
               +params.model_dump_json().encode()+b'\r\n'
               +f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
               +png.tobytes()+f'\r\n--{boundary}--\r\n'.encode())
    req = urllib.request.Request('http://127.0.0.1:8386/ai/check_disk_debug',data=payload,
          headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req,timeout=90) as res:
        return json.loads(res.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'.work/segmentation_deploy_20261001/api_smoke.json')
    parser.add_argument('--all-ha', action='store_true')
    args = parser.parse_args()
    service = DiskCheckingService()
    params = service._default_white_params()
    sources = [(1,'Cam1/Image_20260918130452232.bmp'),(2,'Cam2/Image_20260918133424033.bmp')]
    if args.all_ha:
        ha = Path('D:/huynhvc/OTHERS/disk_checking/disk_checking/new_data/HA')
        sources = [(camera,f'Cam{camera}/{path.name}') for camera in (1,2)
                   for path in sorted((ha/f'Cam{camera}').glob('*.bmp'))]
    report = []
    for camera,name in sources:
        raw = cv2.imread(str(Path('D:/huynhvc/OTHERS/disk_checking/disk_checking/new_data/HA')/name))
        image = normalize_frame(raw,camera)
        expected = service.check_disk_debug(image.copy(),params).model_dump()
        actual = request_debug(image,params)
        assert expected==actual,'Packaged API does not match source: '+name
        report.append({'source':name,'camera':camera,'normalized_shape':list(image.shape),
                       'api_exact_source_equal':True,'result':actual['Result'],
                       'error_code':actual['ErrorCode'],
                       'geometry_status':actual['PointGeometryStatus'],
                       'inferred_slots':len(actual['InferredSlots']),
                       'segmentation_available':actual['SegmentImg'] is not None})
        print(name,'source/API equal','result',actual['Result'],'segment',actual['SegmentImg'] is not None)
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
