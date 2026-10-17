"""Smoke real BaseService wiring without PLC writes or camera acquisition."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'SERVICE'))
from src.service.check_disk_service_yolo import DiskCheckingService


def main():
    service = DiskCheckingService()
    model = service.disk_segmentor_yolo
    assert (model.input_width,model.input_height)==(512,512)
    assert (model.preprocess_mode,model.rgb_input,model.mask_decode,model.tile_merge)==('letterbox',True,'native','core')
    assert model.conf_threshold==.5 and model.iou_threshold==.8
    gallery = ROOT/'.work/HA_faint_20261001_v3_conf05_core'
    rows = []
    for path in sorted((gallery/'band_original').glob('*.png')):
        image = cv2.imread(str(path))
        actual,_ = model.segment_large_image_debug(image,model.conf_threshold,model.iou_threshold)
        expected = cv2.imread(str(gallery/'masks'/path.name),0)
        if not np.array_equal(actual,expected):
            raise AssertionError('Runtime config/model differs from evaluated gallery: '+path.name)
        cleaned = service.clean_mask(actual.copy(),service.min_disk_area)
        rows.append({'band':path.name,'gallery_pixel_equal':True,
                     'predicted_pixels':int(np.count_nonzero(actual)),
                     'cleaned_pixels':int(np.count_nonzero(cleaned)),
                     'removed_by_min_area':int(np.count_nonzero(actual)-np.count_nonzero(cleaned))})
        center = 1133 if path.stem=='Cam1_Image_20260918130452232_band12' else 738
        if path.stem in ['Cam1_Image_20260918130452232_band12',
                          'Cam2_Image_20260918133424033_band23','Cam2_Image_20260918133409702_band23',
                          'Cam2_Image_20260918133355656_band23','Cam2_Image_20260918133346095_band23']:
            before = np.any(actual[:,center-9:center+10]>0,axis=1)
            after = np.any(cleaned[:,center-9:center+10]>0,axis=1)
            assert np.array_equal(before,after),'Production min-area filter dropped targeted rim'
            rows[-1]['target_rows_after_clean'] = int(after.sum())
    assert len(rows)==40
    report = {'input':[512,512],'preprocess':'letterbox','rgb':True,'mask_decode':'native',
              'tile_merge':'core','conf':model.conf_threshold,'iou':model.iou_threshold,
              'production_min_area':service.min_disk_area,'bands':rows,
              'PLC_connected':service.plc_controller.client is not None,
              'limit':'Prepared band geometry only; not real camera/end-to-end validation.'}
    out = ROOT/'.work/segmentation_deploy_20261001/source_smoke.json'
    out.write_text(json.dumps(report,indent=2))
    print('40 bands pixel-equal; all5 targeted rims survive min-area:',service.min_disk_area)
    print('PLC connected:',report['PLC_connected'])


if __name__=='__main__':
    main()
