"""Source-bound diagnosis of the five user-reported faint-edge bands."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from inference.yolo_segment.onnx_segmentor import OnnxSegmentor

ROOT = Path(__file__).resolve().parents[2]
CASES = ['Cam1_Image_20260918130452232_band12',
         'Cam2_Image_20260918133424033_band23',
         'Cam2_Image_20260918133409702_band23',
         'Cam2_Image_20260918133355656_band23',
         'Cam2_Image_20260918133346095_band23']


def main():
    output = ROOT / '.work/segmentation_faint_20261001/probe'
    output.mkdir(parents=True, exist_ok=False)
    curated = json.loads((ROOT / '.work/segmentation_finetune_20260930/curated_v3_reviewed/manifest.json').read_text())
    lookup = {r['camera']+'_'+Path(r['source']).stem+'_'+r['band']: r for r in curated}
    model = OnnxSegmentor(str(ROOT / '.work/segmentation_refine_20261001/runs/refine512_hard/weights/best.onnx'),
                          ['disk'], conf_thres=.5, iou_thres=.8,
                          preprocess_mode='letterbox', rgb_input=True, mask_decode='native')
    report = []
    for stem in CASES:
        record = lookup[stem]
        image = cv2.imread(str(ROOT / '.work/HA_refined_20261001_conf05_core/band_original' / (stem+'.png')))
        target = cv2.imread(record['curated_mask'], 0)
        core = cv2.imread(str(ROOT / '.work/HA_refined_20261001_conf05_core/masks' / (stem+'.png')), 0)
        maximum, _ = model.segment_large_image(image)
        start, end = (950, 1200) if stem.startswith('Cam1') else (600, 1000)
        def paint(mask):
            tinted = image.copy()
            tinted[mask > 0] = (0, 210, 255)
            return cv2.addWeighted(image, .5, tinted, .5, 0)
        review = np.concatenate([image[:, start:end], paint(target)[:, start:end],
                                 paint(core)[:, start:end], paint(maximum)[:, start:end]], axis=0)
        cv2.imwrite(str(output / (stem+'_roi_review.png')), review)
        cv2.imwrite(str(output / (stem+'_reference.png')), image[:, start:end])
        candidates = []
        crops, positions = model.split_image(image)
        for tile, (crop, (x1, x2)) in enumerate(zip(crops, positions)):
            if x2 <= start or x1 >= end:
                continue
            boxes, scores, _, masks = model.segment_objects_debug(crop, .01, .8)
            for box, score, mask in zip(boxes, scores, masks):
                if box[2]+x1 < start or box[0]+x1 > end:
                    continue
                box = box + [x1, 0, x1, 0]
                candidates.append({'tile': tile, 'score': float(score), 'box': box.tolist(),
                                   'mask_rows': int(np.any(mask, axis=1).sum()),
                                   'median_x': float(np.median(np.nonzero(mask)[1])+x1) if mask.any() else None})
        count, components, stats, _ = cv2.connectedComponentsWithStats((target > 0).astype(np.uint8), 8)
        edges = []
        for i in range(1, count):
            ys, xs = np.nonzero(components == i)
            if xs.max() >= start and xs.min() <= end:
                edges.append({'median_x': float(np.median(xs)), 'first_x': float(np.median(xs[ys==ys.min()])),
                              'last_x': float(np.median(xs[ys==ys.max()])), 'rows': int(np.unique(ys).size)})
        report.append({'case': stem, 'split': record['point_split'], 'roi_x': [start, end],
                       'gt_edges': edges, 'predictions_conf01': candidates,
                       'max_not_core_pixels': int(np.count_nonzero((maximum>0)&(core==0)))})
    (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
