"""Compare targeted user-reported edges at fixed confidence using native ONNX."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from probe_faint_edge_cases import CASES, ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from inference.yolo_segment.onnx_segmentor import OnnxSegmentor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    labels = json.loads((ROOT/'.work/segmentation_faint_20261001/curated_v4_faint_train/manifest.json').read_text())
    lookup = {r['camera']+'_'+Path(r['source']).stem+'_'+r['band']:r for r in labels}
    model = OnnxSegmentor(str(args.model.resolve()),['disk'],conf_thres=.5,iou_thres=.8,
                          preprocess_mode='letterbox',rgb_input=True,mask_decode='native',tile_merge='core')
    results = []
    for stem in CASES:
        r = lookup[stem]
        image = cv2.imread(str(ROOT/'.work/HA_refined_20261001_conf05_core/band_original'/(stem+'.png')))
        mask = cv2.imread(r['curated_mask'],0)
        count, ids, stats, _ = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
        center = 1133 if stem.startswith('Cam1') else 738
        index = min(range(1,count),key=lambda i:abs(np.median(np.nonzero(ids==i)[1])-center))
        target = ids==index
        best = None
        crops, positions = model.split_image(image)
        for tile, (crop,(x1,x2)) in enumerate(zip(crops,positions)):
            boxes,scores,_,masks = model.segment_objects_debug(crop,.01,.8)
            for box,score,pred in zip(boxes,scores,masks):
                full = np.zeros(mask.shape,bool)
                full[:,x1:x2] = pred>0
                overlap = np.count_nonzero(full&target)/max(1,np.count_nonzero(full|target))
                if overlap <= .01:
                    continue
                priority = (score>.5 and overlap>.3, overlap)
                previous = (best['confidence']>.5 and best['mask_iou']>.3, best['mask_iou']) if best else (-1,-1)
                if priority > previous:
                    best = {'tile':tile,'confidence':float(score),'mask_iou':float(overlap),
                            'mask_rows':int(np.any(pred,axis=1).sum()),'box':(box+[x1,0,x1,0]).tolist()}
        full, _ = model.segment_large_image(image)
        tolerant_target = cv2.dilate(target.astype(np.uint8),np.ones((1,5),np.uint8))>0
        hit_rows = np.any((full>0)&tolerant_target,axis=1)
        coverage = float(np.mean(hit_rows[target.any(axis=1)]))
        painted = image.copy()
        painted[full>0]=(0,210,255)
        cv2.imwrite(str(args.output/(stem+'.png')),cv2.addWeighted(image,.5,painted,.5,0))
        results.append({'case':stem,'split':r['point_split'],'target_center':center,'best_instance':best,
                        'core_conf05_row_coverage_tolerance2px':coverage,
                        'target_detected_conf05':best is not None and best['confidence']>.5 and best['mask_iou']>.3})
    (args.output/'metrics.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
