"""Mine train-only weak tiles; shift real crop windows and add erased-edge negatives."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from build_segment_dataset import polygons_from_mask, stable_hash
from curate_segment_masks import band_image
from evaluate_edge_continuity import read_instances


def main():
    p = argparse.ArgumentParser()
    for name in ('dataset', 'crops', 'curated', 'point-dataset', 'model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for split in ('train', 'val'):
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    records = json.loads((args.dataset/'manifest.json').read_text())
    # Keep only one old-demo TRAIN crop per source. Preserve all held-out data.
    old_groups = set()
    retained = []
    for record in records:
        if record['origin'] == 'old' and record['split'] == 'train':
            if record['source_group'] in old_groups:
                continue
            old_groups.add(record['source_group'])
        updated = dict(record)
        for key, folder in [('image', 'images'), ('label', 'labels')]:
            target = output/folder/record['split']/Path(record[key]).name
            shutil.copy2(record[key], target)
            updated[key] = str(target)
        retained.append(updated)
    points = json.loads((args.point_dataset/'manifest.json').read_text())['records']
    point_lookup = {r['source']: r for r in points if r['origin']=='factory'}
    curated = json.loads((args.curated/'manifest.json').read_text())
    label_lookup = {(r['source'], r['band']): r for r in curated}
    crops = json.loads((args.crops/'manifest.json').read_text())
    crop_lookup = {(r['source'], r['band'], r['tile']): r for r in crops}
    bases = [r for r in retained if r['origin']=='factory' and r['split']=='train' and r['repeat']==0]
    model = YOLO(str(args.model.resolve()))
    mining = []
    hard = []
    predictions = model.predict([r['image'] for r in bases], stream=True, imgsz=320,
                                batch=4, rect=False, conf=.25, iou=.8, retina_masks=True,
                                device=0, verbose=False)
    for record, prediction in zip(bases, predictions):
        image = cv2.imread(record['image'])
        shape = image.shape[:2]
        targets = read_instances(record['label'], shape)
        predicted = [] if prediction.masks is None else prediction.masks.data.cpu().numpy()>0
        missing = 0
        for target in targets:
            area = target.sum()
            scores = [float(np.count_nonzero(target & candidate)/max(1,np.count_nonzero(target | candidate)))
                      for candidate in predicted]
            best_index = int(np.argmax(scores)) if scores else -1
            iou = scores[best_index] if scores else 0
            coverage = np.mean(np.any(target & predicted[best_index],axis=1)[target.any(axis=1)]) if best_index>=0 else 0
            if iou < .4 or coverage < .9:
                missing += 1
        mining.append({'source': record['source'], 'band': record['band'], 'tile':record['tile'],
                       'weak_instances':missing, 'objects':len(targets)})
        if missing:
            hard.append(record)
    band_cache = {}
    def get_band(record):
        key = (record['source'], record['band'])
        if key not in band_cache:
            band_cache[key] = (band_image(point_lookup[record['source']],record['band']),
                              cv2.imread(label_lookup[key]['curated_mask'],0).squeeze())
        return band_cache[key]
    added = []
    previews = []
    for record in hard:
        full_image, full_mask = get_band(record)
        crop = crop_lookup[(record['source'],record['band'],record['tile'])]
        x1, _, x2, _ = crop['rectified_band_xyxy']
        width = x2-x1
        for direction in (-1,1):
            new_x1 = int(np.clip(x1+direction*max(8,round(width*.055)),0,full_image.shape[1]-width))
            if new_x1 == x1:
                continue
            tile = full_image[:,new_x1:new_x1+width].copy()
            mask = full_mask[:,new_x1:new_x1+width]
            stem = Path(record['image']).stem+f'_hard_shift{direction:+d}'
            rng = np.random.default_rng(int(stable_hash(stem)[:8],16))
            changed = tile.astype(np.float32)
            gain = float(rng.uniform(.93,1.07))
            changed = changed*gain+rng.normal(0,.6,changed.shape)
            changed = np.clip(changed,0,255).astype(np.uint8)
            image_path = output/'images/train'/(stem+'.png')
            label_path = output/'labels/train'/(stem+'.txt')
            lines = polygons_from_mask(mask)
            cv2.imwrite(str(image_path),changed)
            label_path.write_text('\n'.join(lines)+'\n')
            added.append({**record,'origin':'hard_real','image':str(image_path),'label':str(label_path),
                          'objects':len(lines),'augmentation':'shift_crop_mild_sensor_variation',
                          'shifted_band_x1':new_x1,'geometry_exact_relative_to_source_mask':True})
            if len(previews)<12:
                painted = changed.copy()
                painted[mask>0]=(0,210,255)
                previews.append(np.concatenate([changed,cv2.addWeighted(changed,.5,painted,.5,0)],axis=1))
    # Small controlled negative set: 1 TRAIN tile per raw source. Never use val in train.
    negative_sources=set()
    for record in sorted(bases,key=lambda r:(r['source'],abs(r['tile']-3),r['band'])):
        if record['source'] in negative_sources:
            continue
        negative_sources.add(record['source'])
        full_image,full_mask=get_band(record)
        crop=crop_lookup[(record['source'],record['band'],record['tile'])]
        x1,_,x2,_=crop['rectified_band_xyxy']
        image=full_image[:,x1:x2].copy()
        mask=full_mask[:,x1:x2]
        erase=cv2.dilate(mask,np.ones((5,7),np.uint8))
        background=cv2.inpaint(image,erase,5,cv2.INPAINT_TELEA)
        stem=Path(record['image']).stem+'_empty_background'
        image_path=output/'images/train'/(stem+'.png')
        label_path=output/'labels/train'/(stem+'.txt')
        cv2.imwrite(str(image_path),background)
        label_path.write_text('')
        added.append({**record,'origin':'synthetic_negative','image':str(image_path),'label':str(label_path),
                      'objects':0,'scenario':'all_annotated_edges_erased','label_status':'constructed_negative_not_real_empty_tray'})
        if len(negative_sources)<=4:
            previews.append(np.concatenate([image,background],axis=1))
    final=retained+added
    val_sources={r['source'] for r in final if r['split']=='val' and r['origin']!='old'}
    assert all(r['source'] not in val_sources for r in final if r['split']=='train' and r['origin']!='old')
    (output/'manifest.json').write_text(json.dumps(final,indent=2))
    val_paths=[r['image'] for r in final if r['split']=='val' and r['origin']=='factory']
    (output/'factory_val.txt').write_text('\n'.join(val_paths)+'\n')
    (output/'disk_train.yaml').write_text(f'path: {output.as_posix()}\ntrain: images/train\nval: factory_val.txt\nnames:\n  0: disk\n')
    width=max(im.shape[1] for im in previews)
    cv2.imwrite(str(output/'augmentation_review.jpg'),np.concatenate([
        cv2.copyMakeBorder(im,0,0,0,width-im.shape[1],cv2.BORDER_CONSTANT) for im in previews]))
    summary={'counts':dict(Counter(r['origin']+'_'+r['split'] for r in final)),
             'hard_base_tiles':len(hard),'mining':mining,'checkpoint_selection':'HA val only',
             'negative_limit':'Constructed erasure negatives; not evidence of real NG accuracy.'}
    (output/'build_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='mining'},indent=2))


if __name__=='__main__':
    main()
