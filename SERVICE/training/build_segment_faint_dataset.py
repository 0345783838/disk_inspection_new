"""Fix four visually identified missing train labels and add limited AI ROI data.

Centerline anchors are agent visual annotations, not model pseudo-labels and
not independently human-confirmed. All validation images/labels are unchanged.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from audit_segment_dataset import rasterize
from build_segment_dataset import augment_factory, image_hash, polygons_from_mask

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.work/segmentation_faint_20261001'
ANCHORS = {
    'Image_20260918133346095': [741, 740, 740, 740, 740, 739, 739, 738],
    'Image_20260918133355656': [739, 738, 739, 738, 738, 738, 738, 738],
    'Image_20260918133409702': [738, 738, 738, 737, 737, 737, 737, 737],
    'Image_20260918133424033': [738, 737, 738, 738, 738, 738, 738, 737],
}


def trace(shape, ys, xs, width):
    h, w = shape
    path = PchipInterpolator(ys, xs)(np.arange(h))
    left = np.column_stack([path-width/2, np.arange(h)])
    right = np.column_stack([path[::-1]+width/2, np.arange(h)[::-1]])
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(np.concatenate([left, right])).astype(np.int32)], 255)
    return mask


def main():
    source = ROOT / '.work/segmentation_refine_20261001/dataset_hard_v1'
    output = WORK / 'dataset_faint_v2'
    if output.exists():
        raise FileExistsError(output)
    for split in ('train', 'val'):
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    original = json.loads((source/'manifest.json').read_text())
    curated_root = ROOT / '.work/segmentation_finetune_20260930/curated_v3_reviewed'
    curated = json.loads((curated_root/'manifest.json').read_text())
    crops = json.loads((ROOT/'.work/segmentation_finetune_20260930/factory_crops/manifest.json').read_text())
    crop_lookup = {(r['source'], r['band'], r['tile']): r for r in crops}
    mask_lookup = {(r['source'], r['band']): r for r in curated}
    corrected_root = WORK/'curated_v4_faint_train'
    (corrected_root/'mask').mkdir(parents=True, exist_ok=False)
    (corrected_root/'review').mkdir()
    additions, label_changes = {}, []
    for record in curated:
        stem = record['camera']+'_'+Path(record['source']).stem+'_'+record['band']
        mask = cv2.imread(record['curated_mask'], 0)
        if record['camera']=='Cam2' and record['band']=='band23' and Path(record['source']).stem in ANCHORS:
            assert record['point_split']=='train'
            added = trace(mask.shape, [0,40,80,120,160,200,240,286], ANCHORS[Path(record['source']).stem], 3)
            assert not np.any((mask>0)&(added>0)), 'Added edge already annotated'
            additions[(record['source'],record['band'])] = added
            mask = np.maximum(mask, added)
            record['faint_annotation'] = {'method':'agent_visual_anchors_pchip_no_model',
                                          'ys':[0,40,80,120,160,200,240,286],
                                          'xs':ANCHORS[Path(record['source']).stem], 'width':3,
                                          'independent_human_confirmed':False}
            record['accepted_edges'] += 1
            image = cv2.imread(str(ROOT/'.work/HA_refined_20261001_conf05_core/band_original'/(stem+'.png')))
            overlay = image.copy()
            overlay[mask>0] = (0,210,255)
            overlay[added>0] = (0,255,0)
            cv2.imwrite(str(corrected_root/'review'/(stem+'.png')),
                        np.concatenate([image,cv2.addWeighted(image,.5,overlay,.5,0)],axis=0))
        record['curated_mask'] = str(corrected_root/'mask'/(stem+'.png'))
        cv2.imwrite(record['curated_mask'], mask)
    (corrected_root/'manifest.json').write_text(json.dumps(curated,indent=2))
    records = []
    for record in original:
        updated = dict(record)
        for key, directory in [('image','images'),('label','labels')]:
            path = output/directory/record['split']/Path(record[key]).name
            shutil.copy2(record[key], path)
            updated[key] = str(path)
        key = (record['source'], record.get('band'))
        if record['split']=='train' and key in additions:
            crop = crop_lookup[(record['source'],record['band'],record['tile'])]
            x1,_,x2,_ = crop['rectified_band_xyxy']
            if record['origin']=='hard_real':
                x1 = record['shifted_band_x1']
                x2 = x1+cv2.imread(record['image']).shape[1]
            added_tile = additions[key][:,x1:x2]
            lines = polygons_from_mask(added_tile)
            if lines and record['origin']=='synthetic_negative':
                raise ValueError('False negative contains newly identified edge')
            if lines:
                text = Path(updated['label']).read_text().strip()
                Path(updated['label']).write_text(text+'\n'+'\n'.join(lines)+'\n')
                updated['objects'] += len(lines)
                updated['faint_label_correction'] = 'agent_visual_train_only'
                label_changes.append(updated['label'])
        records.append(updated)
    # Different crop positions retain exact image/mask geometry. Include Cam1
    # thin central train rims without reading the named Cam1 validation image.
    selected = [r for r in original if r['origin']=='factory' and r['split']=='train'
                and r['repeat']==0 and ((r['camera']=='Cam1' and r['band']=='band12' and r['tile']==3)
                                      or (r['camera']=='Cam2' and r['band']=='band23' and r['tile']==2))]
    for base in selected:
        key = (base['source'],base['band'])
        stem = base['camera']+'_'+Path(base['source']).stem+'_'+base['band']
        image = cv2.imread(str(ROOT/'.work/HA_refined_20261001_conf05_core/band_original'/(stem+'.png')))
        mask = cv2.imread(mask_lookup[key]['curated_mask'],0)
        crop = crop_lookup[(base['source'],base['band'],base['tile'])]
        x1,_,x2,_ = crop['rectified_band_xyxy']
        width = x2-x1
        for shift in (-60,0,60):
            start = int(np.clip(x1+shift,0,image.shape[1]-width))
            tile, target = image[:,start:start+width], mask[:,start:start+width]
            lines = polygons_from_mask(target)
            for variant in range(4):
                name = f'{stem}_faint_shift{shift:+d}_v{variant}'
                changed, augmentation = augment_factory(tile,variant,930+variant)
                image_path = output/'images/train'/(name+'.png')
                label_path = output/'labels/train'/(name+'.txt')
                cv2.imwrite(str(image_path),changed)
                label_path.write_text('\n'.join(lines)+'\n')
                records.append({**base,'origin':'faint_real','image':str(image_path),'label':str(label_path),
                                'shifted_band_x1':start,'objects':len(lines),'repeat':variant,
                                'augmentation':augmentation,'geometry_exact_relative_to_source_mask':True})
    ai_path = WORK/'ai/faint_reposition_v1.png'
    ai = cv2.imread(str(ai_path))
    assert ai.shape[:2]==(1062,1480), 'Visual anchors bind to exact generated image size'
    ai_masks = [trace(ai.shape[:2],[0,130,265,398,531,664,797,930,1061],
                      [718,717,716,715,714,713,712,711,710],10),
                trace(ai.shape[:2],[0,130,265,398,531,664,797,930,1061],
                      [939,939,939,940,941,942,943,945,947],34)]
    ai_source = next(r for r in selected if '20260918133424033' in r['source'])
    ai_annotation = {'image':str(ai_path),'sha256':image_hash(ai_path),
                     'source':ai_source['source'],'split':'train',
                     'annotation':'agent_visual_reannotation_of_generated_geometry_not_source_mask',
                     'independent_human_confirmed':False,
                     'usable_roi_xyxy':[350,0,1150,1062],
                     'two_visible_rims':'double metallic highlights treated as one physical rim'}
    (WORK/'ai/annotation.json').write_text(json.dumps(ai_annotation,indent=2))
    for start in (350,400,450):
        tile = cv2.resize(ai[:,start:start+700],(189,287),interpolation=cv2.INTER_AREA)
        # Separate instance polygons, never union close rims into one object.
        targets = [cv2.resize(m[:,start:start+700],(189,287),interpolation=cv2.INTER_NEAREST) for m in ai_masks]
        lines = sum([polygons_from_mask(m) for m in targets],[])
        assert len(lines)==2
        for variant in range(4):
            name = f'ai_faint_reposition_roi{start}_v{variant}'
            changed, augmentation = augment_factory(tile,variant,1110+variant)
            image_path = output/'images/train'/(name+'.png')
            label_path = output/'labels/train'/(name+'.txt')
            cv2.imwrite(str(image_path),changed)
            label_path.write_text('\n'.join(lines)+'\n')
            records.append({**ai_source,'origin':'ai_generated','image':str(image_path),'label':str(label_path),
                            'objects':2,'repeat':variant,'augmentation':augmentation,
                            'generated_source':str(ai_path),'generated_sha256':image_hash(ai_path),
                            'label_status':'agent_visual_ai_not_human_confirmed'})
        if start==400:
            painted = tile.copy()
            painted[np.maximum.reduce(targets)>0]=(0,210,255)
            cv2.imwrite(str(WORK/'ai/usable_roi_label_review.png'),np.concatenate([tile,painted],axis=1))
    val_sources = {r['source'] for r in records if r['split']=='val' and r['origin']!='old'}
    assert all(r['source'] not in val_sources for r in records if r['split']=='train' and r['origin']!='old')
    for before, after in zip(original,records):
        if before['split']=='val':
            assert image_hash(before['image'])==image_hash(after['image'])
            assert image_hash(before['label'])==image_hash(after['label'])
    for r in records:
        image = cv2.imread(r['image'])
        rasterize(r['label'],image.shape[:2])
    paths = [r['image'] for r in records if r['origin']=='factory' and r['split']=='val']
    (output/'factory_val.txt').write_text('\n'.join(paths)+'\n')
    (output/'disk_train.yaml').write_text(f'path: {output.as_posix()}\ntrain: images/train\nval: factory_val.txt\nnames:\n  0: disk\n')
    (output/'manifest.json').write_text(json.dumps(records,indent=2))
    report = {'counts':dict(Counter(r['origin']+'_'+r['split'] for r in records)),
              'corrected_train_bands':len(additions),'corrected_tile_labels':len(label_changes),
              'validation_images_and_labels_byte_identical':True,
              'AI_fraction_train':12/sum(r['split']=='train' for r in records),
              'annotation_limit':'Agent visual traces; new annotations not independently user-confirmed.'}
    (output/'build_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
