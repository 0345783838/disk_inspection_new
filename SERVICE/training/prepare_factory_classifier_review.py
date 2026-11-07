"""Create native PNG review crops from a saved factory audit and corrected point geometry.

Use serve_classifier_review.py for separate, persistent human labels. Predictions
are proposals, never ground truth; older reviews and models are left untouched.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from collections import Counter

import cv2
import decouple
import numpy as np

from training.audit_new_factory_images import digest, load, save, write_json
from src.tools.slot_geometry import complete_slot_rows, filter_off_grid_extras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit, output = args.audit.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((audit/'report.json').read_text(encoding='utf-8'))
    runtime = Path(report['runtime'])
    config = decouple.Config(decouple.RepositoryEnv(str(runtime/'config/config.env')))
    decouple.config = config
    os.chdir(runtime)
    from src.service.check_disk_service_yolo import DiskCheckingService
    service = DiskCheckingService()
    output.mkdir(parents=True)
    classifier = service.point_classification_model
    items, geometry_audit, scores_audit = [], [], []
    regions = ('row1_bottom', 'row2_top', 'row2_bottom', 'row3_top')
    for frame in report['white']:
        source = Path(frame['source'])
        image = load(source, frame['camera'], Path(report['app_config']))
        boxes = np.array([d['box'] for d in frame['raw_detections']], np.float32)
        rows = service.split_rows(boxes, expected_points_per_row=25)
        before_counts = list(map(len, rows))
        rows, removed = filter_off_grid_extras(rows)
        rows, inferred = complete_slot_rows(rows, image.shape, 25)
        rectified, matrix, _, quad = service.full_rectify_pipeline(image, rows[0], rows[2], .2, .1)
        warped_rows = [service.update_boxes_after_warp(row, matrix) for row in rows]
        identifier = f"Cam{frame['camera']}_{source.stem}"
        full_context = rectified.copy()
        sides = ((warped_rows[0],'bottom'),(warped_rows[1],'top'),
                 (warped_rows[1],'bottom'),(warped_rows[2],'top'))
        frame_items = []
        for face, (row, direction) in enumerate(sides):
            rects = service.get_line_boxes_ratio_shift(rectified, row, direction)
            crops = service.crop_boxes(rectified, rects, direction)
            probabilities = []
            for start in range(0, len(crops), classifier.batch_size):
                chunk = crops[start:start+classifier.batch_size]
                tensor = classifier._preprocess_batch(chunk, classifier.input_size)
                pred = classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:tensor})[0]
                probabilities.extend(pred.tolist())
            for slot, (rect, crop, probabilities_) in enumerate(zip(rects, crops, probabilities),1):
                label = classifier.labels[int(np.argmax(probabilities_))]
                confidence = max(probabilities_)
                item_id = f'{identifier}__{regions[face]}__slot{slot:02}'
                name = f'proposed/{label}/{confidence:.6f}__{item_id}.png'
                save(output/name, crop)
                x1,y1,x2,y2 = rect
                left,right = max(0,x1-(x2-x1)),min(rectified.shape[1],x2+(x2-x1))
                context = rectified[y1:y2,left:right].copy()
                cv2.rectangle(context,(x1-left,0),(x2-left-1,context.shape[0]-1),(0,220,255),2)
                if direction=='top':
                    context = cv2.rotate(context,cv2.ROTATE_180)
                context_name = f'context/{item_id}.png'
                save(output/context_name,context)
                cv2.rectangle(full_context,(x1,y1),(x2,y2),(0,220,255),1)
                cv2.putText(full_context,f'{face+1}:{slot}',(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1)
                item = {'id':item_id,'camera':frame['camera'],'frame':identifier,'source':str(source),
                    'region':regions[face],'slot':slot,'image':name,'context':context_name,
                    'full_context':f'context/{identifier}_all.jpg','shape':list(crop.shape),
                    'proposed_label':label,'confidence':confidence,'probabilities':dict(zip(classifier.labels,probabilities_)),
                    'review_status':'unreviewed','sha256':digest(output/name),'source_crop':str(output/name),
                    'geometry':{'box':rect,'rotated_180':direction=='top','point_filter_removed':removed}}
                frame_items.append(item)
        save(output/f'context/{identifier}_all.jpg',full_context)
        items.extend(frame_items)
        # Rebuilding must preserve exactly the same tensors/predictions on prior valid geometry.
        if frame['classifications']:
            errors = []
            for previous, current in zip(frame['classifications'],frame_items):
                if previous['label']!=current['proposed_label'] or not np.allclose(
                        list(previous['scores'].values()),list(current['probabilities'].values()),atol=1e-6,rtol=1e-5):
                    errors.append(current['id'])
            scores_audit.append({'frame':identifier,'mismatches':errors})
            if errors:
                raise ValueError('Rebuilt native crops changed baseline predictions: '+str(errors))
        geometry_audit.append({'frame':identifier,'rows_before':before_counts,'rows_after':list(map(len,rows)),
            'removed':removed,'inferred':inferred,'crop_quad':quad.tolist()})
        print(identifier,dict(Counter(i['proposed_label'] for i in frame_items)),'removed',len(removed),flush=True)
    items.sort(key=lambda i:(i['confidence'],i['id']))
    payload = {'schema_version':1,'label_state':'model_proposals_only','training_authorized':False,
        'training_next_step':'User requested supplementary training AFTER reviewing labels; no training from proposals.',
        'model':{'production_path':str(runtime/config('POINT_CLASSIFY_MODEL_PATH')),
                 'sha256':digest(runtime/config('POINT_CLASSIFY_MODEL_PATH')),'class_order':classifier.labels},
        'crop_manifest':str(audit/'report.json'),'crop_manifest_sha256':digest(audit/'report.json'),
        'crop_contract':{'classify_crop_width_ratio':service.classify_crop_width_ratio,
            'point_model':config('DISK_POINT_DETECT_MODEL_PATH'),'slot_count':25,'jpeg_transport':'OpenCV'},
        'native_png_only':True,'class_counts':dict(Counter(i['proposed_label'] for i in items)),
        'preprocessing':'native full crop, RGB stretch cv2.INTER_LINEAR 224x224 float32 /255; top faces rotated 180',
        'items':items,'geometry_audit':geometry_audit,'prior_valid_frame_replay':scores_audit,
        'split_requirement':'Group all crops from the same real capture cycle together; near-duplicate captures cannot cross splits.'}
    write_json(output/'items.json',payload)
    with (output/'proposals.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['id','camera','region','slot','proposed_label','confidence','image','review_status'])
        writer.writeheader()
        writer.writerows({k:item[k] for k in writer.fieldnames} for item in items)
    page = Path(__file__).with_name('classifier_label_review.html').read_text(encoding='utf-8')
    page = page.replace('HA — rà soát nhãn classify','ANH_18_10 — sửa nhãn classify')
    page = page.replace('HA — nhãn đề xuất từ classifier cũ','ANH_18_10 — nhãn để train bổ sung')
    page = page.replace('<option>ng</option>','<option selected>ng</option>')
    (output/'index.html').write_text(page,encoding='utf-8')
    shutil.copy2(runtime/config('POINT_CLASSIFY_MODEL_PATH'), output/'classifier_before_review.onnx')
    write_json(output/'review_preparation.json',{'items':len(items),'class_counts':payload['class_counts'],
        'baseline_prediction_mismatches':sum(len(r['mismatches']) for r in scores_audit),
        'old_reviews_modified':False,'production_model_modified':False,'human_labels_created':False})
    print('Review ready:',output,len(items),payload['class_counts'],flush=True)


if __name__=='__main__':
    main()
