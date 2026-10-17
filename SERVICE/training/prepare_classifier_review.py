"""Old-model proposals on native runtime crops. Never train or confirm labels."""
import argparse
import ast
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SERVICE'))
from src.inference.yolo_classify.yolo_classifier import YoloClassifier


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(crops_root, model, training_model, output):
    if output.exists():
        raise FileExistsError('Use a new output directory; existing review is never overwritten.')
    assert sha256(model) == sha256(training_model), 'Training and deployed classifier differ'
    crop_manifest = crops_root / 'manifest.json'
    data = json.loads(crop_manifest.read_text(encoding='utf-8'))
    assert data['diagnostic_crops'] == 0 and data['production_crops'] == 2000
    classifier = YoloClassifier(str(model), ['ng', 'no_disk', 'ok'], batch_size=2)
    metadata = classifier.ort_sess.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(metadata['names'])
    labels = [names[i] for i in range(len(names))]
    assert labels == classifier.labels
    env = dict(line.split('=', 1) for line in (ROOT/'SERVICE/config/config.env').read_text().splitlines()
               if '=' in line and not line.lstrip().startswith('#'))
    env = {k.strip(): v.strip() for k, v in env.items()}
    assert float(env['POINT_CLASSIFY_CROP_WIDTH_RATIO']) == data['classify_crop_width_ratio'] == 1.8
    assert env['DISK_POINT_DETECT_MODEL_PATH'] == data['point_model']
    output.mkdir(parents=True)
    for label in labels:
        (output/'proposed'/label).mkdir(parents=True)
    (output/'context').mkdir()
    items, sources = [], []
    drift = 0
    rotations = {'Clockwise90': cv2.ROTATE_90_CLOCKWISE,
                 'CounterClockwise90': cv2.ROTATE_90_COUNTERCLOCKWISE,
                 'Rotate180': cv2.ROTATE_180}
    for frame in data['frames']:
        assert not frame['diagnostic_only'] and frame['production_classifier_calls'] == 4
        source = Path(frame['source'])
        frame_id = f"Cam{frame['camera']}_{source.stem}"
        frame_dir = (crops_root/frame['review']).parent
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(str(source))
        rotation = frame['normalization']['rotation']
        if rotation != 'None':
            image = cv2.rotate(image, rotations[rotation])
        roi = frame['normalization']['roi']
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        h, w = frame['rectified_shape'][:2]
        rectified = cv2.warpPerspective(image, np.array(frame['warp_matrix']), (w, h))
        sources.append({'source': str(source), 'sha256': sha256(source),
                        'camera': frame['camera'], 'normalization': frame['normalization']})
        shutil.copy2(frame_dir/'rectified_boxes.png', output/'context'/f'{frame_id}_all.png')
        crops = []
        for record in frame['crops']:
            assert record['production_input']
            crop = cv2.imread(str(frame_dir/record['crop']))
            x1, y1, x2, y2 = record['effective_box']
            rebuilt = rectified[y1:y2, x1:x2]
            if record['rotated_180']:
                rebuilt = cv2.rotate(rebuilt, cv2.ROTATE_180)
            assert np.array_equal(crop, rebuilt), f'Native crop mismatch: {frame_id} {record}'
            crops.append(crop)
        predictions = []
        for start in range(0, len(crops), classifier.batch_size):
            batch = classifier._preprocess_batch(crops[start:start+classifier.batch_size], (224, 224))
            pred = classifier.ort_sess.run(None, {classifier.ort_sess.get_inputs()[0].name: batch})[0]
            assert pred.shape == (len(batch), 3) and np.isfinite(pred).all()
            assert np.all(pred >= 0) and np.allclose(pred.sum(axis=1), 1, atol=1e-5)
            predictions.extend(pred.tolist())
        for record, crop, scores in zip(frame['crops'], crops, predictions):
            label = labels[int(np.argmax(scores))]
            confidence = max(scores)
            drift += int(label != record['label'] or abs(confidence-record['confidence']) > 1e-5)
            identifier = f"{frame_id}__{record['region']}__slot{record['index']:02d}"
            relative = f'proposed/{label}/{confidence:.6f}__{identifier}.png'
            source_crop = frame_dir/record['crop']
            shutil.copy2(source_crop, output/relative)
            assert sha256(source_crop) == sha256(output/relative)
            x1, y1, x2, y2 = record['effective_box']
            left, right = max(0, x1-(x2-x1)), min(w, x2+(x2-x1))
            context = rectified[y1:y2, left:right].copy()
            cv2.rectangle(context, (x1-left, 0), (x2-left-1, context.shape[0]-1), (0, 220, 255), 2)
            if record['rotated_180']:
                context = cv2.rotate(context, cv2.ROTATE_180)
            context_path = f'context/{identifier}.png'
            if not cv2.imwrite(str(output/context_path), context):
                raise OSError(identifier)
            items.append({'id': identifier, 'camera': frame['camera'], 'frame': frame_id,
                          'source': str(source), 'region': record['region'], 'slot': record['index'],
                          'image': relative, 'context': context_path,
                          'full_context': f'context/{frame_id}_all.png',
                          'shape': list(crop.shape), 'proposed_label': label, 'confidence': confidence,
                          'probabilities': dict(zip(labels, scores)), 'review_status': 'unreviewed',
                          'sha256': sha256(source_crop), 'source_crop': str(source_crop.resolve()),
                          'geometry': record})
        print(frame_id, dict(Counter(i['proposed_label'] for i in items if i['frame']==frame_id)), flush=True)
    items.sort(key=lambda i: (i['confidence'], i['id']))
    counts = dict(Counter(item['proposed_label'] for item in items))
    assert len(items) == 2000 and len({i['id'] for i in items}) == len(items)
    payload = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
               'label_state': 'model_proposals_only', 'training_authorized': False,
               'model': {'production_path': str(model.resolve()), 'training_export': str(training_model.resolve()),
                         'sha256': sha256(model), 'class_order': labels, 'metadata': metadata},
               'crop_manifest': str(crop_manifest.resolve()), 'crop_manifest_sha256': sha256(crop_manifest),
               'crop_contract': {k: data[k] for k in ('classify_crop_width_ratio', 'point_model',
                                'detector_roi', 'slot_completion', 'point_threshold', 'point_iou')},
               'preprocessing': 'runtime BGR->RGB; stretch cv2.INTER_LINEAR 224x224; float32 /255; CHW; batch=2',
               'native_png_only': True, 'class_counts': counts, 'prior_prediction_mismatches': drift,
               'sort_order': 'confidence_ascending', 'sources': sources, 'items': items}
    (output/'items.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'proposals.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id','camera','region','slot','proposed_label',
                                                   'confidence','image','review_status'])
        writer.writeheader()
        writer.writerows({k: item[k] for k in writer.fieldnames} for item in items)
    shutil.copy2(Path(__file__).with_name('classifier_label_review.html'), output/'index.html')
    print(json.dumps({'count': len(items), 'proposed': counts, 'confirmed': 0, 'prediction_drift': drift}))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crops', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--training-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.crops.resolve(), args.model.resolve(), args.training_model.resolve(), args.output.resolve())
