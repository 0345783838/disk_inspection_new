"""Compare runtime ONNX predictions on HA and the untouched legacy val split."""
import argparse
import ast
import csv
import hashlib
import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'SERVICE'))
from src.inference.yolo_classify.yolo_classifier import YoloClassifier
from classifier_metrics import classification_metrics, LABELS

EXTENSIONS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_records(root):
    records = []
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == LABELS
    for label in LABELS:
        for path in sorted((root/label).rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                records.append({'id':f'legacy_{label}_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}',
                                'label':label, 'path':str(path.resolve())})
    return records


def corrected_legacy_records(root, manifest):
    records = legacy_records(root)
    if not manifest.get('label_correction_protocol'):
        return records
    overrides = {r['id']: r for r in manifest['records'] if r.get('origin') == 'legacy_val'}
    if set(overrides) != {r['id'] for r in records}:
        raise ValueError('Corrected legacy validation must represent the complete original val set')
    for record in records:
        corrected = overrides[record['id']]
        if sha256(Path(record['path'])) != corrected['sha256']:
            raise ValueError('Legacy source crop changed')
        record['label_before_correction'] = record['label']
        record['label'] = corrected['label']
        record['post_evaluation_review'] = corrected.get('post_evaluation_review')
    return records


def predict(model_path, records):
    classifier = YoloClassifier(str(model_path), LABELS, batch_size=32)
    metadata = classifier.ort_sess.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(metadata['names'])
    assert [names[i] for i in range(3)] == LABELS
    outputs = []
    for start in range(0,len(records),32):
        batch_records = records[start:start+32]
        images = [cv2.imread(r['path']) for r in batch_records]
        if any(image is None for image in images):
            raise ValueError('Unreadable evaluation image')
        batch = classifier._preprocess_batch(images,(224,224))
        probabilities = classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:batch})[0]
        assert probabilities.shape==(len(images),3) and np.isfinite(probabilities).all()
        assert (probabilities>=0).all() and np.allclose(probabilities.sum(1),1,atol=1e-5)
        for scores in probabilities:
            outputs.append({'label':LABELS[int(np.argmax(scores))], 'confidence':float(np.max(scores)),
                            'probabilities':dict(zip(LABELS,map(float,scores)))})
        if start%512==0:
            print(model_path.name, start+len(images), '/', len(records), flush=True)
    targets = [LABELS.index(r['label']) for r in records]
    metrics = classification_metrics(targets,[LABELS.index(p['label']) for p in outputs])
    metrics['confidence_below_0_5'] = sum(p['confidence']<.5 for p in outputs)
    metrics['confident_errors_0_5'] = sum(p['confidence']>=.5 and p['label']!=r['label'] for r,p in zip(records,outputs))
    return metrics, outputs


def torch_onnx_parity(pt_path, onnx_path, records):
    import onnx
    import torch
    from ultralytics import YOLO
    onnx.checker.check_model(str(onnx_path))
    torch.set_num_threads(4)
    model = YOLO(str(pt_path)).model.float().eval()
    classifier = YoloClassifier(str(onnx_path), LABELS)
    max_error, argmax_errors = 0., 0
    # Exercise both single and production batch sizes, plus a larger dynamic batch.
    for count in (1,2,32):
        images = [cv2.imread(r['path']) for r in records[:count]]
        batch = classifier._preprocess_batch(images,(224,224))
        with torch.inference_mode():
            result = model(torch.from_numpy(batch))
            torch_probs = (result[0] if isinstance(result,tuple) else result).numpy()
        onnx_probs = classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:batch})[0]
        max_error = max(max_error,float(np.max(np.abs(torch_probs-onnx_probs))))
        argmax_errors += int(np.sum(torch_probs.argmax(1)!=onnx_probs.argmax(1)))
    if argmax_errors or max_error>1e-3:
        raise ValueError(f'Export parity failed: {max_error}, {argmax_errors}')
    return {'batch_sizes':[1,2,32], 'max_probability_error':max_error,'argmax_mismatches':argmax_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--candidate-pt',type=Path)
    parser.add_argument('--include-test',action='store_true',help='Only AFTER candidate selection is locked')
    parser.add_argument('--selection',type=Path,help='Required immutable pre-test selection when --include-test is used')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.include_test:
        if not args.selection or not args.candidate:
            raise ValueError('Lock a candidate first; --selection is required before viewing test')
        selection=json.loads(args.selection.read_text(encoding='utf-8'))
        reviewed_protocol=selection.get('label_correction_protocol',{})
        assert (selection['selection_locked_before_ha_test'] or
                (reviewed_protocol.get('test_previously_viewed_and_labels_reviewed') and
                 selection.get('selection_locked_before_new_candidate_ha_test')))
        assert selection['onnx_sha256']==sha256(args.candidate)
        assert Path(selection['dataset']).resolve()==args.data.resolve()
    if args.output.exists():
        raise FileExistsError('Evaluation outputs are never overwritten')
    args.output.mkdir(parents=True)
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    assert sha256(args.baseline)==manifest['old_model']['sha256']
    splits = ['val']+(['train','test'] if args.include_test else [])
    domains = {f'ha_{s}':[dict(r,path=str((args.data/r['image']).resolve())) for r in manifest['records']
                          if r['split']==s and r.get('origin','ha')=='ha']
               for s in splits}
    domains['legacy_val'] = corrected_legacy_records(args.legacy/'val',manifest)
    report = {'class_order':LABELS,'preprocessing':'runtime RGB stretch cv2.INTER_LINEAR 224; float32 /255',
              'dataset':str(args.data.resolve()),'selection_sha256':sha256(args.selection) if args.selection else None,
              'test_used_for_training_or_selection':False,'legacy_split':'original val (never copied into training)',
              'ha_split_by_frame':manifest['split_by_frame'],'models':{},'domains':{}}
    if manifest.get('label_correction_protocol'):
        report['label_correction_protocol']=manifest['label_correction_protocol']
        report['validation_safety_gates']=manifest['validation_safety_gates']
        report['test_limitation']='previously viewed, disagreement-reviewed same-session holdout; not blind/independent'
    models = {'baseline':args.baseline}
    if args.candidate:
        models['candidate'] = args.candidate
    for name,path in models.items():
        report['models'][name] = {'path':str(path.resolve()),'sha256':sha256(path)}
    rows, gallery = [], []
    for domain,records in domains.items():
        evaluations = {}
        report['domains'][domain] = {'count':len(records),'label_counts':dict(Counter(r['label'] for r in records))}
        for name,path in models.items():
            metrics,outputs = predict(path,records)
            report['domains'][domain][name] = metrics
            evaluations[name] = outputs
            print(domain,name,json.dumps(metrics),flush=True)
        for index,record in enumerate(records):
            predictions = {name:values[index] for name,values in evaluations.items()}
            row = dict(record,predictions=predictions)
            rows.append(row)
            if any(p['label']!=record['label'] for p in predictions.values()):
                folder = args.output/'errors'/domain
                folder.mkdir(parents=True,exist_ok=True)
                target = folder/(record['id']+'.png')
                image = cv2.imread(record['path'])
                if not cv2.imwrite(str(target),image):
                    raise OSError(str(target))
                label_text = ' / '.join(f'{name}: {p["label"]} ({p["confidence"]:.6f})' for name,p in predictions.items())
                relative = target.relative_to(args.output).as_posix()
                gallery.append(f'<article><a href="{html.escape(relative)}"><img loading="lazy" src="{html.escape(relative)}"></a>'
                               f'<p>{html.escape(domain)} · GT: {record["label"]}<br>{html.escape(label_text)}</p>'
                               f'<small>{html.escape(record["id"])}</small></article>')
    if args.candidate and args.candidate_pt:
        report['export_parity'] = torch_onnx_parity(args.candidate_pt,args.candidate,domains['ha_val'])
    (args.output/'per_image.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fields = ['domain','id','ground_truth']+[f'{n}_{k}' for n in models for k in ('label','confidence')]
    with (args.output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        domain_by_id={r['id']:d for d,records in domains.items() for r in records}
        for r in rows:
            result={'domain':domain_by_id[r['id']],'id':r['id'],'ground_truth':r['label']}
            for name,p in r['predictions'].items():
                result.update({f'{name}_label':p['label'],f'{name}_confidence':p['confidence']})
            writer.writerow(result)
    heading='<h1>Classifier comparison — errors</h1><p>GT = human-confirmed HA or legacy dataset label. '
    heading+='Confidence belongs to each model prediction. Native crop previews; no labels changed.</p>'
    heading+='<p>HA test is same-session chronological holdout, not independent factory qualification.</p><a href="report.json">Metrics</a>'
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'
       'body{background:#181c23;color:#eee;font:15px Arial;margin:24px}a{color:#9ed4ff}'
       '.grid{display:flex;flex-wrap:wrap;gap:12px}article{width:245px;padding:12px;background:#252c37}'
       'img{max-width:100%;height:190px;object-fit:contain}small{overflow-wrap:anywhere}</style>'
       +heading+'<div class="grid">'+''.join(gallery)+'</div>',encoding='utf-8')
    print('Saved',args.output.resolve(),flush=True)


if __name__=='__main__':
    main()
