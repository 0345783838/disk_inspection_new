"""Two matched nano fits with agent-drafted groove/entry supervision, no deployment."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0, str(PROJECT/'SERVICE'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(PROJECT/'SERVICE/.venv/Lib/site-packages'))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import classifier_dataset as canonical
# Frozen copies live below .work/run/code_snapshot, not SERVICE/training.
canonical.PROJECT = PROJECT
canonical.REGISTRY = PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json'
from classifier_dataset import CanonicalDataset, active_dataset, read, atomic_json, atomic_text
from audit_classifier_dataset import audit
from adopt_classifier_dataset import decoded_fingerprint
from classifier_metrics import classification_metrics
import train_classifier_canonical_holdout as base
from classifier_groove_ownership import (LABELS, HEATMAP_NAMES, VISIBILITY_NAMES, OWNERSHIP_NAMES,
    FORMAT, from_parent, save_model, load_model, export_model, tensor, state_sha, gaussian_heatmaps)

FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
FIT_NAMES = ['control', 'auxiliary']
USER_STATEMENT = 'xong mày thử tự label theo ý của mày rồi train thử và đánh giá giải pháp mới xem thế nào nhé, mai tao kiểm tra, có gì ko ổn tao sẽ sửa sau'
DEFAULT_ROOT = PROJECT/'.work/classifier_groove_ownership_20261104'
DEFAULT_AUX = PROJECT/'.work/classifier_accuracy_research_20261103/groove_ownership_preparation/auto_annotations_merged.json'
PARENT_PT = PROJECT/'.work/classifier_same_slot_ai_20261029/artifacts/best.pt'
PARENT_SHA = '8dd29530326ac68295c5acf8ac7aee45441cac39513f2f4eb5d9e1a590603492'
SERVICE_SHA = '89a7ef199e575cdb77a7bcc8edb0d1250affa41bb9f804cadbdc24b1fbccc434'
RETAINED_SHA = '775872fc29f8e00ab9934747cfaa543ad966e2e905d5e53740717d4114b8363c'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.writing')
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_model(path,model,**metadata):
    path=Path(path);temporary=path.with_name(path.name+'.model-writing')
    save_model(temporary,model,**metadata)
    os.replace(temporary,path)


def checkpoint_selection_guard(path,epoch,rank):
    """Reject a best file advanced beyond an interrupted last-checkpoint commit."""
    value=torch.load(path,map_location='cpu',weights_only=False)
    saved=value.get('metadata',{})
    if saved.get('epoch')!=epoch or tuple(saved.get('rank',[]))!=tuple(rank):
        raise ValueError('Checkpoint selection metadata differs from committed state: '+str(path)+
                         '; inspect interrupted checkpoint transaction before recovery')


def metadata_records(manifest):
    return [(r['id'], r['label'], r['sha256'], r['split'], r['image'], r.get('split_group')) for r in manifest['records']]


def same_label_authority(a,b):
    return all(a.get(k,[])==b.get(k,[]) for k in ['records','excluded_records','excluded'])


def metrics(rows, predictions):
    names = ['all'] + sorted({r['origin'] for r in rows})
    names += [f'{o}_Cam{c}' for o in ['old_ha', 'new_factory'] for c in [1, 2]
              if any(r['origin'] == o and r.get('camera') == c for r in rows)]
    result = {}
    for name in names:
        ix = [i for i,r in enumerate(rows) if name == 'all' or r['origin'] == name
              or name == f"{r['origin']}_Cam{r.get('camera')}"]
        if not ix:
            continue
        m = classification_metrics([LABELS.index(rows[i]['label']) for i in ix],
                                   [LABELS.index(predictions[i]['label']) for i in ix])
        cm = m['confusion_matrix_true_rows']
        m.update(errors=sum(cm[i][j] for i in range(3) for j in range(3) if i != j),
                 false_ng=cm[1][0]+cm[2][0], ok_as_empty=cm[2][1])
        result[name] = m
    return result


def safety_rank(current, references):
    """Strict tuple: source/camera NG regression, all NG, falseNG, emptyOK, errors."""
    excess = sum(max(0, value['ng_not_detected']-min(ref[name]['ng_not_detected'] for ref in references))
                 for name,value in current.items() if name != 'all')
    all_m = current['all']
    return tuple(-int(v) for v in [excess, all_m['ng_not_detected'], all_m['false_ng'],
                                  all_m['no_disk_as_ok'], all_m['errors']])


def _point(xy, width, height):
    if xy is None:
        return None
    values = np.asarray(xy, dtype=np.float32)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('Point must be a finite native-pixel xy pair')
    if not (0 <= values[0] < width and 0 <= values[1] < height):
        raise ValueError(f'Point outside native crop: {xy} / {width}x{height}')
    # cv2.resize uses pixel-center coordinates. Then map to model's [0,1] grid.
    return np.clip(((values+.5)*224/np.array([width,height])-.5)/223, 0, 1).astype(np.float32)


def blank_targets():
    return dict(xy=np.zeros((6,2), np.float32), point_weight=np.zeros(6,np.float32),
                visibility=np.full(2,3,np.int64), visibility_weight=np.zeros(2,np.float32),
                count=np.int64(0), count_weight=np.float32(0),
                ownership=np.zeros(2,np.int64), ownership_weight=np.zeros(2,np.float32))


def annotation_targets(record, width, height):
    """Agent drafts stay provisional; estimates/unknown points have zero loss."""
    out = blank_targets()
    if record is None:
        return out
    size = record.get('native_size')
    if size is not None and list(size) != [width,height]:
        raise ValueError('Auxiliary native_size differs from crop')
    vis_alias = {'visible':0, 'partially_visible':1, 'partial':1, 'occluded':2,
                 'not_visible':2, 'unobservable':2, 'unknown':3}
    for index,key in enumerate(['left_peak','right_peak']):
        peak = record.get('groove',{}).get(key,{})
        visibility = peak.get('visibility','unknown')
        if visibility not in vis_alias:
            raise ValueError('Unknown peak visibility: '+str(visibility))
        out['visibility'][index] = vis_alias[visibility]
        if 'visibility' in peak:
            out['visibility_weight'][index] = .5
        coordinate = _point(peak.get('xy'),width,height)
        source = str(peak.get('coordinate_source','')).lower()
        observed = ('observed' in source or source == 'agent_visual') and not any(s in source for s in ['estimated','inferred','unknown'])
        if coordinate is not None:
            out['xy'][index] = coordinate
        if coordinate is not None and observed and vis_alias[visibility] in [0,1]:
            tentative = peak.get('certainty') in ['tentative','uncertain'] or float(peak.get('uncertainty_px',0)) > 4
            out['point_weight'][index] = .25 if tentative else .5
    instances = record.get('instances',[])
    def sort_key(instance):
        point = instance.get('insertion_end_xy') or instance.get('trace_point_xy')
        return (float(point[0]) if point is not None else float('inf'), str(instance.get('id','')))
    instances = sorted(instances,key=sort_key)[:2]
    for index,instance in enumerate(instances):
        certainty = instance.get('certainty','unknown')
        weight = .5 if certainty in ['clear','confirmed','certain'] else .25 if certainty in ['tentative','partial'] else 0.
        source = str(instance.get('coordinate_source','agent_visual_observed')).lower()
        if any(s in source for s in ['estimated','inferred','unknown']):
            weight = 0.
        for field,channel in [('insertion_end_xy',2+index),('trace_point_xy',4+index)]:
            coordinate = _point(instance.get(field),width,height)
            if coordinate is not None:
                out['xy'][channel] = coordinate
                out['point_weight'][channel] = weight
        owner = instance.get('groove_assignment','unknown')
        if owner in OWNERSHIP_NAMES and out['point_weight'][2+index] > 0:
            out['ownership'][index] = OWNERSHIP_NAMES.index(owner)
            out['ownership_weight'][index] = weight
    count = record.get('target_count',{})
    value = count.get('value')
    if count.get('complete_review') is True and value is not None:
        value = 2 if str(value) == '2+' else int(value)
        if value < 0:
            raise ValueError('Negative target occupancy')
        out['count'] = np.int64(min(value,2))
        out['count_weight'] = np.float32(.5)
    return out


def flip_targets(target):
    out = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in target.items()}
    out['xy'][:,0] = 1-out['xy'][:,0]
    out['xy'][[0,1]] = out['xy'][[1,0]]
    out['point_weight'][[0,1]] = out['point_weight'][[1,0]]
    out['visibility'] = out['visibility'][[1,0]]
    out['visibility_weight'] = out['visibility_weight'][[1,0]]
    owners = out['ownership'].copy()
    out['ownership'] = np.where(owners == 1,2,np.where(owners == 2,1,owners))
    def key(i):
        channel = 2+i if out['point_weight'][2+i]>0 else 4+i
        return float(out['xy'][channel,0]) if out['point_weight'][channel]>0 else 2.+i
    order = sorted(range(2),key=key)
    permutation = [0,1]+[2+i for i in order]+[4+i for i in order]
    out['xy'] = out['xy'][permutation]
    out['point_weight'] = out['point_weight'][permutation]
    out['ownership'] = out['ownership'][order]
    out['ownership_weight'] = out['ownership_weight'][order]
    return out


def augment(image, target, rng):
    target = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in target.items()}
    if rng.random() < .5:
        image = cv2.flip(image,1)
        target = flip_targets(target)
    x = image.astype(np.float32)
    mean = x.mean((0,1),keepdims=True)
    x = np.clip(((x-mean)*rng.uniform(.92,1.08)+mean)*rng.uniform(.9,1.1)/255.,0,1)
    x = np.clip(np.power(x,rng.uniform(.94,1.06))*255,0,255).astype(np.uint8)
    return tensor(x),target


def auxiliary_losses(aux, target):
    weights = target['point_weight']
    heatmap = gaussian_heatmaps(target['xy'],size=28,sigma=1.25).flatten(2)
    heatmap = heatmap/heatmap.sum(-1,keepdim=True).clamp_min(1e-12)
    point_ce = -(heatmap*F.log_softmax(aux['heatmap_logits'].flatten(2),-1)).sum(-1)
    def weighted(values, weight):
        return (values*weight).sum()/weight.sum().clamp_min(1.)
    point = weighted(point_ce,weights)
    visibility = weighted(F.cross_entropy(aux['visibility_logits'].flatten(0,1),target['visibility'].flatten(),
                            reduction='none').reshape(-1,2),target['visibility_weight'])
    count = weighted(F.cross_entropy(aux['count_logits'],target['count'],reduction='none'),target['count_weight'])
    ownership = weighted(F.cross_entropy(aux['ownership_logits'].flatten(0,1),target['ownership'].flatten(),
                            reduction='none').reshape(-1,2),target['ownership_weight'])
    return dict(geometry=point,visibility=visibility,count=count,ownership=ownership)


def epoch_indices(count, weights, auxiliary_indices, rng):
    auxiliary_indices = np.asarray(auxiliary_indices,dtype=np.int64)
    if count != 3778 or len(auxiliary_indices) != 48:
        raise ValueError('Expected3778train and48aux references')
    indices = np.r_[np.arange(count),rng.choice(count,count,replace=True,p=weights),np.tile(auxiliary_indices,4)]
    rng.shuffle(indices)
    assert len(indices)==7748 and len(set(indices.tolist()))==count
    return indices


def process_inventory():
    if os.name != 'nt':
        import psutil
        return [dict(pid=p.pid,ppid=p.ppid(),command=' '.join(p.cmdline())) for p in psutil.process_iter() if 'python' in p.name().lower()]
    command = "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    value = subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,
                           creationflags=FLAGS,check=True).stdout.strip()
    data = json.loads(value) if value else []
    data = data if isinstance(data,list) else [data]
    return [dict(pid=int(r['ProcessId']),ppid=int(r['ParentProcessId']),command=r.get('CommandLine') or '') for r in data]


def guard_processes(root, allow_parent=False):
    inventory = process_inventory()
    allowed = {os.getpid()}
    claimed_parent = int(os.environ.get('CLASSIFIER_CAMPAIGN_PID','0'))
    if allow_parent and claimed_parent:
        actual = next((r for r in inventory if r['pid']==claimed_parent),None)
        if actual and root.name in actual['command'] and '--phase campaign' in actual['command']:
            allowed.add(claimed_parent)
    duplicates = [r for r in inventory if r['pid'] not in allowed and 'train_classifier_' in r['command'] and '--phase' in r['command']]
    if duplicates:
        raise RuntimeError('Another classifier parent/child is alive: '+json.dumps(duplicates))
    if (root/'STOP_REQUESTED').exists():
        raise InterruptedError('STOP_REQUESTED exists; never automatically resume a stopped run')
    if (root/'status.json').exists() and read(root/'status.json').get('status') in ['complete','stopped']:
        raise ValueError('Do not restart a completed or user-stopped run')
    return inventory


def publish(root,status,phase,child=None,error=None,**extra):
    previous = read(root/'status.json') if (root/'status.json').exists() else {}
    state = dict(previous,status=status,phase=phase,updated_at=base.now(),error=error,**extra)
    parent = int(os.environ.get('CLASSIFIER_CAMPAIGN_PID','0')) or os.getpid()
    state.update(campaign_pid=parent if status not in ['complete','failed','stopped'] else None,active_pid=child)
    atomic_json(root/'status.json',state)
    pointer = PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'
    if pointer.exists() and Path(read(pointer)['path']).resolve()==root.resolve():
        record=read(pointer)
        record.update({k:state[k] for k in ['status','phase','updated_at','error','campaign_pid','active_pid']})
        if status=='complete':
            record.update(completed_at=base.now(),recommended_for_review=read(root/'release_decision.json')['recommended_for_review'])
        atomic_json(pointer,record)


def check(root, verify_images=False):
    if (root/'STOP_REQUESTED').exists():
        raise InterruptedError('STOP_REQUESTED')
    p=read(root/'protocol.json')
    if digest(root/'protocol.json')!=(root/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol changed')
    for name,expected in p['code_hashes'].items():
        if digest(root/'code_snapshot'/name)!=expected:
            raise ValueError('Frozen code changed: '+name)
    for name,expected in p['reference_hashes'].items():
        if digest(root/'reference'/name)!=expected:
            raise ValueError('Reference changed: '+name)
    for name,expected in p.get('reference_validation_hashes',{}).items():
        if digest(root/'reference'/name)!=expected:
            raise ValueError('Frozen baseline validation cache changed: '+name)
    if (root/'prepare_complete.json').exists() and not p.get('reference_validation_hashes'):
        raise ValueError('Prepared run lacks baseline validation hash bindings')
    if digest(root/'source_manifest_at_training.json')!=p['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    if digest(Path(p['pool'])/'manifest.json')!=p['manifest_sha256']:
        raise ValueError('Current canonical manifest changed; inspect before continuing')
    if digest(root/'auxiliary_annotations.json')!=p['auxiliary_sha256']:
        raise ValueError('Frozen auxiliary annotation changed')
    for name in ['protected_runtime.json','protected_inputs.json']:
        for path,expected in read(root/name).items():
            if digest(path)!=expected:
                raise ValueError('Protected runtime/input changed: '+path)
    if verify_images:
        for r in read(root/'source_manifest_at_training.json')['records']:
            if digest(root/'input_snapshot'/r['image'])!=r['sha256']:
                raise ValueError('Frozen crop changed: '+r['id'])
    return p


def rows(root):
    return [dict(r,path=str(root/'input_snapshot'/r['image'])) for r in read(root/'source_manifest_at_training.json')['records']]


def result_probabilities(values):
    return [dict(label=LABELS[int(p.argmax())],confidence=float(p.max()),probabilities={c:float(v) for c,v in zip(LABELS,p)}) for p in values]


def predict_torch(model, records, images=None, batch=64):
    model.eval()
    device=next(model.parameters()).device
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(records),batch):
            im = images[start:start+batch] if images is not None else [cv2.imread(r['path']) for r in records[start:start+batch]]
            if any(v is None for v in im):
                raise ValueError('Unreadable frozen crop')
            value=torch.from_numpy(np.array([tensor(v) for v in im])).to(device)
            predictions.extend(result_probabilities(model(value).cpu().numpy()))
    return predictions


def predict_onnx(path, records, root=None):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    predictions=[]
    for start in range(0,len(records),32):
        if root is not None and (root/'STOP_REQUESTED').exists():
            raise InterruptedError('STOP_REQUESTED')
        im=[cv2.imread(r['path']) for r in records[start:start+32]]
        if any(v is None for v in im):
            raise ValueError('Unreadable frozen crop')
        value=np.array([tensor(v) for v in im])
        predictions.extend(result_probabilities(session.run(None,{session.get_inputs()[0].name:value})[0]))
    return predictions


def _base_page():
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Khe đĩa · nano</title><style>body{background:#14202e;color:#e6eef6;font:16px system-ui;max-width:1450px;margin:28px}a{color:#9bd4ff}p{line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #63778d;padding:9px}.grid{display:flex;flex-wrap:wrap;gap:15px}article{width:300px;background:#26394b;padding:14px;overflow-wrap:anywhere}img{width:270px;height:250px;object-fit:contain}</style><h1>Hai lượt nano: đối chứng và giám sát đầu đĩa–khe</h1><p>Đĩa thuộc khoảng lõm giữa hai đỉnh lồi. Điểm bị che/phản xạ được ghi độ nhìn thấy; vị trí ước lượng không thành tọa độ quan sát thật. 48 nhãn phụ do agent tự gán theo yêu cầu, chưa được người dùng kiểm tra. Canonical revision12 giữ nguyên nhãn/split.</p><p>Cùng current +AI parent, kiến trúc, seed, inputRGB/stretch224, sampler và lịch học; chỉ auxiliary loss khác nhau. Val đã dùng phát triển/có prior exposure, không phải kiểm thử mù. Không triển khai.</p><p id="status">Đang tải trạng thái…</p><p><a href="auxiliary_review.html">48 nhãn phụ cần xem</a> · <a href="protocol.json">Protocol</a> · <a href="comparison.html">So sánh</a> · <a href="error_review.html">Ca sai</a> · <a href="RESULTS.md">Kết quả</a> · <a href="logs/control.log">Log đối chứng</a> · <a href="logs/auxiliary.log">Log hình học</a></p><script>async function poll(){try{let s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · epoch '+(s.epoch??'')+' · best '+(s.best_epoch??'')+' · '+(s.error??'')}catch(e){}}poll();setInterval(poll,10000)</script></html>'''


def page():
    return _base_page().replace('<a href="auxiliary_review.html">48 nhãn phụ cần xem</a>',
        '<a href="aux_review/index.html">Bộ nhãn phụ đầy đủ</a> · <a href="auxiliary_review.html">Điểm dùng loss</a>')


def auxiliary_review(root, records, annotations):
    output=root/'auxiliary_preview';output.mkdir(exist_ok=True)
    by_id={r['id']:r for r in records}
    cards=[]
    for record in annotations:
        r=by_id[record['canonical_id']]
        image=cv2.imread(r['path']);h,w=image.shape[:2]
        target=annotation_targets(record,w,h)
        preview=cv2.resize(image,(w*3,h*3),interpolation=cv2.INTER_NEAREST)
        for index,point in enumerate(target['xy']):
            if target['point_weight'][index]<=0:
                continue
            x=int(round(float(point[0])*(w*3-1)));y=int(round(float(point[1])*(h*3-1)))
            color=(100,230,120) if target['point_weight'][index]>=.5 else (0,200,255)
            cv2.circle(preview,(x,y),5,color,2)
            cv2.putText(preview,HEATMAP_NAMES[index],(max(0,x-20),max(15,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        filename=r['id']+'.png';assert cv2.imwrite(str(output/filename),preview)
        cards.append(f'<article><a href="auxiliary_preview/{filename}"><img src="auxiliary_preview/{filename}"></a><p>{html.escape(r["id"])}</p><p>Nhãn canonical: {r["label"]}; train</p><p>{html.escape(record.get("notes", ""))}</p><pre>{html.escape(json.dumps(record.get("target_count",{}),ensure_ascii=False))}</pre></article>')
    atomic_text(root/'auxiliary_review.html',page().split('<h1>')[0]+'<h1>48 nhãn phụ do agent tự gán</h1><p>Chưa được người dùng kiểm tra. Xanh: agent cho rõ, trọng số0.5; vàng: agent chưa chắc,0.25. Điểm ước lượng/không thấy không được dùng loss tọa độ. Đây không phải editor nhãn classifier.</p><div class="grid">'+''.join(cards)+'</div></html>')


def prepare(root, auxiliary_source=DEFAULT_AUX):
    if root.exists():
        raise FileExistsError('Run exists; inspect and use safe --resume only for unfinished work')
    inventory=guard_processes(root)
    pool=Path(read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json')['path'])
    before=read(pool/'manifest.json');before_hash=digest(pool/'manifest.json')
    if before['canonical_revision']!=12 or len(before['records'])!=5393:
        raise ValueError('Expected exact canonical revision12')
    base.validate_rows(before['records'],base.BALANCED_POLICY)
    aux=read(auxiliary_source)
    source_manifest_hash=aux.get('canonical_manifest_sha256',aux.get('source_canonical_manifest_sha256'))
    if aux.get('canonical_revision')!=12:
        raise ValueError('Auxiliary annotation source bound to another canonical revision')
    if source_manifest_hash!=before_hash:
        # Authorization changes headers, never records. Compare historical
        # bound records to CURRENT authority; do not import historical labels.
        snapshot=PROJECT/'.work/classifier_context_20261030/source_manifest_at_training.json'
        if not snapshot.exists() or digest(snapshot)!=source_manifest_hash or not same_label_authority(before,read(snapshot)):
            raise ValueError('Auxiliary source manifest differs beyond authorization headers')
    annotations=aux['records'];meta={r['id']:r for r in before['records']}
    if len(annotations)!=48 or len({r['canonical_id'] for r in annotations})!=48:
        raise ValueError('Require exactly48 unique auxiliary references')
    for r in annotations:
        canonical_row=meta[r['canonical_id']]
        if canonical_row['split']!='train' or canonical_row.get('is_synthetic') or r['crop_sha256']!=canonical_row['sha256']:
            raise ValueError('Auxiliary reference must match current real train crop')
        image=cv2.imread(str(pool/canonical_row['image']))
        if image is None:
            raise ValueError('Missing auxiliary crop')
        annotation_targets(r,image.shape[1],image.shape[0])
    if digest(PARENT_PT)!=PARENT_SHA:
        raise ValueError('Current +AI parent PT changed')
    service=PROJECT/'SERVICE/config/models/c_p.onnx'
    retained=PROJECT/'SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.onnx'
    if digest(service)!=SERVICE_SHA or digest(retained)!=RETAINED_SHA:
        raise ValueError('SERVICE or retained baseline differs from authorized trial')
    receipt=dict(authorized_at=base.now(),canonical_revision=12,incomplete_review_acknowledged=True,
        user_statement=USER_STATEMENT,auxiliary_review='agent_drafts_pending_user_review',
        scope='One new bounded matched pilot, control and auxiliary fits only; user-authorized agent geometry drafts on48 canonical-train rows, current +AI nano parent, last2backbone blocks/head, frozenBN statistics, max300/patience100, no classifier label/split edits, no deployment; full-val export parity and28hardware-off White<=2x required.')
    CanonicalDataset(pool).authorize_training(receipt,12)
    pool=active_dataset(required_split_policy=base.BALANCED_POLICY)
    raw=(pool/'manifest.json').read_bytes();manifest=read(pool/'manifest.json')
    if not same_label_authority(manifest,before):
        raise ValueError('Canonical records changed during authorization')
    root.mkdir(parents=True)
    for folder in ['logs','reference','input_snapshot','code_snapshot','jobs','evaluations']:
        (root/folder).mkdir()
    (root/'source_manifest_at_training.json').write_bytes(raw)
    shutil.copy2(auxiliary_source,root/'auxiliary_annotations.json')
    for r in manifest['records']:
        source=pool/r['image']
        if digest(source)!=r['sha256'] or decoded_fingerprint(source)!=r['decoded_sha256']:
            raise ValueError('Canonical crop integrity mismatch: '+r['id'])
        destination=root/'input_snapshot'/r['image'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    atomic_json(root/'input_snapshot/manifest.json',dict(manifest,is_canonical=False,immutable_run_snapshot=True))
    integrity=audit(pool);atomic_json(root/'integrity_before.json',integrity)
    if not integrity['passed']:
        raise ValueError('Canonical audit failed')
    for source,name in [(PARENT_PT,'parent.pt'),(service,'service.onnx'),(retained,'retained.onnx')]:
        shutil.copy2(source,root/'reference'/name)
    protected={}
    for directory in [PROJECT/'SERVICE/src',PROJECT/'SERVICE/config',PROJECT/'APP/DiskInspection/bin/BroSight_Disinspector/plugin/config']:
        for f in directory.rglob('*'):
            if f.is_file() and f.suffix.lower() in ['.py','.env','.onnx','.pt','.json']:
                protected[str(f.resolve())]=digest(f)
    app=PROJECT/'APP/DiskInspection/bin/BroSight_Disinspector/DiskInspection.exe.config'
    if app.exists():protected[str(app)]=digest(app)
    protected[str(PARENT_PT)]=digest(PARENT_PT)
    atomic_json(root/'protected_runtime.json',protected)
    white=read(PROJECT/'.work/factory_audit_20261020_run/report.json')
    inputs={str(Path(r['source']).resolve()):digest(r['source']) for r in white['white']}
    inputs[str(Path(white['app_config']).resolve())]=digest(white['app_config'])
    inputs[str(PROJECT/'.work/factory_audit_20261020_run/report.json')]=digest(PROJECT/'.work/factory_audit_20261020_run/report.json')
    atomic_json(root/'protected_inputs.json',inputs)
    code_hashes={}
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source,root/'code_snapshot'/source.name);code_hashes[source.name]=digest(source)
    train_count=sum(r['split']=='train' for r in manifest['records']);val_count=sum(r['split']=='val' for r in manifest['records'])
    if (train_count,val_count)!=(3778,1615):
        raise ValueError('Unexpected authoritative split')
    p=dict(created_at=base.now(),campaign='groove_ownership_20261104',pool=str(pool),canonical_revision=12,
        manifest_sha256=digest(root/'source_manifest_at_training.json'),annotation_manifest_sha256_before_authorization=before_hash,
        original_auxiliary_source_manifest_sha256=source_manifest_hash,
        auxiliary_source=str(Path(auxiliary_source).resolve()),auxiliary_sha256=digest(root/'auxiliary_annotations.json'),
        auxiliary_count=48,auxiliary_agent_drafts=True,auxiliary_user_review_complete=False,
        authorization=receipt,review_incomplete=True,epochs=300,patience=100,seed=1104,batch=64,
        parent_lr=3e-6,geometry_lr=3e-4,weight_decay=1e-4,train_last_blocks=2,freeze_bn_running_stats=True,
        fit_names=FIT_NAMES,full_fits_max=2,train_count=train_count,val_count=val_count,
        epoch_draws=7748,every_train_once=3778,balanced_extras=3778,auxiliary_extra_draws=192,
        auxiliary_extra_policy='Four complete repetitions of48train references, same in both fits',
        balanced_extras_policy='base.sampling_weights camera_target_slot_legacy_ng_v1: source40/40/20, equal HA/new cameras; classes30/30/40,legacy70/15/15',
        input_transform='BGR->RGB cv2.INTER_LINEAR stretch224 /255 float32; mild photometric and hflip only',
        coordinate_transform='native pixel centers mapped with cv2.resize center formula; exact hflip peak/instance/neighbor remapping',
        loss=dict(class_weights=[2,1,1],teacher_kd=.5,teacher_temperature=2,teacher_min_confidence=.8,
                  auxiliary_weights=dict(geometry=.3,visibility=.1,count=.2,ownership=.1),
                  draft_clear_weight=.5,draft_tentative_weight=.25,estimated_unknown_coordinate_weight=0,
                  peak_uncertainty_clear_max_px=4,heatmap='normalized Gaussian spatial CE,sigma1.25gridpx'),
        selection='Strict NG source/camera regression tuple, all NG, falseNG, empty-asOK, errors; references currentSERVICE and retained best111; epoch0 eligible; strict improvement only',
        gates='zeroNG, no per-source/camera regression in NG/falseNG/empty-asOK/errors vs both baselines; auxiliary must strictly beat matched control; wholeWhite<=2x',
        prior_parent_exposure=True,validation_scope='Repeatedly used same-session spatial development, shared frames; not independent testing',
        reference_hashes={f.name:digest(f) for f in (root/'reference').iterdir()},code_hashes=code_hashes,
        production_deployed=False,report_url='http://127.0.0.1:8808/')
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'))
    atomic_json(root/'authorization_receipt.json',receipt);atomic_json(root/'processes_before_prepare.json',inventory)
    atomic_json(root/'previous_active_training.json',read(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'))
    atomic_json(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json',dict(path=str(root),entrypoint='SERVICE/training/train_classifier_groove_ownership.py',
        report_url=p['report_url'],canonical_revision=12,status='prepared',phase='immutable inputs and draftaux frozen',
        campaign_pid=os.getpid(),active_pid=None,authorization=receipt,production_deployed=False))
    atomic_text(root/'index.html',page())
    data=rows(root);auxiliary_review(root,data,annotations)
    source_gallery=Path(auxiliary_source).parent/'auto_review'
    if source_gallery.is_dir():shutil.copytree(source_gallery,root/'aux_review',dirs_exist_ok=True)
    validation=[r for r in data if r['split']=='val']
    for name in ['service','retained']:
        predictions=predict_onnx(root/'reference'/f'{name}.onnx',validation,root)
        atomic_json(root/'reference'/f'{name}_validation.json',dict(model_sha256=digest(root/'reference'/f'{name}.onnx'),
                    manifest_sha256=p['manifest_sha256'],metrics=metrics(validation,predictions),predictions=predictions))
    p['reference_validation_hashes']={f'{name}_validation.json':digest(root/'reference'/f'{name}_validation.json') for name in ['service','retained']}
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'))
    atomic_json(root/'prepare_complete.json',dict(passed=True,prepared_at=base.now(),manifest_sha256=p['manifest_sha256'],
                auxiliary_sha256=p['auxiliary_sha256'],snapshot_count=5393,aux_train_only=True))
    publish(root,'prepared','canonical and48agent drafts frozen; both baselines evaluated')
    check(root,verify_images=True)
    return p


def finish_prepare(root):
    """Finish only immutable setup artifacts after an interrupted preparation."""
    p=check(root,verify_images=True)
    data=rows(root);annotations=read(root/'auxiliary_annotations.json')['records']
    auxiliary_review(root,data,annotations)
    source_gallery=Path(p['auxiliary_source']).parent/'auto_review'
    if source_gallery.is_dir():
        shutil.copytree(source_gallery,root/'aux_review',dirs_exist_ok=True)
    validation=[r for r in data if r['split']=='val']
    for name in ['service','retained']:
        destination=root/'reference'/f'{name}_validation.json'
        if destination.exists():
            existing=read(destination)
            if existing['manifest_sha256']!=p['manifest_sha256'] or existing['model_sha256']!=digest(root/'reference'/f'{name}.onnx'):
                raise ValueError('Baseline cache binding mismatch')
            continue
        predictions=predict_onnx(root/'reference'/f'{name}.onnx',validation,root)
        atomic_json(destination,dict(model_sha256=digest(root/'reference'/f'{name}.onnx'),
                    manifest_sha256=p['manifest_sha256'],metrics=metrics(validation,predictions),predictions=predictions))
    p['reference_validation_hashes']={f'{name}_validation.json':digest(root/'reference'/f'{name}_validation.json') for name in ['service','retained']}
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'))
    atomic_json(root/'prepare_complete.json',dict(passed=True,prepared_at=base.now(),manifest_sha256=p['manifest_sha256'],
                auxiliary_sha256=p['auxiliary_sha256'],snapshot_count=5393,aux_train_only=True))
    return p


def fit(root,name,resume=False):
    p=check(root);folder=root/'jobs'/name
    if name not in FIT_NAMES:
        raise ValueError('Unknown fit')
    if (folder/'completion.json').exists():
        raise ValueError('Do not repeat completed fit '+name)
    folder.mkdir(exist_ok=True)
    if (folder/'last.pt').exists() and not resume:
        raise ValueError('Unfinished checkpoint exists; explicit safe --resume required')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this bounded campaign')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(p['seed']);np.random.seed(p['seed']);rng=np.random.default_rng(p['seed'])
    data=rows(root);training=[r for r in data if r['split']=='train'];validation=[r for r in data if r['split']=='val']
    aux_by_id={r['canonical_id']:r for r in read(root/'auxiliary_annotations.json')['records']}
    if set(aux_by_id)-{r['id'] for r in training}:
        raise ValueError('Validation auxiliary annotation entered training')
    images=[];targets=[]
    for r in training:
        image=cv2.imread(r['path'])
        if image is None:
            raise ValueError('Missing frozen train crop')
        images.append(image);targets.append(annotation_targets(aux_by_id.get(r['id']),image.shape[1],image.shape[0]))
    val_images=[cv2.imread(r['path']) for r in validation]
    if any(v is None for v in val_images):
        raise ValueError('Missing frozen val crop')
    weights=np.asarray(base.sampling_weights(training,base.BALANCED_POLICY),dtype=np.float64);weights/=weights.sum()
    auxiliary_indices=[i for i,r in enumerate(training) if r['id'] in aux_by_id]
    references=[read(root/'reference'/f'{n}_validation.json')['metrics'] for n in ['service','retained']]
    model=from_parent(root/'reference/parent.pt',train_last_blocks=p['train_last_blocks']).cuda()
    initialization=dict(model_state_sha256=state_sha(model.state_dict()),frozen_state_sha256=model.frozen_state_hash(),
                        parent_sha256=digest(root/'reference/parent.pt'),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad))
    if (folder/'initialization.json').exists() and read(folder/'initialization.json')!=initialization:
        raise ValueError('Initialization no longer matches frozen fit')
    other=root/'jobs'/('auxiliary' if name=='control' else 'control')/'initialization.json'
    if other.exists() and read(other)!=initialization:
        raise ValueError('Control/auxiliary initialization differs')
    atomic_json(folder/'initialization.json',initialization)
    frozen=model.frozen_state()
    parent_predictions=predict_torch(model,validation,val_images)
    cached=read(root/'reference/service_validation.json')['predictions']
    mismatches=[r['id'] for r,a,b in zip(validation,parent_predictions,cached) if a['label']!=b['label']]
    if mismatches:
        raise ValueError('Epoch0 parent/installedONNX argmax mismatch: '+str(mismatches))
    parent_metrics=metrics(validation,parent_predictions);best_rank=safety_rank(parent_metrics,references)
    best_epoch=0;learned_rank=None;learned_epoch=None;start_epoch=1;cumulative=Counter()
    optimizer=torch.optim.AdamW(model.parameter_groups(lr=p['parent_lr'],aux_lr=p['geometry_lr']),weight_decay=p['weight_decay'])
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if last['manifest_sha256']!=p['manifest_sha256'] or last['auxiliary_sha256']!=p['auxiliary_sha256']:
            raise ValueError('Resume checkpoint belongs to another input')
        model.load_state_dict(last['model'],strict=True);optimizer.load_state_dict(last['optimizer'])
        start_epoch=last['epoch']+1;best_epoch=last['best_epoch'];best_rank=tuple(last['best_rank'])
        learned_rank=tuple(last['learned_rank']) if last['learned_rank'] is not None else None
        learned_epoch=last['learned_epoch'];rng.bit_generator.state=last['numpy_rng']
        torch.set_rng_state(last['torch_rng']);torch.cuda.set_rng_state(last['cuda_rng'])
        cumulative.update(last.get('cumulative_sampling',{}));model.assert_frozen(frozen)
        checkpoint_selection_guard(folder/'best.pt',best_epoch,best_rank)
        if learned_epoch is not None:
            checkpoint_selection_guard(folder/'learned_best.pt',learned_epoch,learned_rank)
        with (folder/'recoveries.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(dict(at=base.now(),from_epoch=last['epoch'],checkpoint_sha256=digest(folder/'last.pt'),partial_uncheckpointed_epoch_may_be_replayed=True))+'\n')
    else:
        atomic_model(folder/'best.pt',model,epoch=0,rank=list(best_rank),epoch0_parent=True)
        atomic_json(folder/'best_metrics.json',parent_metrics)
        atomic_json(folder/'epoch0_parity.json',dict(count=1615,argmax_mismatches=0,exact_parent_architecture=True))
    atomic_json(folder/'sampling.json',dict(train_count=3778,epoch_draws=7748,every_train_once=3778,balanced_extras=3778,
        auxiliary_extras=192,auxiliary_references=48,auxiliary_extra_complete_repeats=4,validation_in_sampler=False,
        full_epoch_source_budget_differs_from_extra_budget=True,extra_budgets=base.BALANCED_WEIGHTS,
        extra_synthetic_probability=float(sum(w for r,w in zip(training,weights) if r.get('is_synthetic')))))
    teacher=YOLO(str(root/'reference/parent.pt')).model.float().cuda().eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    epoch=start_epoch-1
    stopping_already_reached=epoch-best_epoch>=p['patience']
    epochs_to_run=[] if stopping_already_reached else range(start_epoch,p['epochs']+1)
    for epoch in epochs_to_run:
        if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
        model.train();indices=epoch_indices(len(training),weights,auxiliary_indices,rng)
        epoch_sampling=Counter();loss_sums=Counter();steps=0
        for offset in range(0,len(indices),p['batch']):
            if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
            part=indices[offset:offset+p['batch']];tensors=[];aux_targets=[];classes=[]
            for raw_index in part:
                i=int(raw_index);r=training[i]
                image,target=augment(images[i],targets[i],rng)
                tensors.append(image);aux_targets.append(target);classes.append(LABELS.index(r['label']))
                epoch_sampling[f"{r['origin']}|Cam{r.get('camera')}|{r['label']}"]+=1
                if r.get('is_synthetic'):epoch_sampling['synthetic']+=1
                if r['id'] in aux_by_id:epoch_sampling['auxiliary_reference']+=1
            x=torch.from_numpy(np.array(tensors)).cuda();y=torch.tensor(classes,device='cuda')
            target={k:torch.as_tensor(np.array([a[k] for a in aux_targets]),device='cuda') for k in aux_targets[0]}
            logits,auxiliary=model(x,return_aux=True)
            primary=F.cross_entropy(logits,y,weight=logits.new_tensor(p['loss']['class_weights']))
            with torch.no_grad():
                output=teacher(x);teacher_logits=output[1] if isinstance(output,(tuple,list)) else output
                teacher_probabilities=teacher_logits.softmax(-1)
            confident=(teacher_probabilities.argmax(1)==y)&(teacher_probabilities.max(1).values>=p['loss']['teacher_min_confidence'])
            kd=logits.sum()*0
            if confident.any():
                temperature=p['loss']['teacher_temperature']
                kd=F.kl_div(F.log_softmax(logits[confident]/temperature,-1),
                            F.softmax(teacher_logits[confident]/temperature,-1),reduction='batchmean')*temperature**2
            aux_losses=auxiliary_losses(auxiliary,target)
            loss=primary+p['loss']['teacher_kd']*kd
            if name=='auxiliary':
                loss=loss+sum(p['loss']['auxiliary_weights'][k]*v for k,v in aux_losses.items())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fit loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5.)
            optimizer.step();steps+=1
            loss_sums.update({k:float(v.detach()) for k,v in dict(total=loss,primary=primary,kd=kd,**aux_losses).items()})
        model.eval();predictions=predict_torch(model,validation,val_images);current=metrics(validation,predictions)
        rank=safety_rank(current,references)
        if learned_rank is None or rank>learned_rank:
            learned_rank=rank;learned_epoch=epoch
            atomic_model(folder/'learned_best.pt',model,epoch=epoch,rank=list(rank),epoch0_parent=False)
            atomic_json(folder/'learned_best_metrics.json',current)
        if rank>best_rank:
            best_rank=rank;best_epoch=epoch
            atomic_model(folder/'best.pt',model,epoch=epoch,rank=list(rank),epoch0_parent=False)
            atomic_json(folder/'best_metrics.json',current)
        model.assert_frozen(frozen);cumulative.update(epoch_sampling)
        summary=dict(epoch=epoch,best_epoch=best_epoch,learned_best_epoch=learned_epoch,rank=list(rank),best_rank=list(best_rank),
                     losses={k:v/max(1,steps) for k,v in loss_sums.items()},sampling=dict(epoch_sampling),
                     **{k:current['all'][k] for k in ['errors','ng_not_detected','false_ng','no_disk_as_ok']})
        atomic_json(folder/'progress.json',summary)
        with (folder/'history.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(summary)+'\n')
        atomic_torch(folder/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            best_epoch=best_epoch,best_rank=list(best_rank),learned_rank=list(learned_rank),learned_epoch=learned_epoch,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            cumulative_sampling=dict(cumulative),manifest_sha256=p['manifest_sha256'],auxiliary_sha256=p['auxiliary_sha256']))
        publish(root,'running',name,child=os.getpid(),epoch=epoch,best_epoch=best_epoch,metrics=summary)
        print(name,json.dumps({k:v for k,v in summary.items() if k!='sampling'}),flush=True)
        check(root)
        if epoch-best_epoch>=p['patience']:break
    model.assert_frozen(frozen)
    atomic_json(folder/'sampling_actual.json',dict(epochs=epoch,draws=sum(v for k,v in cumulative.items() if k not in ['synthetic','auxiliary_reference']),
                counts=dict(cumulative),train_only=True))
    atomic_json(folder/'completion.json',dict(epochs_completed=epoch,best_epoch=best_epoch,best_rank=list(best_rank),
                learned_best_epoch=learned_epoch,learned_best_rank=list(learned_rank) if learned_rank is not None else None,
                selected_is_epoch0_parent=best_epoch==0,frozen_backbone_bn_sha256=model.frozen_state_hash(),
                batchnorm_and_frozen_early_backbone_unchanged=True,fit_complete=True))


def embed_onnx_metadata(path,p):
    import onnx
    model=onnx.load(str(path))
    onnx.helper.set_model_props(model,dict(names=json.dumps(dict(enumerate(LABELS))),class_order=json.dumps(LABELS),
        preprocessing='RGB; cv2.INTER_LINEAR stretch224x224; float32 /255; NCHW',model_format=FORMAT,
        parent_pt_sha256=p['reference_hashes']['parent.pt'],canonical_revision='12',
        canonical_manifest_sha256=p['manifest_sha256'],auxiliary_annotation_sha256=p['auxiliary_sha256'],
        auxiliary_review='agent_drafts_pending_user_review'))
    onnx.checker.check_model(model);onnx.save(model,str(path))


def evaluate(root):
    p=check(root,verify_images=True);data=rows(root);validation=[r for r in data if r['split']=='val']
    results={};predictions_by_model={}
    for name in ['service','retained']:
        path=root/'reference'/f'{name}.onnx'
        predictions=predict_onnx(path,data,root);predictions_by_model[name]=predictions
        result=dict(model_path=str(path),model_sha256=digest(path),manifest_sha256=p['manifest_sha256'],metrics={},predictions=[])
        for split in ['train','val']:
            ix=[i for i,r in enumerate(data) if r['split']==split]
            result['metrics'][split]=metrics([data[i] for i in ix],[predictions[i] for i in ix])
        result['predictions']=[dict(id=r['id'],truth=r['label'],split=r['split'],origin=r['origin'],prediction=a) for r,a in zip(data,predictions)]
        atomic_json(root/'evaluations'/f'{name}.json',result);results[name]=result['metrics']
    for name in FIT_NAMES:
        folder=root/'jobs'/name
        completion=read(folder/'completion.json')
        if not completion['fit_complete']:raise ValueError('Fit incomplete')
        checkpoint_selection_guard(folder/'best.pt',completion['best_epoch'],completion['best_rank'])
        if completion['learned_best_epoch'] is not None:
            checkpoint_selection_guard(folder/'learned_best.pt',completion['learned_best_epoch'],completion['learned_best_rank'])
        model=load_model(folder/'best.pt','cpu')
        if model.frozen_state_hash()!=read(folder/'initialization.json')['frozen_state_sha256']:
            raise ValueError('Frozen backbone/BN state changed')
        export=folder/'best.onnx';export_model(model,export);embed_onnx_metadata(export,p)
        maximum=0.;wrong=[]
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4
        session=ort.InferenceSession(str(export),sess_options=options,providers=['CPUExecutionProvider'])
        with torch.inference_mode():
            for start in range(0,len(validation),32):
                part=validation[start:start+32];x=np.array([tensor(cv2.imread(r['path'])) for r in part])
                a=model(torch.from_numpy(x)).numpy();b=session.run(None,{'images':x})[0]
                maximum=max(maximum,float(np.abs(a-b).max()))
                wrong.extend(r['id'] for r,aa,bb in zip(part,a.argmax(1),b.argmax(1)) if aa!=bb)
        parity=dict(count=len(validation),argmax_mismatches=len(wrong),mismatched_ids=wrong,max_probability_error=maximum,
                    passed=not wrong and maximum<1e-3)
        atomic_json(folder/'parity.json',parity)
        if not parity['passed']:raise ValueError('Full-val PT/ONNX parity failed')
        predictions=predict_onnx(export,data,root);predictions_by_model[name]=predictions
        result=dict(model_path=str(export),model_sha256=digest(export),pt_sha256=digest(folder/'best.pt'),
                    manifest_sha256=p['manifest_sha256'],auxiliary_sha256=p['auxiliary_sha256'],metrics={},predictions=[])
        for split in ['train','val']:
            ix=[i for i,r in enumerate(data) if r['split']==split]
            result['metrics'][split]=metrics([data[i] for i in ix],[predictions[i] for i in ix])
        result['predictions']=[dict(id=r['id'],truth=r['label'],split=r['split'],origin=r['origin'],prediction=a) for r,a in zip(data,predictions)]
        atomic_json(root/'evaluations'/f'{name}.json',result);results[name]=result['metrics']
        atomic_json(folder/'best.json',dict(format=FORMAT,pt_sha256=result['pt_sha256'],onnx_sha256=result['model_sha256'],
            labels=LABELS,preprocessing='RGB/stretch224,float32/255',canonical_revision=12,manifest_sha256=p['manifest_sha256'],
            parent_pt_sha256=p['reference_hashes']['parent.pt'],auxiliary_sha256=p['auxiliary_sha256'],user_review_complete=False))
    failures=[];candidate=results['auxiliary']['val'];references=[results[n]['val'] for n in ['service','retained']]
    if candidate['all']['ng_not_detected']!=0:failures.append('val/zero_ng_required')
    if safety_rank(candidate,references)<=safety_rank(results['control']['val'],references):
        failures.append('auxiliary_not_strictly_better_than_matched_control')
    for reference in ['service','retained']:
        for domain,value in candidate.items():
            for key in ['ng_not_detected','false_ng','no_disk_as_ok','errors']:
                if value[key]>results[reference]['val'][domain][key]:
                    failures.append(f'val/{domain}/{key}_above_{reference}')
    fit_summaries={}
    for name in FIT_NAMES:
        folder=root/'jobs'/name;complete=read(folder/'completion.json')
        learned=read(folder/'learned_best_metrics.json')
        if list(safety_rank(learned,references))!=complete['learned_best_rank']:
            raise ValueError('Learned checkpoint metric/rank binding differs')
        fit_summaries[name]=dict(completion=complete,learned_validation_metrics=learned,
            learned_pt_sha256=digest(folder/'learned_best.pt'),selected_pt_sha256=digest(folder/'best.pt'),
            learned_metrics_scope='Cached PT validation at checkpoint selection, same frozen1615; not separately ONNX-exported')
    release=dict(recommended_for_review=not failures,recommendation_failures=failures,metrics=results,fit_summaries=fit_summaries,
                 selected_pt=str(root/'jobs/auxiliary/best.pt'),selected_onnx=str(root/'jobs/auxiliary/best.onnx'),
                 selection_is_epoch0_parent=read(root/'jobs/auxiliary/completion.json')['selected_is_epoch0_parent'],
                 production_deployed=False,validation_is_development=True,auxiliary_review_complete=False)
    atomic_json(root/'release_decision.json',release)
    lines=[page().split('<h1>')[0],'<h1>So sánh cùng canonical revision12</h1><p>Val dùng chọn checkpoint; train là fitted diagnostics. Nhãn hình học do agent, chưa được người dùng kiểm tra.</p><table><tr><th>Scope</th><th>Model</th><th>N</th><th>NG lọt</th><th>NG ảo</th><th>Trống→OK</th><th>Lỗi</th></tr>']
    for split in ['val','train']:
        for name,value in results.items():
            for domain,q in value[split].items():
                lines.append(f'<tr><td>{split}/{domain}</td><td>{name}</td><td>{q["count"]}</td><td>{q["ng_not_detected"]}</td><td>{q["false_ng"]}</td><td>{q["no_disk_as_ok"]}</td><td>{q["errors"]}</td></tr>')
    for name,summary in fit_summaries.items():
        for domain,q in summary['learned_validation_metrics'].items():
            epoch=summary['completion']['learned_best_epoch']
            lines.append(f'<tr><td>val/{domain}</td><td>{name} learned best epoch{epoch}</td><td>{q["count"]}</td><td>{q["ng_not_detected"]}</td><td>{q["false_ng"]}</td><td>{q["no_disk_as_ok"]}</td><td>{q["errors"]}</td></tr>')
    lines+=['</table><p>'+html.escape(', '.join(failures) or 'Đạt các gate accuracy; chờ White replay')+'</p></html>']
    atomic_text(root/'comparison.html',''.join(lines))
    cards=[]
    for i,r in enumerate(data):
        if not any(predictions_by_model[n][i]['label']!=r['label'] for n in predictions_by_model):continue
        description=' · '.join(f"{n}: {predictions_by_model[n][i]['label']}" for n in predictions_by_model)
        cards.append(f'<article><a href="input_snapshot/{r["image"]}"><img loading="lazy" src="input_snapshot/{r["image"]}"></a><p>{html.escape(r["id"])}</p><p>{r["split"]} · {r["label"]} · {html.escape(description)}</p></article>')
    atomic_text(root/'error_review.html',page().split('<h1>')[0]+'<h1>Các ca ít nhất một model sai</h1><p>Truth từ frozen canonical; không tự sửa nhãn. Train là fitted diagnostics.</p><div class="grid">'+''.join(cards)+'</div></html>')
    check(root)
    atomic_json(root/'evaluation_complete.json',dict(passed=True,release_sha256=digest(root/'release_decision.json'),
                model_sha256={name:digest(root/'jobs'/name/'best.onnx') for name in FIT_NAMES},reports_written=True))


def replay(root):
    p=check(root);import decouple
    from unittest.mock import patch
    decouple.config=decouple.Config(decouple.RepositoryEnv(str(PROJECT/'SERVICE/config/config.env')))
    os.chdir(PROJECT/'SERVICE')
    from src.communication.modbus_client import MbClient
    hardware_attempts=[]
    def forbid(*args,**kwargs):
        hardware_attempts.append(True)
        raise RuntimeError('Hardware forbidden in classifier replay')
    with patch.object(MbClient,'connect',forbid),patch.object(MbClient,'_MbClient__write_bit',forbid),patch.object(MbClient,'_MbClient__read_bit',forbid,create=True):
        from src.service.check_disk_service_yolo import DiskCheckingService
        from src.inference.yolo_classify.yolo_classifier import YoloClassifier
        from training.audit_new_factory_images import load
        service=DiskCheckingService();baseline=service.point_classification_model
        class CountingClassifier(YoloClassifier):
            def predict_batch(self,images):
                self.count+=len(images)
                return super().predict_batch(images)
        candidate=CountingClassifier(str(root/'jobs/auxiliary/best.onnx'),LABELS,input_size=baseline.input_size,batch_size=baseline.batch_size)
        candidate.count=0
        report=read(PROJECT/'.work/factory_audit_20261020_run/report.json')
        if len(report['white'])!=28:raise ValueError('Expected28White inputs')
        first=report['white'][0];im=load(Path(first['source']),first['camera'],Path(report['app_config']))
        for clf in [baseline,candidate]:
            service.point_classification_model=clf;service.check_disk_white(im.copy())
        records=[];debug_checked=set()
        for index,item in enumerate(report['white']):
            if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
            image=load(Path(item['source']),item['camera'],Path(report['app_config']));outputs={};times={}
            for mode in (['baseline','candidate'] if index%2==0 else ['candidate','baseline']):
                service.point_classification_model=baseline if mode=='baseline' else candidate
                candidate.count=0;started=time.perf_counter()
                outputs[mode]=service.check_disk_white(image.copy());times[mode]=(time.perf_counter()-started)*1000
                if mode=='candidate' and candidate.count!=100:raise ValueError('Expected100crop calls per White frame')
            a,b=outputs['candidate'],outputs['baseline']
            fields=['CropBox','UvBox1','UvBox2','Mid1','Mid2','MinDiskDistance','MaxDiskDistance','InferredSlots','PointGeometryStatus']
            if not all(a.model_dump()[k]==b.model_dump()[k] for k in fields):raise ValueError('White geometry/count changed')
            for mode,clf in [('baseline',baseline),('candidate',candidate)]:
                key=(item['camera'],mode)
                if key not in debug_checked:
                    service.point_classification_model=clf
                    debug=service.check_disk_debug(image.copy(),service._default_white_params())
                    if debug.FinalImg!=outputs[mode].ResImg or debug.Result!=outputs[mode].Result:
                        raise ValueError('Debug/production parity failed')
                    debug_checked.add(key)
            records.append(dict(source=item['source'],camera=item['camera'],baseline_ms=times['baseline'],candidate_ms=times['candidate'],
                                ratio=times['candidate']/times['baseline'],geometry_counts_unchanged=True,
                                baseline_result=b.Result,candidate_result=a.Result))
            atomic_json(root/'white_progress.json',records);print('WHITE',index+1,'/28 ratio',records[-1]['ratio'],flush=True)
    if hardware_attempts or len(records)!=28 or debug_checked!={(1,'baseline'),(1,'candidate'),(2,'baseline'),(2,'candidate')}:
        raise ValueError('Hardware-off replay checks incomplete')
    ratios=np.array([r['ratio'] for r in records]);summary={}
    for mode in ['baseline','candidate']:
        values=[r[mode+'_ms'] for r in records]
        summary[mode]=dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
    result=dict(frames=28,records=records,summary=summary,median_paired_ratio=float(np.median(ratios)),
                max_paired_ratio=float(ratios.max()),within_2x=bool((ratios<=2).all()),hardware_calls=False,
                geometry_counts_unchanged=True,both_camera_debug_parity=True,current_service_config=True,
                crop_width=service.classify_crop_width_ratio,batch_size=baseline.batch_size,
                scope='Whole software White afterwarmup, alternating paired order; includes detector/rectification/classifier/segmentation/caliper/rendering, excludes camera/PLC/network/image file load.28paired frames, not a worst-case latency guarantee.')
    release=read(root/'release_decision.json');failures=[f for f in release['recommendation_failures'] if f!='white_latency_over_2x']
    if not result['within_2x']:failures.append('white_latency_over_2x')
    release.update(recommendation_failures=failures,recommended_for_review=not failures,whole_white_latency=result)
    atomic_json(root/'release_decision.json',release)
    atomic_json(root/'white_replay.json',result);check(root)


def finalize(root):
    p=check(root,verify_images=True);current=audit(Path(p['pool']));atomic_json(root/'integrity_after.json',current)
    if not current['passed']:raise ValueError('Canonical audit after campaign failed')
    if current['manifest_sha256']!=p['manifest_sha256']:raise ValueError('Canonical manifest changed')
    annotations=read(root/'auxiliary_annotations.json')['records'];data=rows(root);by_id={r['id']:r for r in data}
    for annotation in annotations:
        row=by_id[annotation['canonical_id']]
        if row['split']!='train' or row['sha256']!=annotation['crop_sha256']:
            raise ValueError('Auxiliary train binding changed')
    for name in FIT_NAMES:
        folder=root/'jobs'/name
        if not read(folder/'completion.json')['fit_complete'] or not read(folder/'parity.json')['passed']:
            raise ValueError('Fit or full-val parity incomplete')
        model=load_model(folder/'best.pt')
        if model.frozen_state_hash()!=read(folder/'initialization.json')['frozen_state_sha256']:
            raise ValueError('Frozen early backbone/BN state changed')
    if read(root/'jobs/control/initialization.json')!=read(root/'jobs/auxiliary/initialization.json'):
        raise ValueError('Matched initializations differ')
    white=read(root/'white_replay.json')
    if white['frames']!=28 or white['hardware_calls'] or not white['both_camera_debug_parity'] or not white['geometry_counts_unchanged']:
        raise ValueError('White replay incomplete')
    release=read(root/'release_decision.json')
    if (not white['within_2x']) != ('white_latency_over_2x' in release['recommendation_failures']):
        raise ValueError('Latency gate/result binding differs')
    verification=dict(passed=True,verified_at=base.now(),canonical_revision=12,canonical_manifest_sha256=p['manifest_sha256'],
        exact_labels_ids_hashes_splits_preserved=True,canonical_integrity_passed=True,review_complete=False,
        auxiliary_count=48,auxiliary_review='agent_drafts_pending_user_review',auxiliary_only_canonical_train=True,
        auxiliary_sha256=p['auxiliary_sha256'],frozen_code_protocol_inputs_runtime_hashes_verified=True,
        all_runtime_and_retained_models_unchanged=True,fit_names=FIT_NAMES,
        fits={name:read(root/'jobs'/name/'completion.json') for name in FIT_NAMES},
        parity={name:read(root/'jobs'/name/'parity.json') for name in FIT_NAMES},
        white_replay=white,recommended_for_review=release['recommended_for_review'],production_deployed=False)
    lines=['# Groove-entry ownership pilot','',
           'Two matched local-crop fits complete. Auxiliary geometry is agent-drafted and awaits user review. No deployment.',
           '','| Scope | Model | N | Missed NG | False NG | Empty as OK | Errors |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for scope in ['val','train']:
        for name,value in release['metrics'].items():
            q=value[scope]['all'];lines.append(f"| {scope} | {name} | {q['count']} | {q['ng_not_detected']} | {q['false_ng']} | {q['no_disk_as_ok']} | {q['errors']} |")
    for name,summary in release['fit_summaries'].items():
        q=summary['learned_validation_metrics']['all'];complete=summary['completion']
        lines.append(f"| val | {name} learned best epoch{complete['learned_best_epoch']} | {q['count']} | {q['ng_not_detected']} | {q['false_ng']} | {q['no_disk_as_ok']} | {q['errors']} |")
    for name,summary in release['fit_summaries'].items():
        complete=summary['completion']
        lines+=['',f"{name}: selected epoch{complete['best_epoch']} rank{complete['best_rank']}; learned best epoch{complete['learned_best_epoch']} rank{complete['learned_best_rank']}. Learned metrics are cached PT development validation; selected checkpoint has full export parity."]
    lines+=['','Failures: '+(', '.join(release['recommendation_failures']) or 'None'),'',
            f"Whole White paired median ratio {white['median_paired_ratio']:.3f}x / max {white['max_paired_ratio']:.3f}x.",'',
            'Both fits use current +AI parent, same 224 local input, seed, trainable blocks, sampler and initialization. Epoch0 remains eligible. Early backbone and all BN running statistics remain unchanged.',
            'The auxiliary fit adds masked entry/trace/peak, visibility, occupancy and ownership losses. Estimated/unknown coordinates are excluded from point losses. An invisible groove is not classified empty merely because it is unseen.',
            'This initial pilot does not implement raw-context lattice reconstruction. Fully hidden geometry may remain unresolved. Agent annotation quality can affect results; the user has not verified all coordinates.',
            'All3778 canonical train crops are seen once per epoch plus3778balanced extra draws and192aux-reference draws,7748total. Actual source/camera/class/synthetic exposure is in each fit sampling_actual.json.',
            'The1615val is repeatedly used same-session spatial development with shared frames and prior parent exposure, not an independent test. Existing labels/splits/gates remain fixed; train metrics are fitted diagnostics.',
            'Full1615PT/ONNX parity and28current-config White hardware-off replay, both-camera debug/geometry checks, frozen input/code/protocol and runtime/retained model hashes passed. No production zero-miss guarantee.',
            'Review auxiliary_review.html and aux_review/index.html for provisional annotations; error_review.html shows all errors. No completed old campaign was resumed.']
    atomic_text(root/'RESULTS.md','\n'.join(lines)+'\n')
    atomic_text(root/'index.html',(root/'comparison.html').read_text(encoding='utf-8').replace('</html>',
                '<p><a href="RESULTS.md">RESULTS</a> · <a href="completion_verification.json">Completion audit</a> · <a href="auxiliary_review.html">Agent drafts</a> · <a href="aux_review/index.html">Full draft gallery</a></p></html>'))
    atomic_json(root/'completion_verification.json',verification)


def dashboard(root):
    with socket.socket() as sock:
        occupied=sock.connect_ex(('127.0.0.1',8808))==0
    if occupied:
        import urllib.request
        try:
            with urllib.request.urlopen('http://127.0.0.1:8808/protocol.sha256',timeout=3) as response:
                if response.read().decode().strip()!=digest(root/'protocol.json'):raise ValueError('Port8808 belongs to another report')
        except Exception as exc:raise RuntimeError('Dashboard port8808 occupied: '+repr(exc))
        return
    with (root/'dashboard.log').open('a',encoding='utf-8') as handle:
        process=subprocess.Popen([sys.executable,'-X','utf8','-u','-B','-m','http.server','8808','--bind','127.0.0.1','--directory',str(root)],
                                 cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT,creationflags=FLAGS)
    atomic_json(root/'dashboard_process.json',dict(pid=process.pid,port=8808,root=str(root),trainer=False))


def checked_child(root,phase,resume=False):
    script=root/'code_snapshot'/Path(__file__).name
    command=[sys.executable,'-X','utf8','-u','-B',str(script),'--phase',phase,'--root',str(root)]
    if resume:command.append('--resume')
    env=os.environ.copy();env['CLASSIFIER_CAMPAIGN_PID']=str(os.getpid());env['PYTHONIOENCODING']='utf-8'
    with (root/'logs'/f'{phase}.log').open('a',encoding='utf-8') as handle:
        process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT,creationflags=FLAGS)
        try:
            while process.poll() is None:
                if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
                current=read(root/'status.json')
                if current.get('phase')!=phase or current.get('active_pid')!=process.pid:
                    publish(root,'running',phase,child=process.pid)
                time.sleep(5)
            if process.returncode:raise RuntimeError(f'{phase} exited {process.returncode}; inspect {root/"logs"/(phase+".log")}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=20)


def campaign(root,resume=False,auxiliary_source=DEFAULT_AUX):
    inventory=guard_processes(root)
    if resume:
        if not (root/'protocol.json').exists():raise ValueError('Resume requires frozen protocol')
        check(root,verify_images=True)
        if not (root/'prepare_complete.json').exists():finish_prepare(root)
    else:
        prepare(root,auxiliary_source)
    lock=root/'.campaign.lock'
    if lock.exists():
        owner=read(lock)
        if any(r['pid']==owner.get('pid') for r in inventory):raise RuntimeError('Campaign lock owner is alive')
        lock.unlink()
    with lock.open('x',encoding='utf-8') as handle:json.dump(dict(pid=os.getpid(),created_at=base.now()),handle)
    try:
        dashboard(root)
        phases=[('control','jobs/control/completion.json'),('auxiliary','jobs/auxiliary/completion.json'),
                ('evaluate','evaluation_complete.json'),('replay','white_replay.json'),('finalize','completion_verification.json')]
        for phase,marker in phases:
            check(root)
            if (root/marker).exists():continue
            checked_child(root,phase,resume=resume)
        if not read(root/'completion_verification.json')['passed']:raise ValueError('Completion audit failed')
        check(root,verify_images=True)
        publish(root,'complete','matched fits/evaluation/export/White/audit complete; no deployment',completed_at=base.now())
    finally:
        if lock.exists() and read(lock).get('pid')==os.getpid():lock.unlink()


def self_test():
    """Masking, cv2 point mapping and flip identity, no canonical/run writes."""
    record=dict(native_size=[100,160],groove={
        'left_peak':dict(xy=[10,50],visibility='visible',coordinate_source='agent_visual_observed',uncertainty_px=2),
        'right_peak':dict(xy=[90,52],visibility='not_visible',coordinate_source='estimated_from_neighbors')},
        instances=[dict(id='a',insertion_end_xy=[30,42],trace_point_xy=[20,90],groove_assignment='left_neighbor',certainty='clear'),
                   dict(id='b',insertion_end_xy=[65,45],trace_point_xy=[80,95],groove_assignment='target',certainty='tentative')],
        target_count=dict(value=2,complete_review=True))
    target=annotation_targets(record,100,160)
    assert target['point_weight'][0]==.5 and target['point_weight'][1]==0
    assert target['ownership'].tolist()==[1,0] and target['count_weight']==.5
    flipped=flip_targets(target);twice=flip_targets(flipped)
    for key in target:
        if not np.allclose(target[key],twice[key],atol=1e-7):raise AssertionError('Flip does not preserve '+key)
    assert flipped['ownership'].tolist()==[0,2]
    estimate=deepcopy(record);estimate['target_count']['complete_review']=False
    assert annotation_targets(estimate,100,160)['count_weight']==0
    b=2;aux={k:torch.randn(shape,requires_grad=True) for k,shape in {
        'heatmap_logits':(b,6,28,28),'count_logits':(b,3),'visibility_logits':(b,2,4),'ownership_logits':(b,2,3)}.items()}
    unknown={k:torch.as_tensor(np.array([blank_targets()[k] for _ in range(b)])) for k in blank_targets()}
    losses=auxiliary_losses(aux,unknown)
    assert all(float(v.detach())==0 for v in losses.values())
    sum(losses.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)==0 for p in aux.values())
    rng=np.random.default_rng(1104);ix=epoch_indices(3778,np.full(3778,1/3778),np.arange(48),rng)
    assert len(ix)==7748 and set(ix)==set(range(3778))
    return dict(passed=True,flip_roundtrip=True,estimated_point_masked=True,incomplete_count_masked=True,
                unknown_auxiliary_loss_and_gradient_zero=True,epoch_draws=7748,every_train_present=True,
                receipt_created=False,run_created=False,full_fits=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--phase',choices=['campaign','prepare','control','auxiliary','evaluate','replay','finalize'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--aux-source',type=Path,default=DEFAULT_AUX)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2));return
    if args.phase is None:parser.error('--phase required unless --self-test')
    root=args.root.resolve();claimed=False
    try:
        guard_processes(root,allow_parent=args.phase!='campaign');claimed=True
        if args.phase=='campaign':campaign(root,args.resume,args.aux_source)
        elif args.phase=='prepare':finish_prepare(root) if args.resume else prepare(root,args.aux_source)
        elif args.phase in FIT_NAMES:fit(root,args.phase,args.resume)
        else:globals()[args.phase](root)
    except Exception as exc:
        if claimed and (root/'protocol.json').exists():
            publish(root,'stopped' if isinstance(exc,InterruptedError) else 'failed',args.phase,error=repr(exc))
        traceback.print_exc();raise


if __name__=='__main__':main()
