"""One official YOLO11s-cls fit, exact canonical split, NG-first selection, no deployment."""
import argparse
from copy import copy
import html,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0,str(PROJECT/'SERVICE'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
import train_classifier_canonical_holdout as base
from train_classifier_safety_campaign import safety_fitness
from classifier_dataset import CanonicalDataset,active_dataset,read,atomic_json,atomic_text
from train_classifier_supplementary import digest
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
USER_STATEMENT='hay thử tăng size model lên từ n sang s xem sao; không lọt NG với tốt nhất không được lỗi ảo nhiều'

def signature(rows):return sorted((r['id'],r['label'],r['sha256'],r['split']) for r in rows)
def totals(metrics):return {k:sum(metrics[o][k] for o in base.BALANCED_WEIGHTS) for k in ('errors','ng_not_detected','false_ng','no_disk_as_ok')}

class NGWeightedLoss:
    def __call__(self,preds,batch):
        logits=preds[1] if isinstance(preds,(tuple,list)) else preds
        loss=F.cross_entropy(logits,batch['cls'].long(),weight=logits.new_tensor([2.,1.,1.]))
        return loss,loss.detach()

class SizeSafetyValidator(base.HoldoutValidator):
    def get_stats(self):
        result=super().get_stats()
        metrics={d:{k:result[f'metrics/{d}_{k}'] for k in ('errors','ng_not_detected','false_ng','no_disk_as_ok')} for d in self.manifest['baseline_validation'] if d!='all'}
        result['fitness']=safety_fitness(metrics,self.manifest['baseline_validation'])
        return result

class SizeSafetyTrainer(base.HoldoutTrainer):
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg,weights,verbose);model.criterion=NGWeightedLoss();return model
    def get_validator(self):
        self.loss_names=['loss'];return SizeSafetyValidator(self.test_loader,self.save_dir,args=copy(self.args),_callbacks=self.callbacks)

def acceptance(candidate,baseline,n_safety,weights,domains):
    failures=base.comparison_failures(candidate,baseline,weights,domains)
    if totals(candidate)['ng_not_detected']!=0:failures.append('all/zero_ng_misses_required')
    for d in domains:
        if candidate[d]['false_ng']>n_safety[d]['false_ng']:failures.append(d+'/false_ng_above_n_safety')
    return failures

def dashboard(p):
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO11s-cls NG-first</title><style>body{font:16px system-ui;background:#14202e;color:#e6eef6;margin:28px;max-width:1400px}a{color:#9bd4ff}td,th{border:1px solid #63778d;padding:9px}table{border-collapse:collapse}p{line-height:1.5}</style><h1>YOLO11s-cls — ưu tiên không lọt NG, ít NG ảo</h1><p>Một fit ImageNet s; 300 epoch tối đa, patience 100; 224px; canonical revision 10, 3.778 train / 1.605 val. Không triển khai.</p><p>Best: NG regression, NG lọt, NG ảo, trống thành OK. Gate cũ giữ nguyên; thêm 0 NG lọt và NG ảo không hơn n ưu tiên NG theo từng nguồn/camera. Validation đã dùng phát triển, HA đối chiếu là retention; nhãn còn partial review.</p><p id="status">Đang tải…</p><p id="error"></p><p><a href="protocol.json">Protocol</a> · <a href="logs/train.log">Log train</a> · <a href="comparison.html">So sánh</a> · <a href="val_best_errors.html">Lỗi val</a> · <a href="RESULTS.md">Kết quả</a> · <a href="completion_verification.json">Kiểm tra hoàn tất</a></p><script>async function refresh(){try{let s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · epoch '+s.epoch+'/300 · best '+s.best_epoch;document.getElementById('error').textContent=s.error||''}catch(e){document.getElementById('error').textContent=e.message}}refresh();setInterval(refresh,10000)</script></html>'''

def check(root):
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
    p=read(root/'protocol.json')
    if digest(root/'protocol.json')!=(root/'protocol.sha256').read_text().strip():raise ValueError('Frozen protocol changed')
    if any(digest(root/'code_snapshot'/n)!=h for n,h in p['code_hashes'].items()):raise ValueError('Frozen code changed')
    if digest(root/'source_manifest_at_training.json')!=p['pool_manifest_sha256']:raise ValueError('Source manifest changed')
    if digest(root/'input_snapshot/manifest.json')!=p['input_manifest_sha256']:raise ValueError('Input manifest changed')
    if any(digest(f)!=h for f,h in read(root/'protected_runtime_hashes.json').items()):raise ValueError('Current runtime changed during experiment')
    return p

def prepare(root):
    if root.exists():raise FileExistsError(root)
    registry=read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json');pool=Path(registry['path']);before=read(pool/'manifest.json')
    receipt=dict(authorized_at=base.now(),canonical_revision=before['canonical_revision'],incomplete_review_acknowledged=True,user_statement=USER_STATEMENT,scope='One official ImageNet YOLO11s-cls fit, max300/patience100, NG-first checkpoint fitness and NG-weighted CE. Exact current canonical labels/splits, no new data, no deployment. Compare installed HA-n, ImageNet-n, fine-tune-n and NG-safety-n; full PT/ONNX parity and hardware-off 28-White replay.')
    CanonicalDataset(pool).authorize_training(receipt,before['canonical_revision'])
    active_dataset(required_split_policy=base.BALANCED_POLICY)
    assert signature(before['records'])==signature(read(pool/'manifest.json')['records'])
    old_protected=read(PROJECT/'.work/classifier_safety_campaign_20261022/protected_runtime_hashes.json')
    paths={Path(s) for s in old_protected}
    paths.update([PROJECT/'SERVICE/config/config.env',PROJECT/'SERVICE/src/service/base_service.py',PROJECT/'SERVICE/src/service/check_disk_service_yolo.py',PROJECT/'SERVICE/src/inference/yolo_classify/yolo_classifier.py'])
    import decouple
    runtime=Path(read(PROJECT/'.work/factory_audit_20261020_run/report.json')['runtime'])
    for config_path,basepath in [(runtime/'config/config.env',runtime),(PROJECT/'SERVICE/config/config.env',PROJECT/'SERVICE')]:
        cfg=decouple.Config(decouple.RepositoryEnv(str(config_path)))
        for key in ('POINT_CLASSIFY_MODEL_PATH','DISK_POINT_DETECT_MODEL_PATH','DISK_SEGMENT_MODEL_PATH'):
            model_path=Path(cfg(key));paths.add(model_path if model_path.is_absolute() else basepath/model_path)
    paths.add(Path(read(PROJECT/'.work/factory_audit_20261020_run/report.json')['app_config']))
    paths.update(PROJECT/'SERVICE'/f for f in ['src/dtos/meta.py','src/tools/slot_geometry.py','src/inference/yolo_detect/onnx_detector.py','src/inference/yolo_segment/onnx_segmentor.py','src/tools/caliper_advanced_new.py','training/smoke_local_white.py','training/audit_new_factory_images.py'])
    protected={str(f):digest(f) for f in paths}
    p=base.prepare(root,base.BALANCED_POLICY,'imagenet',pretrained_override=PROJECT/'.work/pretrained_ultralytics/yolo11s-cls.pt',protected_runtime=protected)
    rows=read(root/'input_snapshot/manifest.json')['records'];nroot=PROJECT/'.work/classifier_safety_campaign_20261022'
    assert signature(rows)==signature(read(nroot/'source_manifest_at_training.json')['records'])
    (root/'evaluations').mkdir(exist_ok=True)
    references={}
    for key,old_name in [('baseline','baseline'),('previous_imagenet','previous_imagenet'),('n_finetune','previous_finetune_a100'),('n_safety','candidate')]:
        old=read(nroot/'evaluations'/f'{old_name}.json');assert {x['id'] for x in old['predictions']}=={r['id'] for r in rows}
        by_id={x['id']:x for x in old['predictions']}
        assert all(by_id[r['id']]['truth']==r['label'] and by_id[r['id']]['split']==r['split'] for r in rows)
        source=Path(old['model_path']);assert digest(source)==old['model_sha256']
        target=root/'reference'/f'{key}.onnx';shutil.copy2(source,target)
        old.update(model_path=str(target),predictions=[by_id[r['id']] for r in rows]);atomic_json(root/'evaluations'/f'{key}.json',old)
        references[key]=dict(path=str(target),sha256=digest(target),source_run=str(nroot),evaluation=str(root/'evaluations'/f'{key}.json'))
    baseline=read(root/'evaluations/baseline.json')
    p.update(model_name='yolo11s-cls',architecture='yolo11s-cls.yaml',initialization='imagenet',freeze_bn=False,lr0=.0003,
        checkpoint_selection='NG-priority: first NG regression vs installed baseline by source/camera, then NG misses, false NG, no_disk-as-OK, total errors.',
        fitness_formula='-1e10*NG_excess -1e7*NG_misses -1000*false_NG -no_disk_as_OK -.001*errors',
        loss='Cross entropy class weights NG/no_disk/OK = 2/1/1; no KD, no synthetic data.',
        comparison_note='Same canonical inputs, split, seed, input size and sampler as fresh ImageNet-n. Checkpoint objective and NG loss weighting differ to follow the current NG-first instruction; comparison is not a size-only causal ablation.',
        baseline_validation=baseline['metrics']['val'],comparison_models=references,previous_imagenet_onnx=references['previous_imagenet']['path'],previous_imagenet_sha256=references['previous_imagenet']['sha256'],
        recommendation_policy='Existing per-source/camera gates unchanged; additionally require zero validation NG misses and false NG no higher than NG-safety-n for each source/camera. No deployment.',
        full_fits_max=1,ha_validation_is_independent_test=False,authorization_receipt=receipt)
    fm=read(root/'input_snapshot/manifest.json');fm['baseline_validation']=p['baseline_validation'];atomic_json(root/'input_snapshot/manifest.json',fm)
    p['input_manifest_sha256']=digest(root/'input_snapshot/manifest.json')
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'));atomic_json(root/'authorization_receipt.json',receipt);atomic_text(root/'index.html',dashboard(p))
    oldpointer=read(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json');atomic_json(root/'previous_active_training_pointer.json',oldpointer)
    atomic_json(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json',dict(path=str(root),report_url='http://127.0.0.1:8798/',source_dataset=str(pool),canonical_revision=p['canonical_revision'],status='prepared',phase='s inputs frozen; NG-first selection',entrypoint='SERVICE/training/train_classifier_size_s.py',campaign_pid=os.getpid(),active_pid=None,authorization=receipt,production_deployed=False))
    atomic_json(root/'start_verification.json',dict(canonical_revision=p['canonical_revision'],labels_hashes_splits_preserved=True,train_count=p['train_count'],val_count=p['val_count'],canonical_integrity_passed=read(root/'integrity_before.json')['passed'],model_size='s',pretrained_sha256=p['pretrained_sha256'],no_deployment=True))
    return p

def fit(root):
    p=check(root)
    if digest(root/'input_snapshot/manifest.json')!=p['input_manifest_sha256']:raise ValueError('Input manifest changed')
    base.train(root,trainer_class=SizeSafetyTrainer)

def evaluate(root):
    p=check(root);base.evaluate(root)
    named={k:read(root/'evaluations'/f'{k}.json') for k in ['baseline','previous_imagenet','n_finetune','n_safety','candidate']}
    release=read(root/'release_decision.json');failures=acceptance(named['candidate']['metrics']['val'],named['baseline']['metrics']['val'],named['n_safety']['metrics']['val'],p['fitness_weights'],p['comparison_domains'])
    release.update(recommendation_failures=failures,recommended_for_review=not failures,model_name='yolo11s-cls',ng_priority=True,comparison_models={k:v['metrics'] for k,v in named.items()},installed_baseline_retained=True)
    atomic_json(root/'release_decision.json',release)
    page=dashboard(p).replace('</html>','')+f'<h2>Kết quả</h2><p>Qua gate rà tiếp: {not failures}. Chưa triển khai.</p><p>{html.escape(", ".join(failures) or "Không có gate lỗi")}</p><table><tr><th>Nguồn/camera</th><th>Model</th><th>Lỗi</th><th>NG lọt</th><th>NG ảo</th><th>Trống → OK</th></tr>'
    for d in p['comparison_domains']:
        for name,v in named.items():
            q=v['metrics']['val'][d];page+=f'<tr><td>{d}</td><td>{name}</td><td>{q["errors"]}</td><td>{q["ng_not_detected"]}</td><td>{q["false_ng"]}</td><td>{q["no_disk_as_ok"]}</td></tr>'
    atomic_text(root/'comparison.html',page+'</table></html>')
    cards=[]
    for r in named['candidate']['predictions']:
        if r['split']=='val' and r['truth']!=r['prediction']['label']:
            meta=next(x for x in read(root/'input_snapshot/manifest.json')['records'] if x['id']==r['id']);src='input_snapshot/'+meta['image']
            cards.append(f'<article><img src="{html.escape(src)}"><p>{html.escape(r["id"])}</p><p>Truth {r["truth"]} → {r["prediction"]["label"]} ({r["prediction"]["confidence"]:.3f})</p></article>')
    atomic_text(root/'val_best_errors.html','<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;background:#14202e;color:white}.grid{display:flex;flex-wrap:wrap;gap:14px}article{width:260px;overflow-wrap:anywhere}img{width:220px;height:190px;object-fit:contain}</style><h1>Lỗi validation YOLO11s-cls</h1><p>Truth frozen revision 10; development validation, không phải test độc lập.</p><div class="grid">'+''.join(cards)+'</div>')

def replay(root):
    check(root)
    sys.path.append(str(PROJECT/'SERVICE/.venv/Lib/site-packages'))
    import decouple
    from unittest.mock import patch
    from training.audit_new_factory_images import load,save,decode
    audit=read(PROJECT/'.work/factory_audit_20261020_run/report.json');runtime=Path(audit['runtime']);decouple.config=decouple.Config(decouple.RepositoryEnv(str(runtime/'config/config.env')));os.chdir(runtime)
    from src.communication.modbus_client import MbClient
    def forbid(*args,**kwargs):raise RuntimeError('Hardware call forbidden during replay')
    with patch.object(MbClient,'connect',forbid),patch.object(MbClient,'_MbClient__write_bit',forbid),patch.object(MbClient,'_MbClient__read_bit',forbid,create=True):
        from src.service.check_disk_service_yolo import DiskCheckingService
        from src.inference.yolo_classify.yolo_classifier import YoloClassifier
        service=DiskCheckingService();baseline=service.point_classification_model;candidate=YoloClassifier(str(root/'artifacts/best.onnx'),base.LABELS,batch_size=2);records=[];debug_checked=set()
        assert digest(runtime/'config/models/c_p.onnx')==digest(root/'reference/baseline.onnx')
        for old in audit['white']:
            image=load(Path(old['source']),old['camera'],Path(audit['app_config']))
            service.point_classification_model=baseline;b=service.check_disk_white(image.copy())
            service.point_classification_model=candidate;a=service.check_disk_white(image.copy())
            fields=('CropBox','UvBox1','UvBox2','Mid1','Mid2','MinDiskDistance','MaxDiskDistance','InferredSlots','PointGeometryStatus')
            assert all(a.model_dump()[k]==b.model_dump()[k] for k in fields),(old['source'],'geometry mismatch')
            matched=None
            if old['camera'] not in debug_checked:
                dbg=service.check_disk_debug(image.copy(),service._default_white_params());assert dbg.FinalImg==a.ResImg and dbg.Result==a.Result
                debug_checked.add(old['camera']);matched=True
            out=root/'white_replay'/old['folder']/'result.png';save(out,decode(a.ResImg))
            records.append(dict(camera=old['camera'],source=old['source'],baseline_white_result=b.Result,candidate=a.model_dump(exclude={'ResImg'}),geometry_and_counts_unchanged=True,debug_production_match=matched,image=out.relative_to(root).as_posix()))
            atomic_json(root/'white_replay_progress.json',records);print('WHITE',len(records),old['camera'],b.Result,'->',a.Result,flush=True)
    check(root);assert len(records)==28 and debug_checked=={1,2}
    atomic_json(root/'white_replay.json',dict(records=records,all_geometry_and_counts_unchanged=True,both_camera_debug_production_parity=True,hardware_calls=False,production_deployed=False,baseline='Current source flow and installed runtime config, rerun immediately before candidate on each frame.',scope='Integration only, no whole-tray ground truth.'))

def finalize(root):
    p=check(root);rows=[dict(r,path=str(root/'input_snapshot'/r['image'])) for r in read(root/'input_snapshot/manifest.json')['records']];base.validate_rows(rows,base.BALANCED_POLICY)
    if any(digest(r['path'])!=r['sha256'] for r in rows):raise ValueError('Frozen image changed')
    from evaluate_classifier_ha import YoloClassifier
    model=YOLO(str(root/'artifacts/best.pt')).model.float().cpu().eval();classifier=YoloClassifier(str(root/'artifacts/best.onnx'),base.LABELS);torch.set_num_threads(4)
    maximum=0.;wrong=[];val=[r for r in rows if r['split']=='val']
    with torch.inference_mode():
        for start in range(0,len(val),32):
            part=val[start:start+32];batch=classifier._preprocess_batch([cv2.imread(r['path']) for r in part],(224,224));a=model(torch.from_numpy(batch));a=(a[0] if isinstance(a,(tuple,list)) else a).numpy();b=classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:batch})[0]
            maximum=max(maximum,float(np.max(np.abs(a-b))));wrong.extend(r['id'] for r,x,y in zip(part,a.argmax(1),b.argmax(1)) if x!=y)
    parity=dict(count=len(val),argmax_mismatches=len(wrong),max_probability_error=maximum,passed=not wrong and maximum<1e-3);atomic_json(root/'artifacts/full_validation_export_parity.json',parity);assert parity['passed']
    images=[cv2.imread(r['path']) for r in val[:2]];latency={}
    for name,onnx in [('HA_n',root/'reference/baseline.onnx'),('s',root/'artifacts/best.onnx')]:
        clf=YoloClassifier(str(onnx),base.LABELS,batch_size=2)
        for _ in range(3):clf.predict_batch(images)
        times=[]
        for _ in range(30):
            start=time.perf_counter();out=clf.predict_batch(images);assert out[0] is not None;times.append((time.perf_counter()-start)*1000)
        latency[name]=dict(batch_size=2,iterations=30,median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),scope='Two-crop CPU preprocess+ORT, not whole-frame pipeline latency')
    atomic_json(root/'artifacts/cpu_latency.json',latency)
    white=read(root/'white_replay.json');assert len(white['records'])==28 and not white['hardware_calls']
    integrity=base.audit(Path(p['pool']));atomic_json(root/'integrity_after.json',integrity);assert integrity['passed'];check(root)
    release=read(root/'release_decision.json');completion=read(root/'artifacts/completion.json')
    atomic_json(root/'completion_verification.json',dict(passed=True,verified_at=base.now(),canonical_revision_at_start=p['canonical_revision'],current_canonical_revision=read(Path(p['pool'])/'manifest.json')['canonical_revision'],frozen_inputs_used=True,frozen_hashes_verified=True,canonical_integrity_passed=True,runtime_hashes_unchanged=True,white_count=28,hardware_calls=False,full_validation_export_parity=parity,recommended_for_review=release['recommended_for_review'],production_deployed=False))
    lines=['# YOLO11s-cls ưu tiên NG','',f"Hoàn tất {completion['epochs_completed']} epoch; best {completion['best_checkpoint']['epoch']}. Qua gate rà tiếp: {release['recommended_for_review']}. Chưa triển khai.",'','| Model | Lỗi | NG lọt | NG ảo | Trống → OK |','| --- | ---: | ---: | ---: | ---: |']
    for name,m in release['comparison_models'].items():
        q=totals(m['val']);lines.append(f"| {name} | {q['errors']} | {q['ng_not_detected']} | {q['false_ng']} | {q['no_disk_as_ok']} |")
    lines+=['','Gate chưa đạt: '+(', '.join(release['recommendation_failures']) or 'Không có'),'','Validation đã dùng phát triển, không phải test độc lập. Review còn partial. So sánh thay cả size và objective/loss theo ưu tiên NG; không quy toàn bộ khác biệt cho size.','Giữ installed HA-n. PT/ONNX toàn bộ val, hardware-off 28 White và hash runtime đã kiểm tra.']
    atomic_text(root/'RESULTS.md','\n'.join(lines)+'\n');atomic_text(root/'index.html',(root/'comparison.html').read_text(encoding='utf8'));base.publish(root,'complete','s training/evaluation/parity/White replay complete; no deployment')

def campaign(root,resume=False):
    cmd="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,creationflags=FLAGS,check=True).stdout.strip();processes=json.loads(raw) if raw else [];processes=processes if isinstance(processes,list) else [processes]
    active=[x for x in processes if x['ProcessId']!=os.getpid() and 'train_classifier_' in (x.get('CommandLine') or '') and '--phase' in x['CommandLine']]
    if active:raise RuntimeError('Another classifier trainer is alive: '+str(active))
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
    if resume and read(root/'status.json')['status']=='complete':raise ValueError('Do not resume completed run')
    p=check(root) if resume else prepare(root);script=root/'code_snapshot'/Path(__file__).name
    for phase,done in [('fit','artifacts/completion.json'),('evaluate','release_decision.json'),('replay','white_replay.json'),('finalize','completion_verification.json')]:
        if not (root/done).exists():base.checked(root,phase,[sys.executable,'-X','utf8','-u','-B',str(script),'--phase',phase,'--root',str(root)],root/'logs'/f'{"train" if phase=="fit" else phase}.log')
    if read(root/'status.json')['status']!='complete':base.publish(root,'complete','s evaluation complete; no deployment')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=['campaign','fit','evaluate','replay','finalize'],required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args();root=args.root.resolve()
    try:
        if args.phase=='campaign':campaign(root,args.resume)
        else:globals()[args.phase](root)
    except Exception as exc:
        if root.exists():base.publish(root,'stopped' if isinstance(exc,InterruptedError) else 'failed',args.phase,error=repr(exc))
        traceback.print_exc();raise
if __name__=='__main__':main()
