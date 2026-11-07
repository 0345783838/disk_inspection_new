"""One supplementary nano fit on revision11, train-only accepted AI crops, no deployment."""
import argparse
from collections import Counter, defaultdict
from copy import copy
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0,str(PROJECT/'SERVICE'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
import train_classifier_canonical_holdout as base
import train_classifier_size_s as shared
from train_classifier_safety_campaign import safety_fitness
from classifier_dataset import CanonicalDataset,active_dataset,atomic_json,atomic_text,read
from train_classifier_supplementary import digest
from src.inference.yolo_classify.yolo_classifier import YoloClassifier

USER_STATEMENT='ok cho vào train bổ sung đi'
SYNTHETIC_NG_BUCKET_SHARE=.15
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0

def signature(rows):
    return sorted((r['id'],r['label'],r['sha256'],r['split']) for r in rows)

def report_metrics(metrics,domain):
    # The cached 'all' classification_metrics has no derived errors/false_ng.
    return shared.totals(metrics) if domain=='all' else metrics[domain]

def supplementary_weights(rows):
    """Keep source/camera/class budgets; AI gets 15% only within its NG buckets."""
    if any(r['split']!='train' for r in rows):raise ValueError('Validation entered training sampler')
    original=base.sampling_weights(rows,base.BALANCED_POLICY)
    bucket=lambda r:(r['origin'],r.get('camera'),r['label'])
    masses=defaultdict(float);counts=Counter()
    for r,w in zip(rows,original):
        if r.get('is_synthetic') and r['label']!='ng':raise ValueError('Only accepted NG synthetic rows expected')
        masses[bucket(r)]+=w;counts[bucket(r),bool(r.get('is_synthetic'))]+=1
    weights=[]
    for r in rows:
        key=bucket(r);synthetic=bool(r.get('is_synthetic'));has_synthetic=counts[key,True]>0
        if has_synthetic and not counts[key,False]:raise ValueError('Synthetic bucket lacks real training anchors')
        share=(SYNTHETIC_NG_BUCKET_SHARE if synthetic else 1-SYNTHETIC_NG_BUCKET_SHARE) if has_synthetic else 1.
        weights.append(masses[key]*share/counts[key,synthetic])
    return weights

class SupplementaryTrainer(shared.SizeSafetyTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        if rank!=-1:raise ValueError('Single GPU only')
        dataset=self.build_dataset(dataset_path,mode)
        rows=[dataset.record_by_path[str(Path(s[0]).resolve())] for s in dataset.samples]
        weights=supplementary_weights(rows)
        atomic_json(self.save_dir/'sampling.json',dict(training_count=len(rows),
            synthetic_count=sum(bool(r.get('is_synthetic')) for r in rows),
            synthetic_probability=sum(w for r,w in zip(rows,weights) if r.get('is_synthetic')),
            synthetic_share_within_present_ng_bucket=SYNTHETIC_NG_BUCKET_SHARE,
            domain_probabilities={o:sum(w for r,w in zip(rows,weights) if r['origin']==o) for o in base.BALANCED_WEIGHTS},
            class_probabilities={c:sum(w for r,w in zip(rows,weights) if r['label']==c) for c in base.LABELS},
            validation_in_sampler=False))
        generator=torch.Generator().manual_seed(self.args.seed)
        return InfiniteDataLoader(dataset,batch_size=min(batch_size,len(dataset)),
            sampler=WeightedRandomSampler(weights,len(dataset),replacement=True,generator=generator),
            num_workers=self.args.workers,pin_memory=True,worker_init_fn=seed_worker,generator=generator)

def dashboard():
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Fine-tune nano + 10 ảnh AI</title><style>body{font:16px system-ui;background:#14202e;color:#e6eef6;margin:28px;max-width:1400px}a{color:#9bd4ff}td,th{border:1px solid #63778d;padding:9px}table{border-collapse:collapse}p{line-height:1.6}img{height:210px;max-width:250px;object-fit:contain}article{width:280px;overflow-wrap:anywhere;padding:14px;background:#26394b}.grid{display:flex;flex-wrap:wrap;gap:14px}</style><h1>Fine-tune YOLO11n-cls + 10 ảnh AI cùng khe</h1><p>Khởi tạo bản fine-tune best111 đã giữ lại · canonical revision11 · 3.788 train / 1.605 val · chỉ thêm 10 train NG. 300 epoch tối đa / patience100 · lr3e-6 · toàn bộ weights học, thống kê BatchNorm đóng băng · CE NG/no_disk/OK 2/1/1.</p><p>Ưu tiên giữ NG, rồi NG lọt, NG ảo và trống thành OK. Giữ ngân sách nguồn/camera/class; ảnh AI nhận 15% trong ba nhóm NG có AI (2,7% tổng xác suất lấy mẫu). So best đã học với parent epoch0 theo tiêu chí khai báo trước. Không triển khai. Validation đã dùng phát triển, HA là retention; rà toàn bộ nhãn vẫn còn một phần chưa xong.</p><p id="status">Đang tải…</p><p id="error"></p><p><a href="protocol.json">Protocol</a> · <a href="logs/train.log">Log train</a> · <a href="comparison.html">So sánh</a> · <a href="val_best_errors.html">Lỗi validation</a> · <a href="diagnostic_7_cases.html">7 NG cũ</a> · <a href="synthetic_train.html">10 ảnh AI trong train</a> · <a href="RESULTS.md">Kết quả</a></p><script>async function refresh(){try{let s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · epoch '+s.epoch+'/300 · best đã học '+s.best_epoch;document.getElementById('error').textContent=s.error||''}catch(e){document.getElementById('error').textContent=e.message}}refresh();setInterval(refresh,10000)</script></html>'''

def check(root):
    return shared.check(root)

def frozen_rows(root):
    return [dict(r,path=str(root/'input_snapshot'/r['image'])) for r in read(root/'input_snapshot/manifest.json')['records']]

def prepare(root):
    registry=read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json');pool=Path(registry['path']);before=read(pool/'manifest.json')
    adoption=read(PROJECT/'.work/classifier_ai_same_slot_adoption_20261029/adoption_verification.json')
    assert before['canonical_revision']==adoption['canonical_revision']==11
    assert len(before['records'])==5393 and sum(bool(r.get('is_synthetic')) for r in before['records'])==10
    old=read(PROJECT/'.work/classifier_finetune_ha_7030_20261022/source_manifest_at_training.json')
    assert signature([r for r in before['records'] if not r.get('is_synthetic')])==signature(old['records'])
    assert signature([r for r in before['records'] if r['split']=='val'])==signature([r for r in old['records'] if r['split']=='val'])
    receipt=dict(authorized_at=base.now(),canonical_revision=11,incomplete_review_acknowledged=True,
        user_statement=USER_STATEMENT,scope='Accept ten previewed AI NG crops into canonical train with root train groups; one supplementary YOLO11n fine-tune from retained best111, max300/patience100, lr3e-6, BN statistics frozen, NG-weighted CE. Exact 1605 original val unchanged. NG-first checkpoint selection, epoch0 parent guard. No deployment; full parity, 28 hardware-off White, whole software White <=2x baseline.')
    CanonicalDataset(pool).authorize_training(receipt,11)
    active_dataset(required_split_policy=base.BALANCED_POLICY)
    protected=read(PROJECT/'.work/classifier_spatial_ng_20261029/protected_runtime_hashes.json')
    assert all(digest(f)==sha for f,sha in protected.items())
    parent=PROJECT/'SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.pt'
    p=base.prepare(root,base.BALANCED_POLICY,'retained',pretrained_override=parent,protected_runtime=protected)
    rows=frozen_rows(root);(root/'evaluations').mkdir(exist_ok=True)
    paths={'baseline':root/'reference/baseline.onnx',
           'fine_tune':PROJECT/'SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.onnx',
           'n_safety':PROJECT/'.work/classifier_safety_campaign_20261022/artifacts/best.onnx'}
    references={}
    for name,source in paths.items():
        target=root/'reference'/f'{name}.onnx'
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
        result=base.evaluate_model(target,rows,root/'evaluations'/f'{name}.json')
        references[name]=dict(path=str(target),sha256=digest(target),metrics=result['metrics'])
    diagnostic=[r['id'] for r,pred in zip(rows,read(root/'evaluations/fine_tune.json')['predictions'])
                if r['split']=='val' and r['label']=='ng' and pred['prediction']['label']!='ng']
    assert len(diagnostic)==7
    p.update(model_name='yolo11n-cls',initial_classes=3,freeze_bn=True,lr0=3e-6,
        baseline_validation=references['baseline']['metrics']['val'],comparison_models=references,
        checkpoint_selection='NG excess vs installed baseline by source/camera, NG misses, false NG, empty-as-OK, errors; strict improvement over parent required to replace epoch0.',
        fitness_formula='-1e10*NG_excess -1e7*NG_misses -1000*false_NG -no_disk_as_OK -.001*errors',
        sampling='HA/new/legacy 40/40/20; equal HA/new cameras; class30/30/40, legacy70/15/15. Within each of the three NG buckets with AI: 15% synthetic /85% real, 2.7% AI total.',
        synthetic_ng_bucket_share=SYNTHETIC_NG_BUCKET_SHARE,synthetic_train_count=10,
        loss='Cross entropy NG/no_disk/OK = 2/1/1; no distillation; runtime geometry-preserving train augmentation.',
        selection_guard='Compare best learned checkpoint against unchanged retained parent with same safety fitness; keep parent on tie or regression. Report learned result separately.',
        recommendation_policy='Unchanged installed-baseline per-source/camera gates plus zero NG misses and false NG no higher than retained fine-tune per source/camera; <=2x whole software White. No deployment.',
        diagnostic_val_ids=diagnostic,validation_original_1605_unchanged=True,full_fits_max=1,
        ha_validation_is_independent_test=False,ha_validation_is_retention=True,authorization_receipt=receipt,
        retained_parent_sha256=digest(parent),adoption_verification=adoption)
    fm=read(root/'input_snapshot/manifest.json');fm['baseline_validation']=p['baseline_validation'];atomic_json(root/'input_snapshot/manifest.json',fm)
    p['input_manifest_sha256']=digest(root/'input_snapshot/manifest.json')
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'))
    atomic_json(root/'authorization_receipt.json',receipt);atomic_text(root/'index.html',dashboard())
    atomic_json(root/'previous_active_training_pointer.json',read(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'))
    atomic_json(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json',dict(path=str(root),report_url='http://127.0.0.1:8802/',source_dataset=str(pool),canonical_revision=11,status='prepared',phase='supplementary nano inputs frozen',entrypoint='SERVICE/training/train_classifier_same_slot_ai.py',campaign_pid=os.getpid(),active_pid=None,authorization=receipt,production_deployed=False))
    atomic_json(root/'start_verification.json',dict(canonical_revision=11,train_count=3788,val_count=1605,
        previous_5383_records_preserved=True,ten_synthetic_train_only=True,validation_original_1605_unchanged=True,
        integrity_passed=read(root/'integrity_before.json')['passed'],review_remains_partial=True,production_deployed=False))
    return p

def fit(root):
    check(root);base.train(root,trainer_class=SupplementaryTrainer)

def evaluate(root):
    p=check(root);rows=frozen_rows(root)
    for ext in ['pt','onnx']:
        dest=root/'artifacts'/f'learned_best.{ext}'
        if not dest.exists():shutil.copy2(root/'artifacts'/f'best.{ext}',dest)
    learned=base.evaluate_model(root/'artifacts/learned_best.onnx',rows,root/'evaluations/learned_candidate.json')
    parent=read(root/'evaluations/fine_tune.json')
    learned_fitness=safety_fitness(learned['metrics']['val'],p['baseline_validation'])
    parent_fitness=safety_fitness(parent['metrics']['val'],p['baseline_validation'])
    selected_parent=learned_fitness<=parent_fitness
    completion=read(root/'artifacts/completion.json')
    if 'learned_best_checkpoint' not in completion:completion['learned_best_checkpoint']=completion['best_checkpoint']
    for ext in ['pt','onnx']:
        source=(Path(p['pretrained']) if ext=='pt' else root/'reference/fine_tune.onnx') if selected_parent else root/'artifacts'/f'learned_best.{ext}'
        shutil.copy2(source,root/'artifacts'/f'best.{ext}')
    if selected_parent:completion['best_checkpoint']=dict(epoch=0,fitness=parent_fitness)
    completion['selected_is_unchanged_parent']=selected_parent
    atomic_json(root/'artifacts/completion.json',completion)
    atomic_json(root/'selection.json',dict(selected_is_unchanged_parent=selected_parent,parent_fitness=parent_fitness,learned_fitness=learned_fitness,learned_best_checkpoint=completion['learned_best_checkpoint'],parent_validation=parent['metrics']['val'],learned_validation=learned['metrics']['val']))
    base.evaluate(root)
    candidate=read(root/'evaluations/candidate.json');baseline=read(root/'evaluations/baseline.json');safety=read(root/'evaluations/n_safety.json')
    failures=base.comparison_failures(candidate['metrics']['val'],baseline['metrics']['val'],p['fitness_weights'],p['comparison_domains'])
    if shared.totals(candidate['metrics']['val'])['ng_not_detected']!=0:failures.append('all/zero_ng_misses_required')
    for d in p['comparison_domains']:
        if candidate['metrics']['val'][d]['false_ng']>parent['metrics']['val'][d]['false_ng']:failures.append(d+'/false_ng_above_fine_tune')
    release=read(root/'release_decision.json');release.update(recommendation_failures=failures,recommended_for_review=not failures,
        fine_tune=parent['metrics'],n_safety=safety['metrics'],learned=learned['metrics'],selected_is_unchanged_parent=selected_parent)
    atomic_json(root/'release_decision.json',release)
    page=dashboard().replace('</html>','')+'<h2>Validation gốc: 1.605 ảnh</h2><p>Chọn parent epoch0: '+str(selected_parent)+'. Gate: '+html.escape(', '.join(failures) or 'Đạt')+'</p><table><tr><th>Nguồn/camera</th><th>Model</th><th>Lỗi</th><th>NG lọt</th><th>NG ảo</th><th>Trống → OK</th></tr>'
    for domain in ['all']+p['comparison_domains']:
        for name,result in [('HA đang cài',baseline),('Fine-tune trước',parent),('n-safety',safety),('AI best đã học',learned),('Model được giữ',candidate)]:
            m=report_metrics(result['metrics']['val'],domain);page+=f'<tr><td>{domain}</td><td>{name}</td><td>{m["errors"]}</td><td>{m["ng_not_detected"]}</td><td>{m["false_ng"]}</td><td>{m["no_disk_as_ok"]}</td></tr>'
    atomic_text(root/'comparison.html',page+'</table></html>')
    def cards(predicate,title,note):
        output=[]
        for r,old,new in zip(rows,parent['predictions'],candidate['predictions']):
            if predicate(r,new):
                output.append(f'<article><a href="input_snapshot/{html.escape(r["image"])}"><img src="input_snapshot/{html.escape(r["image"])}"></a><p>{html.escape(r["id"])}</p><p>Nhãn {r["label"]} · trước {old["prediction"]["label"]} → sau {new["prediction"]["label"]} ({new["prediction"]["confidence"]:.2%})</p></article>')
        return dashboard().split('<h1>')[0]+'<h1>'+title+'</h1><p>'+note+'</p><div class="grid">'+''.join(output)+'</div>'
    atomic_text(root/'val_best_errors.html',cards(lambda r,p:r['split']=='val' and r['label']!=p['prediction']['label'],'Lỗi validation sau bổ sung AI','Validation gốc giữ nguyên; dữ liệu phát triển, không phải test độc lập.'))
    atomic_text(root/'diagnostic_7_cases.html',cards(lambda r,pred:r['id'] in p['diagnostic_val_ids'],'7 NG cũ của fine-tune','Cả 7 ảnh vẫn nằm nguyên trong validation.'))
    atomic_text(root/'synthetic_train.html',cards(lambda r,p:bool(r.get('is_synthetic')),'10 ảnh AI trong train','Đây là chẩn đoán dữ liệu đã học; không dùng làm bằng chứng tổng quát hóa.'))
    print('LEARNED',shared.totals(learned['metrics']['val']),'SELECTED',shared.totals(candidate['metrics']['val']),'PARENT_SELECTED',selected_parent,flush=True)

def replay(root):
    check(root);sys.path.append(str(PROJECT/'SERVICE/.venv/Lib/site-packages'))
    import decouple
    from unittest.mock import patch
    runtime=Path(read(PROJECT/'.work/factory_audit_20261020_run/report.json')['runtime'])
    decouple.config=decouple.Config(decouple.RepositoryEnv(str(runtime/'config/config.env')));os.chdir(runtime)
    from src.communication.modbus_client import MbClient
    def forbid(*a,**k):raise RuntimeError('Hardware forbidden')
    with patch.object(MbClient,'connect',forbid),patch.object(MbClient,'_MbClient__write_bit',forbid):
        from src.service.check_disk_service_yolo import DiskCheckingService
        original=DiskCheckingService.check_disk_white;times=[]
        def timed(service,*a,**k):
            start=time.perf_counter();result=original(service,*a,**k);times.append((time.perf_counter()-start)*1000);return result
        with patch.object(DiskCheckingService,'check_disk_white',timed):shared.replay(root)
    assert len(times)==56
    old=np.array(times[0::2]);new=np.array(times[1::2]);ratios=new/old
    latency=dict(frames=28,baseline_median_ms=float(np.median(old)),candidate_median_ms=float(np.median(new)),
        baseline_p95_ms=float(np.percentile(old,95)),candidate_p95_ms=float(np.percentile(new,95)),
        median_paired_ratio=float(np.median(ratios)),max_paired_ratio=float(ratios.max()),within_2x=bool((ratios<=2).all()),
        scope='Whole source White software call; same normalized frames, baseline then candidate. Excludes camera, PLC, network, file load.',hardware_calls=False)
    atomic_json(root/'artifacts/whole_white_latency.json',latency)
    release=read(root/'release_decision.json')
    if not latency['within_2x']:release['recommendation_failures'].append('whole_white/latency_over_2x')
    release['recommended_for_review']=not release['recommendation_failures'];release['whole_white_latency']=latency
    atomic_json(root/'release_decision.json',release)

def finalize(root):
    p=check(root);rows=frozen_rows(root);base.validate_rows(rows,base.BALANCED_POLICY)
    assert all(digest(r['path'])==r['sha256'] for r in rows)
    assert digest(Path(p['pool'])/'manifest.json')==p['pool_manifest_sha256']
    model=YOLO(str(root/'artifacts/best.pt')).model.float().cpu().eval()
    classifier=YoloClassifier(str(root/'artifacts/best.onnx'),base.LABELS)
    torch.set_num_threads(4);maximum=0.;wrong=[];val=[r for r in rows if r['split']=='val']
    with torch.inference_mode():
        for start in range(0,len(val),32):
            part=val[start:start+32];batch=classifier._preprocess_batch([cv2.imread(r['path']) for r in part],(224,224))
            a=model(torch.from_numpy(batch));a=(a[0] if isinstance(a,(tuple,list)) else a).numpy()
            b=classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:batch})[0]
            maximum=max(maximum,float(np.max(np.abs(a-b))));wrong.extend(r['id'] for r,x,y in zip(part,a.argmax(1),b.argmax(1)) if x!=y)
    parity=dict(count=len(val),argmax_mismatches=len(wrong),max_probability_error=maximum,passed=not wrong and maximum<1e-3)
    atomic_json(root/'artifacts/full_validation_export_parity.json',parity);assert parity['passed']
    integrity=base.audit(Path(p['pool']));atomic_json(root/'integrity_after.json',integrity);assert integrity['passed'];check(root)
    white=read(root/'white_replay.json');assert len(white['records'])==28 and not white['hardware_calls']
    release=read(root/'release_decision.json');completion=read(root/'artifacts/completion.json');latency=read(root/'artifacts/whole_white_latency.json')
    verification=dict(passed=True,verified_at=base.now(),canonical_revision=11,canonical_manifest_unchanged_since_freeze=True,
        frozen_hashes_verified=True,runtime_hashes_unchanged=True,canonical_integrity_passed=True,
        validation_original_1605_unchanged=True,synthetic_train_count=10,full_validation_export_parity=parity,
        white_count=28,hardware_calls=False,whole_white_latency=latency,
        selected_is_unchanged_parent=release['selected_is_unchanged_parent'],recommended_for_review=release['recommended_for_review'],production_deployed=False)
    atomic_json(root/'completion_verification.json',verification)
    lines=['# Fine-tune nano with ten accepted AI same-slot crops','',f"Completed {completion['epochs_completed']} epochs; learned best {completion['learned_best_checkpoint']['epoch']}. Parent selected: {release['selected_is_unchanged_parent']}. No deployment.",'','| Model | Errors | Missed NG | False NG | Empty as OK |','| --- | ---: | ---: | ---: | ---: |']
    for name,value in [('Installed HA',release['baseline']),('Retained fine-tune',release['fine_tune']),('n-safety',release['n_safety']),('AI learned best',release['learned']),('Selected',release['candidate'])]:
        m=shared.totals(value['val']);lines.append(f"| {name} | {m['errors']} | {m['ng_not_detected']} | {m['false_ng']} | {m['no_disk_as_ok']} |")
    lines+=['','Gates: '+(', '.join(release['recommendation_failures']) or 'Passed'),'',f"Whole White median ratio {latency['median_paired_ratio']:.3f}x, max {latency['max_paired_ratio']:.3f}x.",'','Ten accepted AI NG crops added only to train with source target groups. Original 5,383 records and fixed 1,605 validation preserved. Partial review remains. Validation is reused development data, HA is retention. No independent-test or production-zero-miss claim. Full validation parity and 28 hardware-off White verified; runtime and saved parent unchanged.']
    atomic_text(root/'RESULTS.md','\n'.join(lines)+'\n')
    atomic_text(root/'index.html',(root/'comparison.html').read_text(encoding='utf-8'))
    base.publish(root,'complete','AI supplementary training/evaluation/parity/White complete; no deployment')

def campaign(root,resume=False):
    cmd="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,creationflags=FLAGS,check=True).stdout.strip()
    processes=json.loads(raw) if raw else [];processes=processes if isinstance(processes,list) else [processes]
    if any(x['ProcessId']!=os.getpid() and 'train_classifier_' in (x.get('CommandLine') or '') and '--phase' in x['CommandLine'] for x in processes):raise RuntimeError('Another classifier trainer is alive')
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
    if resume and read(root/'status.json')['status']=='complete':raise ValueError('Never resume completed run')
    p=check(root) if resume else prepare(root);script=root/'code_snapshot'/Path(__file__).name
    for phase,done in [('fit','artifacts/completion.json'),('evaluate','release_decision.json'),('replay','white_replay.json'),('finalize','completion_verification.json')]:
        if not (root/done).exists():base.checked(root,phase,[sys.executable,'-X','utf8','-u','-B',str(script),'--phase',phase,'--root',str(root)],root/'logs'/f'{"train" if phase=="fit" else phase}.log')
    # checked() publishes a final running/phase snapshot when the child exits;
    # the campaign must publish completion after all child phases have returned.
    if not read(root/'completion_verification.json')['passed']:raise ValueError('Completion verification failed')
    base.publish(root,'complete','AI supplementary training/evaluation/parity/White complete; no deployment')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=['campaign','fit','evaluate','replay','finalize'],required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.phase=='campaign':campaign(root,args.resume)
        else:globals()[args.phase](root)
    except Exception as exc:
        if root.exists():base.publish(root,'stopped' if isinstance(exc,InterruptedError) else 'failed',args.phase,error=repr(exc))
        traceback.print_exc();raise

if __name__=='__main__':main()
