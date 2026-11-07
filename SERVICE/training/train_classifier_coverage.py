"""Fresh ImageNet nano fit on canonical region/visual coverage split, no deployment."""
import argparse
from collections import Counter,defaultdict
import html,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0,str(PROJECT/'SERVICE'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2,numpy as np,torch
from torch.utils.data import Sampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader,seed_worker
import train_classifier_canonical_holdout as base
import train_classifier_size_s as shared
import train_classifier_same_slot_ai as supplementary
from classifier_dataset import CanonicalDataset,active_dataset,read,atomic_json,atomic_text
from train_classifier_supplementary import digest
from src.inference.yolo_classify.yolo_classifier import YoloClassifier

FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
USER_STATEMENT='không ổn lắm nhỉ, thử phân bổ lại tập dữ liệu train/val rồi train lại từ đầu được không, làm sao để đảm bảo phân phối của tập train và val bao phủ nhau, model học được hết'
EVIDENCE=PROJECT/'.work/classifier_coverage_split_20261030'

def coverage_weights(rows,visual):
    """Balanced extras: source/camera/class -> real region -> visual -> target group."""
    if any(r['split']!='train' for r in rows):raise ValueError('Validation entered sampler')
    baseline=base.sampling_weights(rows,base.BALANCED_POLICY)
    bucket=lambda r:(r['origin'],r.get('camera'),r['label'])
    budgets=defaultdict(float);counts=Counter();regions=defaultdict(set);clusters=defaultdict(set);groups=defaultdict(set)
    for r,w in zip(rows,baseline):
        b=bucket(r);budgets[b]+=w;synthetic=bool(r.get('is_synthetic'));counts[b,synthetic]+=1
        if not synthetic:
            region=r.get('region') or 'unknown';cluster=visual[r['id']];g=r['split_group']
            regions[b].add(region);clusters[b,region].add(cluster);groups[b,region,cluster].add(g)
            counts[b,region,cluster,g]+=1
    weights=[]
    for r in rows:
        b=bucket(r);synthetic=bool(r.get('is_synthetic'));share=.15 if counts[b,True] else 0.
        if synthetic:
            if r['label']!='ng':raise ValueError('Unexpected synthetic class')
            weight=budgets[b]*share/counts[b,True]
        else:
            region=r.get('region') or 'unknown';cluster=visual[r['id']];g=r['split_group']
            weight=budgets[b]*(1-share)/len(regions[b])/len(clusters[b,region])/len(groups[b,region,cluster])/counts[b,region,cluster,g]
        weights.append(weight)
    if not all(w>0 for w in weights) or abs(sum(weights)-1)>1e-8:raise ValueError('Invalid coverage weights')
    return weights

class CompleteCoverageSampler(Sampler):
    """Every train image once plus N balanced draws, shuffled into one epoch."""
    def __init__(self,weights,seed):
        self.weights=torch.as_tensor(weights,dtype=torch.double);self.generator=torch.Generator().manual_seed(seed)
        if len(self.weights)==0 or (self.weights<=0).any():raise ValueError('Nonpositive sampler weights')
    def __len__(self):return 2*len(self.weights)
    def __iter__(self):
        n=len(self.weights)
        indices=torch.cat([torch.arange(n),torch.multinomial(self.weights,n,replacement=True,generator=self.generator)])
        return iter(indices[torch.randperm(2*n,generator=self.generator)].tolist())

class CoverageTrainer(shared.SizeSafetyTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        if rank!=-1:raise ValueError('Single GPU only')
        dataset=self.build_dataset(dataset_path,mode);rows=[dataset.record_by_path[str(Path(s[0]).resolve())] for s in dataset.samples]
        p=read(Path(self.args.data).parent/'protocol.json');visual=read(Path(self.args.data).parent/'visual_clusters.json')['labels_by_id']
        weights=coverage_weights(rows,visual);n=len(rows)
        atomic_json(self.save_dir/'sampling.json',dict(training_count=n,epoch_draws=2*n,all_train_images_once_per_epoch=True,
            balanced_extras=n,extras_synthetic_probability=sum(w for r,w in zip(rows,weights) if r.get('is_synthetic')),
            expected_full_epoch_synthetic_probability=.5*sum(w for r,w in zip(rows,weights) if r.get('is_synthetic'))+.5*sum(bool(r.get('is_synthetic')) for r in rows)/n,
            extras_domain_probabilities={o:sum(w for r,w in zip(rows,weights) if r['origin']==o) for o in base.BALANCED_WEIGHTS},
            effective_epoch_domain_probabilities={o:.5*sum(w for r,w in zip(rows,weights) if r['origin']==o)+.5*sum(r['origin']==o for r in rows)/n for o in base.BALANCED_WEIGHTS},
            region_visual_counts_from_train_only=True,validation_in_sampler=False))
        generator=torch.Generator().manual_seed(self.args.seed)
        return InfiniteDataLoader(dataset,batch_size=min(batch_size,n),sampler=CompleteCoverageSampler(weights,self.args.seed),
            num_workers=self.args.workers,pin_memory=True,worker_init_fn=seed_worker,generator=generator)

def dashboard(p):
    return f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fresh nano · split phủ vùng/cụm</title><style>body{{font:16px system-ui;background:#14202e;color:#e6eef6;margin:28px;max-width:1500px}}a{{color:#9bd4ff}}td,th{{border:1px solid #63778d;padding:9px}}table{{border-collapse:collapse}}p{{line-height:1.6}}.grid{{display:flex;flex-wrap:wrap;gap:14px}}article{{width:270px;padding:14px;background:#26394b;overflow-wrap:anywhere}}img{{width:240px;height:210px;object-fit:contain}}</style><h1>Train mới YOLO11n-cls · split bao phủ vùng và cụm ảnh</h1><p>Canonical revision12 · {p['train_count']}train / {p['val_count']}val · ImageNet1000 lớp → head3 lớp mới · toàn bộ weights/BN được học · lr3e-4 · tối đa300epoch/patience100.</p><p>Mỗi ảnh train xuất hiện ít nhất1lần/epoch, cộng thêm một lượt lấy mẫu cân bằng nguồn/camera/lớp/vùng/cụm/nhóm vị trí. 51/51 tổ hợp vùng/lớp và131/131 cụm có cả hai split; không rò nhóm vị trí/ảnh trùng. 10AI và gốc giữ train. Cụm thiết kế từ ImageNet cố định, không dùng dự đoán/lỗi classifier cũ. Không bảo đảm học hết kiểu lỗi.</p><p>Đây là validation mới theo vị trí cùng phiên chụp; có chung frame/ngữ cảnh. Không so trực tiếp con số cũ khác cohort. Model cũ được đánh giá lại trên cùng val mới nhưng có thể đã học một phần dữ liệu này. Rà nhãn vẫn partial; không triển khai.</p><p id="status">Đang tải…</p><p id="error"></p><p><a href="coverage.html">Phân bố trước/sau</a> · <a href="protocol.json">Protocol</a> · <a href="logs/train.log">Log</a> · <a href="comparison.html">So sánh trên val mới</a> · <a href="val_best_errors.html">Lỗi val mới</a> · <a href="old_scope_diagnostic.html">Val cũ: chỉ chẩn đoán</a> · <a href="RESULTS.md">Kết quả</a></p><script>async function refresh(){{try{{const s=await(await fetch('status.json?t='+Date.now(),{{cache:'no-store'}})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · epoch '+s.epoch+'/300 · best '+s.best_epoch;document.getElementById('error').textContent=s.error||''}}catch(e){{document.getElementById('error').textContent=e.message}}}}refresh();setInterval(refresh,10000)</script></html>'''

def check(root):return shared.check(root)
def rows(root):return supplementary.frozen_rows(root)

def prepare(root):
    pool=Path(read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json')['path']);m=read(pool/'manifest.json')
    adoption=read(EVIDENCE/'adoption_verification.json');assert m['canonical_revision']==adoption['canonical_revision']==12
    assert digest(pool/'manifest.json')==adoption['manifest_sha256']
    receipt=dict(authorized_at=base.now(),canonical_revision=12,incomplete_review_acknowledged=True,user_statement=USER_STATEMENT,
        scope='Repartition canonical by source/camera/region/class and frozen ImageNet visual coverage with original target/alias groups, 70/30; one fresh official ImageNet YOLO11n-cls fit, not prior fine-tune. Max300/patience100, train all weights/BN lr3e-4, NG CE2/1/1; all train rows once plus balanced extra draws each epoch. No deployment; full new-val parity and 28 hardware-off White, <=2x whole-software latency.')
    CanonicalDataset(pool).authorize_training(receipt,12);active_dataset(required_split_policy=base.BALANCED_POLICY)
    protected=read(PROJECT/'.work/classifier_same_slot_ai_20261029/protected_runtime_hashes.json')
    assert all(digest(f)==h for f,h in protected.items())
    p=base.prepare(root,base.BALANCED_POLICY,'imagenet',pretrained_override=PROJECT/'.work/pretrained_ultralytics/yolo11n-cls.pt',protected_runtime=protected)
    for name in ['visual_clusters.json','coverage_before.json','coverage_after.json','summary.json','feature_provenance.json']:
        shutil.copy2(EVIDENCE/name,root/name)
    data_rows=rows(root);(root/'evaluations').mkdir(exist_ok=True)
    sources={'baseline':root/'reference/baseline.onnx','fine_tune':PROJECT/'SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.onnx',
        'n_safety':PROJECT/'.work/classifier_safety_campaign_20261022/artifacts/best.onnx','previous_ai':PROJECT/'.work/classifier_same_slot_ai_20261029/artifacts/best.onnx'}
    references={}
    for name,source in sources.items():
        target=root/'reference'/f'{name}.onnx'
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
        result=base.evaluate_model(target,data_rows,root/'evaluations'/f'{name}.json')
        references[name]=dict(path=str(target),sha256=digest(target),metrics=result['metrics'])
    old_manifest=read(PROJECT/'.work/classifier_same_slot_ai_20261029/source_manifest_at_training.json')
    old_val_ids=[r['id'] for r in old_manifest['records'] if r['split']=='val']
    train_ids={r['id'] for r in data_rows if r['split']=='train'}
    p.update(model_name='yolo11n-cls',architecture='yolo11n-cls.yaml',initialization='imagenet',initial_classes=1000,freeze_bn=False,lr0=.0003,
        full_fits_max=1,baseline_validation=references['baseline']['metrics']['val'],comparison_models=references,
        checkpoint_selection='Unchanged NG-priority fitness: NG regression vs installed baseline per source/camera, NG misses, false NG, empty-as-OK, errors.',
        fitness_formula='-1e10*NG_excess -1e7*NG_misses -1000*false_NG -no_disk_as_OK -.001*errors',
        loss='CE NG/no_disk/OK 2/1/1; no KD, no head reuse from any inspection classifier.',
        sampling='Each train image once + N balanced draws by source/camera/class then real region/visual/target group; AI15% within three present NG buckets of extra draws only. Counts computed only from train.',
        visual_cluster_sha256=digest(root/'visual_clusters.json'),split_coverage_summary=read(root/'summary.json'),
        old_val_diagnostic_ids=old_val_ids,old_val_now_in_train=sum(i in train_ids for i in old_val_ids),
        old_scope_is_not_holdout=True,comparison_exposure_note='All models evaluated on same new canonical val. Old inspection models may have seen new val rows before; historical run metrics stay unchanged.',
        recommendation_policy='Existing installed-baseline per-source/camera gates plus zero NG misses and false NG <=retained fine-tune per source/camera on same new val; <=2x software White. No deployment.',
        authorization_receipt=receipt)
    fm=read(root/'input_snapshot/manifest.json');fm['baseline_validation']=p['baseline_validation'];atomic_json(root/'input_snapshot/manifest.json',fm)
    p['input_manifest_sha256']=digest(root/'input_snapshot/manifest.json')
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'));atomic_json(root/'authorization_receipt.json',receipt)
    atomic_text(root/'index.html',dashboard(p))
    before=read(root/'coverage_before.json');after=read(root/'coverage_after.json')
    page=dashboard(p).split('<p id="status">')[0]+'<h2>Coverage: ảnh thật, không tính AI</h2><table><tr><th>Nhóm</th><th>Train cũ</th><th>Val cũ</th><th>Train mới</th><th>Val mới</th></tr>'
    for key,value in after.items():
        if key.startswith('region/'):
            b=before[key];page+=f'<tr><td>{html.escape(key)}</td><td>{b["train"]}</td><td>{b["val"]}</td><td>{value["train"]}</td><td>{value["val"]}</td></tr>'
    atomic_text(root/'coverage.html',page+'</table><p>Solver tìm được nghiệm đáp ứng đủ ràng buộc coverage; hết90s, không chứng minh tối ưu toàn cục. Mọi nhóm ảnh/đĩa cùng vị trí giữ một phía.</p></html>')
    atomic_json(root/'previous_active_training_pointer.json',read(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'))
    atomic_json(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json',dict(path=str(root),report_url='http://127.0.0.1:8803/',source_dataset=str(pool),canonical_revision=12,status='prepared',phase='fresh nano coverage split frozen',entrypoint='SERVICE/training/train_classifier_coverage.py',campaign_pid=os.getpid(),active_pid=None,authorization=receipt,production_deployed=False))
    atomic_json(root/'start_verification.json',dict(canonical_revision=12,train=p['train_count'],val=p['val_count'],pretrained=p['pretrained_sha256'],fresh_imagenet_1000_class_parent=True,region_cells_both=51,visual_clusters_both=131,group_leakage=False,old_val_now_train=p['old_val_now_in_train'],no_deployment=True))
    return p

def fit(root):
    p=check(root);assert digest(root/'visual_clusters.json')==p['visual_cluster_sha256'];base.train(root,trainer_class=CoverageTrainer)

def evaluate(root):
    p=check(root);base.evaluate(root)
    results={name:read(root/'evaluations'/f'{name}.json') for name in ['baseline','fine_tune','n_safety','previous_ai','candidate']}
    candidate=results['candidate']['metrics']['val'];parent=results['fine_tune']['metrics']['val']
    failures=base.comparison_failures(candidate,results['baseline']['metrics']['val'],p['fitness_weights'],p['comparison_domains'])
    if shared.totals(candidate)['ng_not_detected']!=0:failures.append('all/zero_ng_misses_required')
    for domain in p['comparison_domains']:
        if candidate[domain]['false_ng']>parent[domain]['false_ng']:failures.append(domain+'/false_ng_above_fine_tune')
    release=read(root/'release_decision.json');release.update(recommendation_failures=failures,recommended_for_review=not failures,comparison_models={name:r['metrics'] for name,r in results.items()})
    atomic_json(root/'release_decision.json',release)
    page=dashboard(p).replace('</html>','')+'<h2>Cùng validation mới</h2><p>Gate: '+html.escape(', '.join(failures) or 'Đạt')+'</p><table><tr><th>Nguồn</th><th>Model</th><th>Lỗi</th><th>NG lọt</th><th>NG ảo</th><th>Trống→OK</th></tr>'
    for domain in ['all']+p['comparison_domains']:
        for name,result in results.items():
            q=supplementary.report_metrics(result['metrics']['val'],domain);page+=f'<tr><td>{domain}</td><td>{name}</td><td>{q["errors"]}</td><td>{q["ng_not_detected"]}</td><td>{q["false_ng"]}</td><td>{q["no_disk_as_ok"]}</td></tr>'
    atomic_text(root/'comparison.html',page+'</table></html>')
    metadata={r['id']:r for r in rows(root)};predictions=results['candidate']['predictions']
    def cards(items,title,note):
        content=[]
        for pred in items:
            r=metadata[pred['id']];src='input_snapshot/'+r['image'];content.append(f'<article><a href="{html.escape(src)}"><img src="{html.escape(src)}"></a><p>{html.escape(r["id"])}</p><p>split mới: {r["split"]} · nhãn {r["label"]} → {pred["prediction"]["label"]} ({pred["prediction"]["confidence"]:.2%})</p></article>')
        return dashboard(p).split('<h1>')[0]+'<h1>'+title+'</h1><p>'+note+'</p><div class="grid">'+''.join(content)+'</div>'
    atomic_text(root/'val_best_errors.html',cards([r for r in predictions if r['split']=='val' and r['truth']!=r['prediction']['label']],'Lỗi validation mới','Chỉ new-val; vẫn là spatial same-session development validation.'))
    old_scope=set(p['old_val_diagnostic_ids']);old=[r for r in predictions if r['id'] in old_scope]
    atomic_text(root/'old_scope_diagnostic.html',cards([r for r in old if r['truth']!=r['prediction']['label']],'Val cũ: chỉ chẩn đoán, có ảnh nay ở train','Một phần đã chuyển vào train do chia lại. Không dùng tổng số này làm holdout hay chọn best.'))
    old_by_id={r['id']:r for r in predictions}
    atomic_json(root/'old_scope_diagnostic.json',dict(scope='Old revision11 val; mixed train/val in revision12, not held out.',count=len(old),now_train=sum(r['split']=='train' for r in old),metrics=base.scoped_metrics([metadata[r['id']] for r in old],[r['prediction'] for r in old])))
    print('NEW_VAL',shared.totals(candidate),'GATES',failures,flush=True)

def replay(root):supplementary.replay(root)

def finalize(root):
    p=check(root);data_rows=rows(root);base.validate_rows(data_rows,base.BALANCED_POLICY)
    assert all(digest(r['path'])==r['sha256'] for r in data_rows)
    assert digest(Path(p['pool'])/'manifest.json')==p['pool_manifest_sha256']
    assert digest(root/'visual_clusters.json')==p['visual_cluster_sha256']
    model=YOLO(str(root/'artifacts/best.pt')).model.float().cpu().eval();classifier=YoloClassifier(str(root/'artifacts/best.onnx'),base.LABELS)
    torch.set_num_threads(4);maximum=0.;wrong=[];val=[r for r in data_rows if r['split']=='val']
    with torch.inference_mode():
        for start in range(0,len(val),32):
            part=val[start:start+32];batch=classifier._preprocess_batch([cv2.imread(r['path']) for r in part],(224,224));a=model(torch.from_numpy(batch));a=(a[0] if isinstance(a,(tuple,list)) else a).numpy();b=classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:batch})[0]
            maximum=max(maximum,float(np.max(np.abs(a-b))));wrong.extend(r['id'] for r,x,y in zip(part,a.argmax(1),b.argmax(1)) if x!=y)
    parity=dict(count=len(val),argmax_mismatches=len(wrong),max_probability_error=maximum,passed=not wrong and maximum<1e-3)
    atomic_json(root/'artifacts/full_validation_export_parity.json',parity);assert parity['passed']
    integrity=base.audit(Path(p['pool']));atomic_json(root/'integrity_after.json',integrity);assert integrity['passed'];check(root)
    white=read(root/'white_replay.json');assert len(white['records'])==28 and not white['hardware_calls']
    release=read(root/'release_decision.json');completion=read(root/'artifacts/completion.json');latency=read(root/'artifacts/whole_white_latency.json')
    atomic_json(root/'completion_verification.json',dict(passed=True,verified_at=base.now(),canonical_revision=12,full_validation_export_parity=parity,
        canonical_manifest_unchanged=True,frozen_inputs_verified=True,runtime_hashes_unchanged=True,split_group_leakage=False,
        white_count=28,hardware_calls=False,whole_white_latency=latency,old_val_is_diagnostic_only=True,recommended_for_review=release['recommended_for_review'],production_deployed=False))
    lines=['# Fresh nano on coverage split revision12','',f"Completed {completion['epochs_completed']} epochs; best saved {completion['best_checkpoint']['epoch']}. No deployment.",'','| Model, same NEW val | Errors | Missed NG | False NG | Empty as OK |','| --- | ---: | ---: | ---: | ---: |']
    for name,metrics in release['comparison_models'].items():
        q=shared.totals(metrics['val']);lines.append(f"| {name} | {q['errors']} | {q['ng_not_detected']} | {q['false_ng']} | {q['no_disk_as_ok']} |")
    lines+=['','Gates: '+(', '.join(release['recommendation_failures']) or 'Passed'),'',f"Whole software White median ratio{latency['median_paired_ratio']:.3f}x, max{latency['max_paired_ratio']:.3f}x.",'','Current canonical labels/hashes preserved; new split covers51region-class cells and131fixed visual clusters in both splits. All repeated targets/aliases remain one side. AI and real roots train only.','Validation cohort changed; never directly compare old run error counts. Old val is partly training now and is diagnostic only. Old-model comparisons can have prior exposure. Same-session spatial validation, shared frame context, partial label review; no independent generalization or production-zero-miss guarantee.','Full new-val PT/ONNX parity,28hardware-off White, canonical/input/code/protocol/runtime hashes verified. Installed baseline retained.']
    atomic_text(root/'RESULTS.md','\n'.join(lines)+'\n');atomic_text(root/'index.html',(root/'comparison.html').read_text(encoding='utf-8'))

def campaign(root,resume=False):
    cmd="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,creationflags=FLAGS,check=True).stdout.strip();processes=json.loads(raw) if raw else [];processes=processes if isinstance(processes,list) else [processes]
    if any(x['ProcessId']!=os.getpid() and 'train_classifier_' in (x.get('CommandLine') or '') and '--phase' in x['CommandLine'] for x in processes):raise RuntimeError('Another classifier trainer is alive')
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
    if resume and read(root/'status.json')['status']=='complete':raise ValueError('Do not resume completed run')
    p=check(root) if resume else prepare(root);script=root/'code_snapshot'/Path(__file__).name
    for phase,done in [('fit','artifacts/completion.json'),('evaluate','release_decision.json'),('replay','white_replay.json'),('finalize','completion_verification.json')]:
        if not (root/done).exists():base.checked(root,phase,[sys.executable,'-X','utf8','-u','-B',str(script),'--phase',phase,'--root',str(root)],root/'logs'/f'{"train" if phase=="fit" else phase}.log')
    if not read(root/'completion_verification.json')['passed']:raise ValueError('Completion verification failed')
    base.publish(root,'complete','Fresh nano coverage training/evaluation/White complete; no deployment')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=['campaign','fit','evaluate','replay','finalize'],required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args();root=args.root.resolve()
    try:
        if args.phase=='campaign':campaign(root,args.resume)
        else:globals()[args.phase](root)
    except Exception as exc:
        if root.exists():base.publish(root,'stopped' if isinstance(exc,InterruptedError) else 'failed',args.phase,error=repr(exc))
        traceback.print_exc();raise

if __name__=='__main__':main()
