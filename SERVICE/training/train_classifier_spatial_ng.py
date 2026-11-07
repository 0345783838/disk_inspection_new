"""One frozen-parent nano spatial NG correction fit; exact canonical split, no deployment."""
from pathlib import Path
import argparse,csv,html,json,os,shutil,subprocess,sys,time,traceback,hashlib
PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'SERVICE/training/classifier_dataset.py').is_file())
sys.path.insert(0,str(PROJECT/'SERVICE'));sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2,numpy as np,torch
import torch.nn.functional as F
import train_classifier_canonical_holdout as base
import train_classifier_size_s as shared
from train_classifier_safety_campaign import safety_fitness
from train_classifier_ha import runtime_tensor,safe_augment
from classifier_dataset import CanonicalDataset,active_dataset,read,atomic_json,atomic_text
from classifier_spatial_ng import from_parent,save_model,load_model,state_digest
from train_classifier_supplementary import digest
from evaluate_classifier_ha import YoloClassifier
cv2.setNumThreads(1)
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
PARENT=PROJECT/'SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.pt'
USER_STATEMENT='tạm thời giữ lại 1 model bản fine-tune đang khá tốt vào thư mục D:\\huynhvc\\OTHERS\\4.DISK_INSPECTION\\SERVICE\\config\\models. Sau đó thử cách mày nói xem sao'

def check(root):return shared.check(root)
def rows(root):return read(root/'input_snapshot/manifest.json')['records']
def progress(root,epoch,best,fitness):
    atomic_json(root/'training/fit/early_stopping_state.json',dict(epoch=epoch,best_epoch=best,best_fitness=fitness))
def metrics_for_rows(records,labels):return base.scoped_metrics(records,[dict(label=base.LABELS[int(i)]) for i in labels])
def dashboard():
    return '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nano spatial NG</title><style>body{font:16px system-ui;background:#14202e;color:#e6eef6;margin:28px;max-width:1400px}a{color:#9bd4ff}p{line-height:1.6}td,th{border:1px solid #63778d;padding:10px}table{border-collapse:collapse}</style><h1>Nano fine-tune + nhánh spatial NG</h1><p>Backbone/head cũ/BN khóa hoàn toàn; chỉ học 5.120 trọng số hiệu chỉnh logit NG theo lưới 2×2. Khởi tạo khớp fine-tune. Max300/patience100. Không triển khai.</p><p>Canonical revision10, 3.778 train / 1.605 val. 7 NG đã xem giữ nguyên val. Train features có 4 view ánh sáng/flip an toàn; val chỉ view gốc. Không đổi nhãn/split. Review còn partial; val đã dùng phát triển, không phải test độc lập.</p><p id="status">Đang tải…</p><p id="error"></p><p><a href="protocol.json">Protocol</a> · <a href="logs/fit.log">Log train</a> · <a href="comparison.html">So sánh</a> · <a href="diagnostic_7_cases.html">7 NG ban đầu</a> · <a href="val_best_errors.html">Lỗi val</a> · <a href="RESULTS.md">Kết quả</a> · <a href="completion_verification.json">Kiểm tra hoàn tất</a></p><script>async function refresh(){try{let s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('status').textContent=s.status+' / '+s.phase+' · epoch '+s.epoch+'/300 · best '+s.best_epoch;document.getElementById('error').textContent=s.error||''}catch(e){document.getElementById('error').textContent=e.message}}refresh();setInterval(refresh,10000)</script></html>'''

def prepare(root):
    if root.exists():raise FileExistsError(root)
    pool=Path(read(PROJECT/'.work/ACTIVE_CLASSIFIER_DATASET.json')['path']);before=read(pool/'manifest.json')
    receipt=dict(authorized_at=base.now(),canonical_revision=before['canonical_revision'],incomplete_review_acknowledged=True,user_statement=USER_STATEMENT,scope='Retain fine-tune files; one frozen-parent spatial NG residual fit only (5120 trainable weights), max300/patience100. Exact canonical labels/split and seven diagnostic NG remain validation. No deployment, no deeper fine-tuning or DINO. Full validation parity and hardware-off White replay; software White processing <=2x installed baseline.')
    CanonicalDataset(pool).authorize_training(receipt,before['canonical_revision']);active_dataset(required_split_policy=base.BALANCED_POLICY)
    assert shared.signature(before['records'])==shared.signature(read(pool/'manifest.json')['records'])
    oldpaths=read(PROJECT/'.work/classifier_size_s_20261029/protected_runtime_hashes.json')
    paths={Path(x) for x in oldpaths};paths.update(PARENT.parent.glob('c_p_finetune_ha_r10_best111_20261029.*'))
    protected={str(p):digest(p) for p in paths}
    p=base.prepare(root,base.BALANCED_POLICY,'retained',pretrained_override=PARENT,protected_runtime=protected)
    records=rows(root);oldroot=PROJECT/'.work/classifier_safety_campaign_20261022';assert shared.signature(records)==shared.signature(read(oldroot/'source_manifest_at_training.json')['records'])
    (root/'evaluations').mkdir();references={}
    for name,previous in [('baseline','baseline'),('n_safety','candidate'),('fine_tune','previous_finetune_a100')]:
        e=read(oldroot/'evaluations'/f'{previous}.json');by_id={r['id']:r for r in e['predictions']};assert set(by_id)=={r['id'] for r in records}
        assert all(by_id[r['id']]['truth']==r['label'] and by_id[r['id']]['split']==r['split'] for r in records)
        assert digest(e['model_path'])==e['model_sha256'];target=root/'reference'/f'{name}.onnx';shutil.copy2(e['model_path'],target)
        e.update(model_path=str(target),predictions=[by_id[r['id']] for r in records]);atomic_json(root/'evaluations'/f'{name}.json',e);references[name]=dict(path=str(target),sha256=digest(target))
    initial=from_parent(p['pretrained']);frozen=initial.freeze_state();diagnostic=[r['id'] for r in read(root/'evaluations/fine_tune.json')['predictions'] if r['split']=='val' and r['truth']=='ng' and r['prediction']['label']!='ng']
    assert len(diagnostic)==7 and not set(diagnostic)&{r['id'] for r in records if r['split']=='train'}
    p.update(model_name='nano_spatial_ng_residual',initialization='retained_fine_tune_frozen',initial_classes=3,freeze_bn=True,epochs=300,patience=100,seed=1022,head_batch=128,head_lr=.001,head_weight_decay=.0001,train_views=4,feature_dim=5120,std_floor=.05,aux_binary_weight=.5,aux_ng_positive_weight=2.,supervised_weights=[2.,1.,1.],parent_frozen_state_sha256=state_digest(frozen),parent_parameters=sum(v.numel() for v in initial.parameters())-5120,trainable_parameters=5120,comparison_models=references,baseline_validation=read(root/'evaluations/baseline.json')['metrics']['val'],diagnostic_val_ids=diagnostic,
        checkpoint_selection='Existing NG-first fitness; minimize source/camera NG regression, NG misses, false NG, empty-as-OK, errors. Epoch0 unchanged fine-tune is an eligible baseline.',
        loss='CE on combined logits weights2/1/1 +0.5*binary NG loss on correction branch pos_weight2. Auxiliary loss supplies gradients even where frozen parent is already confident/correct on train.',
        feature_policy='Train: original plus3 deterministic safe_augment views. Validation: original only. Standardization uses original train features only. Frozen parent parameters/BN/dropout do not change.',
        recommendation_policy='Keep original baseline gates; additionally require zero val NG misses, false NG no higher than fine-tune per source/camera and full White software processing <=2x. No deployment.',authorization_receipt=receipt)
    p['input_manifest_sha256']=digest(root/'input_snapshot/manifest.json')
    atomic_json(root/'protocol.json',p);atomic_text(root/'protocol.sha256',digest(root/'protocol.json'));atomic_json(root/'authorization_receipt.json',receipt);atomic_text(root/'index.html',dashboard())
    atomic_json(root/'previous_active_training_pointer.json',read(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json'))
    atomic_json(PROJECT/'.work/ACTIVE_CLASSIFIER_TRAINING.json',dict(path=str(root),report_url='http://127.0.0.1:8800/',source_dataset=str(pool),canonical_revision=p['canonical_revision'],status='prepared',phase='spatial inputs frozen',entrypoint='SERVICE/training/train_classifier_spatial_ng.py',campaign_pid=os.getpid(),active_pid=None,authorization=receipt,production_deployed=False))
    atomic_json(root/'start_verification.json',dict(canonical_integrity_passed=read(root/'integrity_before.json')['passed'],canonical_revision=p['canonical_revision'],train_count=p['train_count'],val_count=p['val_count'],labels_hashes_splits_preserved=True,diagnostic_seven_remain_val=True,trainable_parameters=5120,parent_parameters_frozen=True,focused_tests_passed=14,retained_parent=str(PARENT)))
    return p

def extract(root):
    p=check(root);records=rows(root);cache=root/'features';cache.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(p['seed'])
    model=from_parent(p['pretrained']).cuda().eval();frozen=model.freeze_state();assert state_digest(frozen)==p['parent_frozen_state_sha256']
    plan=[(i,v) for i,r in enumerate(records) for v in range(p['train_views'] if r['split']=='train' else 1)];n=len(plan)
    features=np.lib.format.open_memmap(cache/'spatial.npy',mode='w+',dtype='float32',shape=(n,p['feature_dim']));logits=np.lib.format.open_memmap(cache/'parent_logits.npy',mode='w+',dtype='float32',shape=(n,3));orig=np.full(len(records),-1,dtype=np.int64);positions=np.full((len(records),p['train_views']),-1,dtype=np.int64)
    with torch.inference_mode():
        for start in range(0,n,64):
            part=plan[start:start+64];images=[]
            for offset,(i,view) in enumerate(part):
                r=records[i];image=cv2.imread(str(root/'input_snapshot'/r['image']));assert image is not None
                if view:
                    np.random.seed(int(hashlib.sha256((r['id']+'|'+str(view)+'|1022').encode()).hexdigest()[:8],16));image=safe_augment(image)
                images.append(runtime_tensor(image));positions[i,view]=start+offset
                if not view:orig[i]=start+offset
            a,b=model.extract(torch.stack(images).cuda());features[start:start+len(part)]=a.cpu().numpy();logits[start:start+len(part)]=b.cpu().numpy()
            if start%1024==0:print('FEATURES',start+len(part),'/',n,flush=True)
    assert (orig>=0).all() and np.isfinite(features).all() and np.isfinite(logits).all();features.flush();logits.flush();model.assert_frozen(frozen)
    train_rows=np.array([i for i,r in enumerate(records) if r['split']=='train']);x=np.asarray(features[orig[train_rows]]);mean=x.mean(0,dtype=np.float64).astype(np.float32);std=x.std(0,dtype=np.float64).astype(np.float32);inv=1/np.maximum(std,p['std_floor'])
    np.savez(cache/'indices_stats.npz',orig=orig,positions=positions,train_rows=train_rows,mean=mean,inv_std=inv)
    cached=read(root/'evaluations/fine_tune.json')['predictions'];label=np.asarray(logits[orig]).argmax(1);assert all(base.LABELS[int(a)]==b['prediction']['label'] for a,b in zip(label,cached))
    atomic_json(cache/'complete.json',dict(count=n,original_count=len(records),train_original_count=len(train_rows),val_views=1,all_original_parent_argmax_match=True,parent_unchanged=True,files={name:digest(cache/name) for name in ['spatial.npy','parent_logits.npy','indices_stats.npz']}))
    print('FEATURES_COMPLETE',n,flush=True)

def cache_data(root,p):
    cache=root/'features';done=read(cache/'complete.json')
    if any(digest(cache/k)!=v for k,v in done['files'].items()):raise ValueError('Feature cache changed')
    stats=np.load(cache/'indices_stats.npz');x=np.load(cache/'spatial.npy',mmap_mode='r');old=np.load(cache/'parent_logits.npy',mmap_mode='r');return x,old,stats

def fit(root):
    p=check(root);records=rows(root);x,old,stats=cache_data(root,p);torch.set_num_threads(4);torch.manual_seed(p['seed']);rng=np.random.default_rng(p['seed'])
    model=from_parent(p['pretrained']).cuda();model.feature_mean.copy_(torch.from_numpy(stats['mean']).cuda());model.feature_inv_std.copy_(torch.from_numpy(stats['inv_std']).cuda());frozen=model.freeze_state()
    assert state_digest(frozen)==p['parent_frozen_state_sha256'];features=torch.tensor(np.asarray(x),device='cuda');parent_logits=torch.tensor(np.asarray(old),device='cuda');train_rows=stats['train_rows'];train_records=[records[int(i)] for i in train_rows];assert all(r['split']=='train' for r in train_records)
    weights=np.asarray(base.sampling_weights(train_records,base.BALANCED_POLICY));weights/=weights.sum();pos=stats['positions'];orig=stats['orig'];targets=torch.tensor([base.LABELS.index(records[i]['label']) for i,v in [(i,v) for i,r in enumerate(records) for v in range(p['train_views'] if r['split']=='train' else 1)]],device='cuda')
    val_rows=np.array([i for i,r in enumerate(records) if r['split']=='val']);val_records=[records[int(i)] for i in val_rows];val_idx=torch.tensor(orig[val_rows],device='cuda')
    optimizer=torch.optim.AdamW(model.ng_residual.parameters(),lr=p['head_lr'],weight_decay=p['head_weight_decay']);folder=root/'training/fit';folder.mkdir(parents=True,exist_ok=True);artifacts=root/'artifacts';artifacts.mkdir(exist_ok=True)
    def validate():
        model.eval()
        with torch.no_grad():z,_=model.logits_from_features(features[val_idx],parent_logits[val_idx]);pred=z.argmax(1).cpu().numpy()
        metrics=metrics_for_rows(val_records,pred);return safety_fitness(metrics,p['baseline_validation']),metrics
    best_fitness,initial_metrics=validate();best_epoch=0;start_epoch=1
    if not (folder/'last_training.pt').exists():
        save_model(model,artifacts/'best.pt',epoch=0,fitness=best_fitness);atomic_json(folder/'best_checkpoint_epoch.json',dict(epoch=0,fitness=best_fitness,metrics=initial_metrics))
    last=folder/'last_training.pt'
    if last.exists():
        assert (artifacts/'best.pt').exists() and (folder/'best_checkpoint_epoch.json').exists()
        state=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);best_fitness=state['best_fitness'];best_epoch=state['best_epoch'];start_epoch=state['epoch']+1;rng.bit_generator.state=state['rng']
    ce_weights=torch.tensor(p['supervised_weights'],device='cuda');positive=torch.tensor(p['aux_ng_positive_weight'],device='cuda');file=folder/'results.csv'
    epoch=start_epoch-1
    for epoch in range(start_epoch,p['epochs']+1):
        if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
        model.train();lr=p['head_lr']*(.05+.95*(1+np.cos(np.pi*(epoch-1)/p['epochs']))/2)
        for group in optimizer.param_groups:group['lr']=lr
        sampled=rng.choice(len(train_rows),size=len(train_rows),replace=True,p=weights);views=rng.integers(p['train_views'],size=len(train_rows));chosen=pos[train_rows[sampled],views];losses=[]
        for start in range(0,len(chosen),p['head_batch']):
            idx=torch.tensor(chosen[start:start+p['head_batch']],device='cuda');optimizer.zero_grad(set_to_none=True);combined,correction=model.logits_from_features(features[idx],parent_logits[idx]);target=targets[idx]
            loss=F.cross_entropy(combined,target,weight=ce_weights)+p['aux_binary_weight']*F.binary_cross_entropy_with_logits(correction[:,0],(target==0).float(),pos_weight=positive)
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.ng_residual.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        model.assert_frozen(frozen);fitness,metrics=validate();total=shared.totals(metrics)
        if fitness>best_fitness:
            best_fitness=fitness;best_epoch=epoch;save_model(model,artifacts/'best.pt',epoch=epoch,fitness=fitness);atomic_json(folder/'best_checkpoint_epoch.json',dict(epoch=epoch,fitness=fitness,metrics=metrics))
        flat=dict(epoch=epoch,loss=float(np.mean(losses)),fitness=fitness,best_epoch=best_epoch,**total)
        for d,q in metrics.items():
            for k in ['errors','ng_not_detected','false_ng','no_disk_as_ok']:flat[d+'_'+k]=q[k]
        new=not file.exists()
        with file.open('a',encoding='utf8',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(flat));
            if new:writer.writeheader()
            writer.writerow(flat)
        torch.save(dict(epoch=epoch,model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),best_fitness=best_fitness,best_epoch=best_epoch,rng=rng.bit_generator.state),last);progress(root,epoch,best_epoch,best_fitness)
        print('EPOCH',epoch,'BEST',best_epoch,'LOSS',round(flat['loss'],4),'VAL',total,flush=True)
        if epoch-best_epoch>=p['patience']:break
    model.assert_frozen(frozen);atomic_json(artifacts/'frozen_parent_verification.json',dict(passed=True,parent_state_sha256=state_digest(frozen),trainable_parameters=5120,bn_and_old_head_unchanged=True))
    atomic_json(artifacts/'completion.json',dict(epochs_completed=epoch,best_checkpoint=read(folder/'best_checkpoint_epoch.json'),epochs_limit=p['epochs'],patience=p['patience'],trainable_parameters=5120))

def evaluate(root):
    p=check(root);records=rows(root);x,old,stats=cache_data(root,p);model=load_model(root/'artifacts/best.pt').eval();torch.set_num_threads(4)
    model.assert_frozen(from_parent(p['pretrained']).freeze_state());model_path=root/'artifacts/best.onnx'
    example=torch.zeros(2,3,224,224);torch.onnx.export(model,example,model_path,input_names=['images'],output_names=['probabilities'],dynamic_axes={'images':{0:'batch'},'probabilities':{0:'batch'}},opset_version=13)
    import onnx
    exported=onnx.load(str(model_path));item=exported.metadata_props.add();item.key='names';item.value=str(dict(enumerate(base.LABELS)));onnx.checker.check_model(exported);onnx.save(exported,str(model_path))
    orig=stats['orig'];prob=[]
    with torch.no_grad():
        for start in range(0,len(orig),128):
            part=orig[start:start+128];z,_=model.logits_from_features(torch.from_numpy(np.array(x[part])),torch.from_numpy(np.array(old[part])));prob.extend(z.softmax(-1).numpy())
    predictions=[dict(label=base.LABELS[int(a.argmax())],confidence=float(a.max()),probabilities=dict(zip(base.LABELS,map(float,a)))) for a in prob]
    metrics={s:base.scoped_metrics([r for r in records if r['split']==s],[a for r,a in zip(records,predictions) if r['split']==s]) for s in ['train','val']}
    evaluation=dict(model_path=str(model_path),model_sha256=digest(model_path),metrics=metrics,predictions=[dict(id=r['id'],truth=r['label'],split=r['split'],origin=r['origin'],prediction=a) for r,a in zip(records,predictions)],inference_note='Cached frozen-parent features; full validation full-image PT/ONNX parity verified separately.')
    atomic_json(root/'evaluations/candidate.json',evaluation)
    classifier=YoloClassifier(str(model_path),base.LABELS,batch_size=32);maximum=0.;wrong=[];cache_wrong=[]
    with torch.no_grad():
        val_indices=[i for i,r in enumerate(records) if r['split']=='val']
        for start in range(0,len(val_indices),32):
            indices=val_indices[start:start+32];batch=classifier._preprocess_batch([cv2.imread(str(root/'input_snapshot'/records[i]['image'])) for i in indices],(224,224));a=model(torch.from_numpy(batch)).numpy();b=classifier.ort_sess.run(None,{classifier.ort_sess.get_inputs()[0].name:batch})[0];maximum=max(maximum,float(np.max(np.abs(a-b))))
            wrong.extend(records[i]['id'] for i,u,v in zip(indices,a.argmax(1),b.argmax(1)) if u!=v);cache_wrong.extend(records[i]['id'] for i,u in zip(indices,a.argmax(1)) if u!=int(np.argmax(prob[i])))
    parity=dict(count=len(val_indices),argmax_mismatches=len(wrong),cached_feature_argmax_mismatches=len(cache_wrong),max_probability_error=maximum,passed=not wrong and not cache_wrong and maximum<1e-3);atomic_json(root/'artifacts/full_validation_export_parity.json',parity);assert parity['passed']
    refs={k:read(root/'evaluations'/f'{k}.json') for k in ['baseline','fine_tune','n_safety']};failures=base.comparison_failures(metrics['val'],refs['baseline']['metrics']['val'],p['fitness_weights'],p['comparison_domains'])
    if shared.totals(metrics['val'])['ng_not_detected']!=0:failures.append('all/zero_ng_misses_required')
    for d in p['comparison_domains']:
        if metrics['val'][d]['false_ng']>refs['fine_tune']['metrics']['val'][d]['false_ng']:failures.append(d+'/false_ng_above_fine_tune')
    release=dict(training_complete=True,production_deployed=False,recommended_for_review=not failures,selected_pt=str(root/'artifacts/best.pt'),selected_onnx=str(model_path),pt_sha256=digest(root/'artifacts/best.pt'),onnx_sha256=digest(model_path),candidate=metrics,baseline=refs['baseline']['metrics'],fine_tune=refs['fine_tune']['metrics'],n_safety=refs['n_safety']['metrics'],recommendation_failures=failures,canonical_revision=p['canonical_revision'],installed_baseline_retained=True)
    atomic_json(root/'release_decision.json',release)
    page=dashboard().replace('</html>','')+'<h2>Validation</h2><table><tr><th>Nguồn</th><th>Model</th><th>Lỗi</th><th>NG lọt</th><th>NG ảo</th><th>Trống → OK</th></tr>'
    for d in p['comparison_domains']:
        for name,value in [('HA đang cài',refs['baseline']['metrics']),('Fine-tune',refs['fine_tune']['metrics']),('n-safety',refs['n_safety']['metrics']),('Spatial NG',metrics)]:
            q=value['val'][d];page+=f'<tr><td>{d}</td><td>{name}</td><td>{q["errors"]}</td><td>{q["ng_not_detected"]}</td><td>{q["false_ng"]}</td><td>{q["no_disk_as_ok"]}</td></tr>'
    atomic_text(root/'comparison.html',page+'</table><p>Gate: '+html.escape(', '.join(failures) or 'Không có')+'</p></html>')
    def cards(indices,title):
        output=[]
        for i in indices:
            r=records[i];a=predictions[i];output.append(f'<article><img src="input_snapshot/{html.escape(r["image"])}"><p>{html.escape(r["id"])}</p><p>Nhãn {r["label"]} → spatial {a["label"]} ({a["confidence"]:.2%})</p></article>')
        return '<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;background:#14202e;color:white}.grid{display:flex;flex-wrap:wrap;gap:16px}article{width:275px;overflow-wrap:anywhere}img{width:250px;height:210px;object-fit:contain}</style><h1>'+title+'</h1><p>Truth đóng băng revision10, development validation.</p><div class="grid">'+''.join(output)+'</div>'
    atomic_text(root/'val_best_errors.html',cards([i for i,r in enumerate(records) if r['split']=='val' and predictions[i]['label']!=r['label']],'Lỗi validation spatial NG'))
    atomic_text(root/'diagnostic_7_cases.html',cards([i for i,r in enumerate(records) if r['id'] in p['diagnostic_val_ids']],'7 NG fine-tune từng bỏ sót — kết quả spatial'))
    print('CANDIDATE_VAL',shared.totals(metrics['val']),'GATES',failures,flush=True)

def replay(root):
    check(root);sys.path.append(str(PROJECT/'SERVICE/.venv/Lib/site-packages'));import decouple
    from unittest.mock import patch
    audit=read(PROJECT/'.work/factory_audit_20261020_run/report.json');runtime=Path(audit['runtime']);decouple.config=decouple.Config(decouple.RepositoryEnv(str(runtime/'config/config.env')));os.chdir(runtime)
    from src.communication.modbus_client import MbClient
    def forbid(*a,**k):raise RuntimeError('Hardware forbidden')
    with patch.object(MbClient,'connect',forbid),patch.object(MbClient,'_MbClient__write_bit',forbid):
        from src.service.check_disk_service_yolo import DiskCheckingService
        original=DiskCheckingService.check_disk_white;times=[]
        def timed(service,*a,**k):
            start=time.perf_counter();result=original(service,*a,**k);times.append((time.perf_counter()-start)*1000);return result
        with patch.object(DiskCheckingService,'check_disk_white',timed):shared.replay(root)
    assert len(times)==56;old=np.array(times[0::2]);new=np.array(times[1::2]);ratios=new/old
    latency=dict(frames=28,baseline_median_ms=float(np.median(old)),candidate_median_ms=float(np.median(new)),baseline_p95_ms=float(np.percentile(old,95)),candidate_p95_ms=float(np.percentile(new,95)),median_paired_ratio=float(np.median(ratios)),max_paired_ratio=float(ratios.max()),within_2x=bool((ratios<=2).all()),scope='Whole current source White software call, same normalized frame; excludes camera/PLC/network/file load. Baseline then candidate per frame.',hardware_calls=False)
    atomic_json(root/'artifacts/whole_white_latency.json',latency);release=read(root/'release_decision.json')
    if not latency['within_2x']:release['recommendation_failures'].append('whole_white/latency_over_2x')
    release['recommended_for_review']=not release['recommendation_failures'];release['whole_white_latency']=latency;atomic_json(root/'release_decision.json',release)

def finalize(root):
    p=check(root);cache_data(root,p);records=rows(root);base.validate_rows(records,base.BALANCED_POLICY)
    if any(digest(root/'input_snapshot'/r['image'])!=r['sha256'] for r in records):raise ValueError('Frozen image changed')
    parity=read(root/'artifacts/full_validation_export_parity.json');white=read(root/'white_replay.json');assert parity['passed'] and len(white['records'])==28 and not white['hardware_calls']
    model=load_model(root/'artifacts/best.pt');parent=from_parent(p['pretrained']);model.assert_frozen(parent.freeze_state());assert state_digest(model.freeze_state())==p['parent_frozen_state_sha256']
    integrity=base.audit(Path(p['pool']));atomic_json(root/'integrity_after.json',integrity);assert integrity['passed'];check(root)
    release=read(root/'release_decision.json');completion=read(root/'artifacts/completion.json');latency=read(root/'artifacts/whole_white_latency.json')
    atomic_json(root/'completion_verification.json',dict(passed=True,verified_at=base.now(),canonical_revision_at_start=p['canonical_revision'],current_canonical_revision=read(Path(p['pool'])/'manifest.json')['canonical_revision'],frozen_inputs_used=True,frozen_hashes_verified=True,parent_all_weights_bn_head_unchanged=True,runtime_hashes_unchanged=True,canonical_integrity_passed=True,white_count=28,hardware_calls=False,full_validation_export_parity=parity,whole_white_latency=latency,recommended_for_review=release['recommended_for_review'],production_deployed=False))
    lines=['# Nano spatial NG residual','',f"Hoàn tất {completion['epochs_completed']} epoch, best {completion['best_checkpoint']['epoch']}; chỉ học 5.120 tham số. Qua gate: {release['recommended_for_review']}. Không triển khai.",'','| Model | Lỗi | NG lọt | NG ảo | Trống → OK |','| --- | ---: | ---: | ---: | ---: |']
    for name,value in [('HA đang cài',release['baseline']),('Fine-tune giữ lại',release['fine_tune']),('n-safety',release['n_safety']),('Spatial NG',release['candidate'])]:
        q=shared.totals(value['val']);lines.append(f"| {name} | {q['errors']} | {q['ng_not_detected']} | {q['false_ng']} | {q['no_disk_as_ok']} |")
    lines+=['','Gate chưa đạt: '+(', '.join(release['recommendation_failures']) or 'Không có'),'',f"White backend median {latency['baseline_median_ms']:.2f} → {latency['candidate_median_ms']:.2f} ms; median ratio {latency['median_paired_ratio']:.3f}x, max {latency['max_paired_ratio']:.3f}x.",'','Parent backbone/head/BN giữ nguyên tuyệt đối. PT/ONNX đủ 1.605 val và 28 hardware-off White đạt. Dataset vẫn revision10, nhãn/split giữ nguyên.','Validation đã dùng phát triển, HA là retention; review còn partial. Không coi kết quả là kiểm thử độc lập hoặc bảo đảm không lọt NG sản xuất.','Bản fine-tune gốc được giữ riêng tại SERVICE/config/models/c_p_finetune_ha_r10_best111_20261029.{pt,onnx,json}. Model active/config không đổi.']
    atomic_text(root/'RESULTS.md','\n'.join(lines)+'\n');atomic_text(root/'index.html',(root/'comparison.html').read_text(encoding='utf8'));base.publish(root,'complete','spatial NG evaluated; frozen parent/parity/White verified; no deployment')

def campaign(root,resume=False):
    cmd="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,creationflags=FLAGS,check=True).stdout.strip();processes=json.loads(raw) if raw else [];processes=processes if isinstance(processes,list) else [processes]
    if any(x['ProcessId']!=os.getpid() and 'train_classifier_' in (x.get('CommandLine') or '') and '--phase' in x['CommandLine'] for x in processes):raise RuntimeError('Another classifier trainer is alive')
    if (root/'STOP_REQUESTED').exists():raise InterruptedError('STOP_REQUESTED')
    if resume and read(root/'status.json')['status']=='complete':raise ValueError('Do not resume completed run')
    p=check(root) if resume else prepare(root);script=root/'code_snapshot'/Path(__file__).name
    for phase,done in [('extract','features/complete.json'),('fit','artifacts/completion.json'),('evaluate','release_decision.json'),('replay','white_replay.json'),('finalize','completion_verification.json')]:
        if not (root/done).exists():base.checked(root,phase,[sys.executable,'-X','utf8','-u','-B',str(script),'--phase',phase,'--root',str(root)],root/'logs'/f'{phase}.log')
    if read(root/'status.json')['status']!='complete':base.publish(root,'complete','spatial NG complete; no deployment')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=['campaign','extract','fit','evaluate','replay','finalize'],required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args();root=args.root.resolve()
    try:
        if args.phase=='campaign':campaign(root,args.resume)
        else:globals()[args.phase](root)
    except Exception as exc:
        if root.exists():base.publish(root,'stopped' if isinstance(exc,InterruptedError) else 'failed',args.phase,error=repr(exc))
        traceback.print_exc();raise
if __name__=='__main__':main()
