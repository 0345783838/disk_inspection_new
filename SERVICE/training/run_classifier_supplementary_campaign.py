"""Run the authorized 200-epoch comparison serially on one GPU, then evaluate.

The campaign survives the chat turn. Each recipe retains its own protocol,
checkpoints and reports; no installed model or packaged backend is modified.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[2]
TRAINING = PROJECT / 'SERVICE' / 'training'
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
RECIPES = (
    ('head', 'Chỉ train phần phân loại, cố định BatchNorm', {'DISK_SUPPLEMENT_STABLE_HEAD': '1', 'DISK_SUPPLEMENT_FREEZE_MODULES': '10'}),
    ('bn', 'Train toàn mạng, cố định BatchNorm', {'DISK_SUPPLEMENT_STABLE_HEAD': '1', 'DISK_SUPPLEMENT_FREEZE_MODULES': '0'}),
    ('full', 'Train toàn mạng và BatchNorm', {'DISK_SUPPLEMENT_STABLE_HEAD': '0'}),
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def progress(root):
    result = []
    for recipe, title, _ in RECIPES:
        runs = []
        for name in ('group_cv_0', 'group_cv_1', 'group_cv_2', 'final_all_new'):
            p = root / recipe / 'training' / name / 'results.csv'
            if p.exists():
                try:
                    rows = list(csv.DictReader(p.open(encoding='utf-8')))
                    rows = [r for r in rows if r.get('epoch', '').strip().isdigit()]
                    if rows:
                        runs.append({'name': name, 'epoch': int(rows[-1]['epoch']), 'seconds': rows[-1].get('time'),
                                     'human_false_ng': rows[-1].get('metrics/new_human_false_ng'),
                                     'human_ng_missed': rows[-1].get('metrics/new_human_ng_not_detected')})
                except (OSError, csv.Error, ValueError):
                    pass  # A currently appended CSV line will be read on the next poll.
        result.append({'id': recipe, 'title': title, 'runs': runs,
                       'report_ready': (root / recipe / 'release_decision.json').exists()})
    return result


def publish(root, state, **updates):
    if updates.get('phase') != state.get('phase') or updates.get('status') in ('complete', 'failed'):
        with (root / 'campaign.log').open('a', encoding='utf-8') as handle:
            handle.write(datetime.now(timezone.utc).isoformat() + ' ' + json.dumps(updates, ensure_ascii=False) + '\n')
    state.update(updates, updated_at=datetime.now(timezone.utc).isoformat(), recipes=progress(root))
    write(root / 'status.json', state)


def checked_process(command, root, state, phase, log, env=None):
    with log.open('w', encoding='utf-8') as handle:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT, creationflags=FLAGS)
        while True:
            code = child.poll()
            publish(root, state, phase=phase, active_pid=child.pid if code is None else None)
            if code is not None:
                if code:
                    raise RuntimeError(f'{phase} exited {code}; see {log}')
                return
            time.sleep(15)


PAGE = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Train 200 epoch / patience 100</title>
<style>body{font:16px system-ui;background:#15212e;color:#e5edf5;margin:30px;max-width:1400px}a{color:#8ed0ff}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #64778a;padding:10px;text-align:left}.note{padding:16px;background:#423b27;line-height:1.6}pre{white-space:pre-wrap}li{margin:10px}</style>
<h1>Train bổ sung: 200 epoch, patience 100</h1>
<p id="state">Đang tải trạng thái…</p>
<p>Áp dụng cho 3 fold đánh giá và lượt train cuối của cả 3 phương án. Dừng sớm sau 100 epoch liên tiếp không cải thiện validation safety fitness. Lấy checkpoint tốt nhất; không mặc định lấy epoch cuối.</p>
<p class="note">113 crop người rà (75 NG / 26 OK / 12 no_disk); 2.687 crop còn lại là nhãn model được yêu cầu bổ sung. Chúng chưa phải nhãn người xác nhận. Kết quả trên dữ liệu đã train không phải độ chính xác trên bộ kiểm tra độc lập. Model và backend đang dùng chưa bị thay.</p>
<div id="progress"></div><div id="final"></div>
<p><a href="campaign_protocol.json">Cấu hình đã chốt</a> · <a href="status.json">Trạng thái JSON</a> · <a href="campaign.log">Log điều phối</a> · <a href="http://127.0.0.1:8785/">Dataset đã chuẩn bị</a> · <a href="http://127.0.0.1:8787/">Kết quả đợt ngắn trước</a></p>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function refresh(){try{const s=await(await fetch('status.json?t='+Date.now(),{cache:'no-store'})).json();
document.getElementById('state').textContent='Trạng thái: '+s.status+' / '+s.phase+' — cập nhật '+new Date(s.updated_at).toLocaleString('vi-VN');
let out='<table><tr><th>Phương án</th><th>Các lượt đã chạy</th><th>Báo cáo</th></tr>';
for(const r of s.recipes){out+='<tr><td>'+esc(r.title)+'</td><td>'+r.runs.map(x=>esc(x.name)+': '+x.epoch+'/200 epoch').join('<br>')+'</td><td>'+(r.report_ready?'<a href="'+r.id+'/">Xem kết quả</a>':'Đang chờ')+' · <a href="'+r.id+'_job.log">Log</a></td></tr>'}document.getElementById('progress').innerHTML=out+'</table>';
if(s.status==='complete')document.getElementById('final').innerHTML='<p><a href="comparison.html">Mở so sánh cuối và model đã chọn</a></p>';
if(s.status==='failed')document.getElementById('final').textContent=s.error;
}catch(e){document.getElementById('state').textContent='Chưa đọc được trạng thái: '+e.message}}
refresh();setInterval(refresh,10000);
</script></html>'''


def finalize(root, state):
    candidates = []
    rows = []
    for key, title, _ in RECIPES:
        folder = root / key
        release = read(folder / 'release_decision.json')
        cv = read(folder / 'cv_selection.json')
        chosen = next(x for x in cv['summaries'] if x['fraction'] == cv['chosen_fraction'])
        m, oof = release['candidate']['metrics'], chosen['metrics']['new_human']
        rows.append({'recipe': key, 'title': title, 'fraction': release['cv_fraction'],
                     'recommended': release['recommended_for_review'], 'new_human': m['new_human'],
                     'new_human_oof': oof, 'old_ha': m['old_ha'], 'legacy_val': m['legacy_val'],
                     'retention_failures': release['candidate']['retention_failures']})
        if release['recommended_for_review'] and chosen['eligible_all_folds']:
            # Recipe selection uses grouped held-out predictions, never final fitted error counts.
            candidates.append(((oof['ng_not_detected'], oof['false_ng'], chosen['metrics']['new_all']['errors']), key, release))
    selection = {'training_complete': True, 'production_deployed': False, 'comparison': rows,
                 'selection_basis': 'Grouped new-capture CV, followed by final retention and reviewed-NG safety gates; fitted improvement alone cannot select a recipe.'}
    if candidates:
        _, key, release = min(candidates, key=lambda x: x[0])
        selection.update(selected_recipe=key, selected_onnx=release['selected_onnx'], selected_pt=release['selected_pt'],
                         onnx_sha256=release['onnx_sha256'], pt_sha256=release['pt_sha256'])
        # Replay the actual White pipeline with hardware disabled; installed files stay unchanged.
        checked_process([str(PROJECT / 'SERVICE/.venv/Scripts/python.exe'), '-u',
                         str(PROJECT / '.work/classifier_supplementary_20261020/verify_white_replay.py'),
                         '--root', str(root / key)], root, state, f'{key}: white replay', root / 'white_replay.log')
        replay = read(root / key / 'white_replay.json')
        selection['white_replay'] = {k: v for k, v in replay.items() if k != 'records'}
        selection['white_verdict_changes'] = sum(r['baseline_white_result'] != r['candidate']['Result'] for r in replay['records'])
        sys.path.insert(0, str(TRAINING))
        from evaluate_classifier_ha import torch_onnx_parity
        records = read(root / key / 'evaluations/final/predictions.json')
        reviewed = [r for r in records if r['origin'] == 'new_factory' and r['human_reviewed']]
        def margin(r):
            p = sorted(r['prediction']['probabilities'].values(), reverse=True)
            return p[0] - p[1]
        reviewed = sorted(reviewed, key=margin)[:32]
        selection['new_review_export_parity'] = torch_onnx_parity(Path(release['selected_pt']), Path(release['selected_onnx']),
                                                                 [dict(r, path=r['image']) for r in reviewed])
    else:
        selection.update(selected_recipe=None, reason='No learned update passed the frozen safety/retention gates and improved reviewed false NG. Existing runtime retained.')
    protected = read(root / 'protected_runtime_hashes.json')
    assert all(digest(p) == h for p, h in protected.items()), 'Installed runtime changed'
    protocol = read(root / 'campaign_protocol.json')
    assert digest(Path(protocol['pool']) / 'manifest.json') == protocol['pool_manifest_sha256']
    selection['protected_runtime_files_unchanged'] = True
    write(root / 'final_selection.json', selection)
    page = ['<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kết quả train 200 epoch</title>',
            '<style>body{font:16px system-ui;margin:30px;max-width:1400px}td,th{border:1px solid #aaa;padding:10px}table{border-collapse:collapse}</style>',
            '<h1>So sánh đợt 200 epoch / patience 100</h1>',
            '<p>Baseline: 38 NG ảo / 0 bỏ sót NG trên 113 crop đã rà. Legacy: 5 bỏ sót NG / 4 no_disk thành OK. Old HA: 0 bỏ sót NG / 1 NG ảo.</p>',
            '<table><tr><th>Phương án</th><th>Tỷ lệ cập nhật</th><th>CV: NG ảo / bỏ sót NG</th><th>Sau train: NG ảo / bỏ sót NG</th><th>Legacy: bỏ sót NG / no_disk thành OK</th><th>Qua điều kiện</th></tr>']
    for r in rows:
        page.append(f'<tr><td><a href="{r["recipe"]}/">{html.escape(r["title"])}</a></td><td>{r["fraction"]}</td>'
                    f'<td>{r["new_human_oof"]["false_ng"]} / {r["new_human_oof"]["ng_not_detected"]}</td>'
                    f'<td>{r["new_human"]["false_ng"]} / {r["new_human"]["ng_not_detected"]}</td>'
                    f'<td>{r["legacy_val"]["ng_not_detected"]} / {r["legacy_val"]["no_disk_as_ok"]}</td>'
                    f'<td>{r["recommended"]}: {html.escape(", ".join(r["retention_failures"]))}</td></tr>')
    page += ['</table>', f'<p>Phương án chọn để rà tiếp: {html.escape(str(selection["selected_recipe"]))}. Chưa triển khai lên máy.</p>',
             '<p>113 nhãn người rà được chọn vì model cũ báo NG; 2.687 nhãn khác chưa được người rà. CV là ước lượng chọn model trong cùng bộ thu thập, không phải bộ test mù độc lập. Kết quả sau train trên bộ mới là kết quả đã fit.</p>',
             '<p><a href="final_selection.json">Model, checksum và bằng chứng kiểm tra</a> · <a href="index.html">Tiến độ và số epoch thực chạy</a></p></html>']
    (root / 'comparison.html').write_text('\n'.join(page), encoding='utf-8')


def main(args):
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError('Campaign output must be new: ' + str(root))
    pool, pretrained = args.pool.resolve(), args.pretrained.resolve()
    root.mkdir(parents=True)
    (root / 'index.html').write_text(PAGE, encoding='utf-8')
    protected = read(PROJECT / '.work/classifier_supplementary_20261020/protected_runtime_hashes.json')
    assert all(digest(p) == h for p, h in protected.items()), 'Baseline runtime differs from previous audit'
    write(root / 'protected_runtime_hashes.json', protected)
    manifest = read(pool / 'manifest.json')
    assert len(manifest['records']) == 8570
    assert all(digest(pool / r['image']) == r['sha256'] for r in manifest['records']), 'Dataset crop changed'
    write(root / 'input_validation.json', {'records': 8570, 'all_crop_hashes_match': True,
          'training_pool': 5800, 'legacy_validation_only': 2770,
          'new_human_reviewed': 113, 'new_model_proposals': 2687,
          'manifest_sha256': digest(pool / 'manifest.json')})
    snapshot = root / 'code_snapshot'
    snapshot.mkdir()
    source_hashes = {}
    for p in TRAINING.glob('*.py'):
        shutil.copy2(p, snapshot / p.name)
        source_hashes[str(p)] = digest(p)
    write(root / 'campaign_protocol.json', {'created_at': datetime.now(timezone.utc).isoformat(),
          'epochs': 200, 'patience': 100, 'pool': str(pool), 'pool_manifest_sha256': digest(pool / 'manifest.json'),
          'pretrained': str(pretrained), 'training_authorized': True, 'production_deployed': False,
          'authorization': 'User requested 200 epochs and early stopping after 100 unimproved epochs on 2026-10-21.',
          'fractions_predeclared': [0., .05, .1, .15, .2, .25, .5, .75, 1.],
          'recipe_order': [r[0] for r in RECIPES], 'source_hashes': source_hashes})
    state = {'status': 'running', 'pid': os.getpid(), 'epochs': 200, 'patience': 100, 'phase': 'initializing'}
    try:
        for key, title, env_updates in RECIPES:
            assert all(digest(p) == h for p, h in source_hashes.items()), 'Training source changed mid-campaign'
            env = os.environ.copy()
            env.pop('DISK_SUPPLEMENT_STABLE_HEAD', None)
            env.pop('DISK_SUPPLEMENT_FREEZE_MODULES', None)
            env.update(env_updates, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
            command = [sys.executable, '-u', '-B', str(TRAINING / 'train_classifier_supplementary.py'),
                       '--phase', 'run', '--root', str(root / key), '--pool', str(pool), '--pretrained', str(pretrained),
                       '--epochs', '200', '--patience', '100', '--fractions', '0,.05,.1,.15,.2,.25,.5,.75,1']
            checked_process(command, root, state, title, root / f'{key}_job.log', env)
            write(root / key / 'protected_runtime_hashes.json', protected)
            assert all(digest(p) == h for p, h in protected.items()), 'Installed runtime changed'
        finalize(root, state)
        publish(root, state, status='complete', phase='Training, comparison and evaluation completed', active_pid=None)
    except Exception as exc:
        publish(root, state, status='failed', error=str(exc), active_pid=None)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    main(parser.parse_args())
