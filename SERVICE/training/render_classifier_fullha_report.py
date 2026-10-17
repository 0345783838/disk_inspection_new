"""Separate group-CV selection estimates from full-HA seen-data fitting comparisons."""
import argparse
import csv
import html
import json
from pathlib import Path

import cv2


def render(root):
    decision_path=root/('release_decision_v2.json' if (root/'release_decision_v2.json').exists() else 'release_decision.json')
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    selection = json.loads((root/'cv_selection_checked.json').read_text(encoding='utf-8'))
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    output = root/'report'
    if output.exists():
        raise FileExistsError('Full-HA report is never overwritten')
    output.mkdir()
    reports, predictions = {}, {}
    names=['retained_previous','full_from_original','full_from_backup']
    names.extend(name for name in ('blend_cold25','blend_cold50') if (root/'evaluations'/name/'report.json').exists())
    for name in names:
        folder = root/'evaluations'/name
        reports[name] = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        predictions[name] = {r['id']: r for r in json.loads((folder/'per_image.json').read_text(encoding='utf-8'))}
    chosen = next(name for name,r in reports.items() if r['model'] == decision['chosen_source'])
    old, new = reports['retained_previous'], reports[chosen]
    rows = []
    for identifier, r in predictions[chosen].items():
        values = {name: p[identifier]['prediction'] for name,p in predictions.items()}
        if any(p[identifier]['label'] != r['label'] for p in predictions.values()):
            raise ValueError('Before/after corrected labels differ')
        rows.append(dict(r, predictions=values, original_ha_id=r.get('source_ha_id',identifier)))
    report = {'protocol': {key: protocol[key] for key in ('correction_sha256','hard_ids',
               'final_protocol','cross_validation_limitation','cv_initialization','old_holdout')},
              'cv_selection': selection, 'final_reports': reports, 'release_decision': decision,
              'chosen_name': chosen, 'production_deployed': False}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'per_image.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    fields = ['domain','id','ha_id','ground_truth','previous_label','previous_confidence','selected_label','selected_confidence']
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in rows:
            a,b=r['predictions']['retained_previous'],r['predictions'][chosen]
            writer.writerow({'domain':r['domain'],'id':r['id'],'ha_id':r['original_ha_id'],'ground_truth':r['label'],
                'previous_label':a['label'],'previous_confidence':a['confidence'],
                'selected_label':b['label'],'selected_confidence':b['confidence']})
    cv_table=[]
    for summary in selection['summaries']:
        m=summary['ha_group_cv_selection_estimate']
        cv_table.append(f'<tr><td>{summary["recipe"]}</td><td>{m["accuracy"]*100:.2f}%</td>'
                        f'<td>{m["ng_not_detected"]}/{m["per_class"]["ng"]["support"]}</td>'
                        f'<td>{summary["all_folds_eligible"]}</td></tr>')
    fit_table=[]
    for domain in ('ha_fit','legacy_val'):
        a,b=old['domains'][domain],new['domains'][domain]
        caption='HA đã đưa toàn bộ vào train — IN-SAMPLE' if domain=='ha_fit' else 'Validation cũ — không vào train mới'
        fit_table.append(f'<tr><td>{caption}</td><td>{b["count"]}</td><td>{a["accuracy"]*100:.2f}% → {b["accuracy"]*100:.2f}%</td>'
                         f'<td>{a["ng_not_detected"]}/{a["per_class"]["ng"]["support"]} → '
                         f'{b["ng_not_detected"]}/{b["per_class"]["ng"]["support"]}</td></tr>')
    cards=[]
    hard=set(protocol['hard_ids'])
    for r in sorted(rows,key=lambda v:-v['predictions'][chosen]['confidence']):
        a,b=r['predictions']['retained_previous'],r['predictions'][chosen]
        is_hard=r['original_ha_id'] in hard
        if a['label']==r['label'] and b['label']==r['label'] and not is_hard:
            continue
        native=f'images/{r["domain"]}/{r["id"]}.png'
        target=output/native;target.parent.mkdir(parents=True,exist_ok=True)
        image=cv2.imread(r['path'])
        if image is None or not cv2.imwrite(str(target),image):raise ValueError('Missing native crop')
        error=b['label']!=r['label'];miss=r['label']=='ng' and b['label']!='ng'
        fixed=a['label']!=r['label'] and b['label']==r['label']
        regression=a['label']==r['label'] and error
        cards.append(f'<article data-domain="{r["domain"]}" data-error="{int(error)}" data-miss="{int(miss)}" '
                     f'data-fixed="{int(fixed)}" data-hard="{int(is_hard)}" data-regression="{int(regression)}">'
                     f'<a href="{html.escape(native)}" target="_blank"><img loading="lazy" src="{html.escape(native)}"></a>'
                     f'<p><b>GT: {r["label"]}</b> · {r["domain"]}<br>Trước / backup: {a["label"]} · {a["confidence"]:.6f}'
                     f'<br>Bản được chọn: {b["label"]} · {b["confidence"]:.6f}</p>'
                     f'<small>{html.escape(r["original_ha_id"])}</small></article>')
    document='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Classifier — học toàn bộ HA</title>
<style>:root{color-scheme:dark}body{background:#171b22;color:#eef1f6;font:15px system-ui;margin:24px}a{color:#a0d7ff}p{line-height:1.5}table{border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #61728a;text-align:left}.warning{border:1px solid #f1b868;padding:14px;background:#3a3020}select{padding:8px;background:#293444;color:#fff}.controls{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}.grid{display:flex;flex-wrap:wrap;gap:14px}article{width:270px;padding:12px;background:#232d3c}article[hidden]{display:none}img{width:100%;height:210px;object-fit:contain}small{color:#cad5e7;overflow-wrap:anywhere}</style>
<h1>Classifier — học toàn bộ 2.000 crop HA đã rà nhãn</h1>
<p>Bản chọn: CHOSEN. Model tốt trước đợt này vẫn được backup riêng. <b>Chưa thay model thực tế.</b></p>
<p class="warning"><b>HA bên dưới đã được model cuối học toàn bộ, gồm cả 800 crop validation/test cũ.</b> Số liệu đó là fitting trên dữ liệu đã học, KHÔNG phải accuracy test độc lập. Kiểm tra chéo ở bảng riêng: mỗi ảnh gốc giữ riêng một lần, checkpoint khởi đầu chưa học HA. Tuy nhiên cùng phiên chụp, đã rà nhãn và dùng chọn cách train, nên cũng chưa chứng minh chất lượng ở phiên chụp mới.</p>
<h2>Kiểm tra chéo theo ảnh gốc — 3 lượt / 2 cách lấy mẫu</h2>
<p>Mỗi crop HA được giữ ngoài train một lần trong mỗi recipe; đây là ước lượng kiểm tra chéo dùng chọn recipe/checkpoint, không phải test mù cuối cùng.</p>
<table><tr><th>Recipe</th><th>Accuracy HA kiểm tra chéo</th><th>NG bỏ sót</th><th>Mọi lượt đạt retention tập cũ</th></tr>CV_TABLE</table>
<h2>Bản cuối so với model backup — cùng nhãn đã sửa</h2>
<table><tr><th>Tập / phạm vi</th><th>Crop</th><th>Accuracy trước → bản chọn</th><th>NG bỏ sót trước → bản chọn</th></tr>FIT_TABLE</table>
<p>NG bỏ sót = ng→ok hoặc ng→no_disk; thống kê từng crop, không phải tỷ lệ PASS sai cả khay. Không hạ confidence threshold. Tập cũ giữ riêng trong đợt train này nhưng checkpoint gốc đã có old train/val overlap.</p>
<p><a href="report.json">Báo cáo đầy đủ</a> · <a href="predictions.csv">CSV nhãn + confidence</a></p>
<div class="controls"><label>Xem <select id="mode"><option value="hard">19 ảnh HA khó / đã đổi nhãn</option><option value="error">Tất cả lỗi bản được chọn</option><option value="miss">NG còn bị bỏ sót</option><option value="fixed">Trước sai, nay đúng</option><option value="regression">Trước đúng, nay sai</option><option value="all">Tất cả ảnh trong gallery</option></select></label><label>Tập <select id="domain"><option value="">Tất cả</option><option value="ha_fit">HA đã học — in-sample</option><option value="legacy_val">Validation cũ</option></select></label><span id="count"></span></div>
<p>Bấm ảnh để xem PNG native. Trang chỉ xem kết quả, không sửa nhãn / train lại.</p><div class="grid">CARDS</div>
<script>function filter(){let n=0;const mode=document.getElementById('mode').value,domain=document.getElementById('domain').value;for(const c of document.querySelectorAll('article')){c.hidden=!!((domain&&c.dataset.domain!==domain)||(mode!=='all'&&c.dataset[mode]!=='1'));if(!c.hidden)n++;}document.getElementById('count').textContent=n+' ảnh';}document.getElementById('mode').onchange=filter;document.getElementById('domain').onchange=filter;filter();</script></html>'''
    document=document.replace('CHOSEN',html.escape(chosen)).replace('CV_TABLE',''.join(cv_table)).replace('FIT_TABLE',''.join(fit_table)).replace('CARDS',''.join(cards))
    (output/'index.html').write_text(document,encoding='utf-8')
    print('Saved full-HA gallery',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    render(parser.parse_args().root.resolve())
