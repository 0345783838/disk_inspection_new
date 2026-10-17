"""Read-only before/after comparison on the SAME corrected ground truth."""
import argparse
import csv
import html
import json
from pathlib import Path

import cv2


def build(evaluation, previous, output):
    if output.exists():
        raise FileExistsError('Comparison reports are never overwritten')
    report = json.loads((evaluation/'report.json').read_text(encoding='utf-8'))
    records = json.loads((evaluation/'per_image.json').read_text(encoding='utf-8'))
    prior_report = json.loads((previous/'report.json').read_text(encoding='utf-8'))
    prior = {r['id']: r for r in json.loads((previous/'per_image.json').read_text(encoding='utf-8'))}
    if set(prior) != {r['id'] for r in records}:
        raise ValueError('Before/after image populations differ')
    for r in records:
        if r['label'] != prior[r['id']]['label']:
            raise ValueError('Before/after ground truth differs')
        r['original_label_before_review'] = prior[r['id']]['original_label']
        r['predictions']['previous'] = prior[r['id']]['predictions']['candidate']
    report['models']['previous'] = prior_report['models']['candidate']
    for domain, values in report['domains'].items():
        values['previous'] = prior_report['domains'][domain]['candidate']
    report['comparison_note'] = 'Fixed old production / previous fine-tune / new fine-tune; SAME corrected GT'
    output.mkdir(parents=True)
    table = []
    for domain in ('ha_val', 'ha_test', 'legacy_val'):
        values = report['domains'][domain]
        a, b = values['previous'], values['candidate']
        table.append(f'<tr><td>{domain}</td><td>{values["count"]}</td>'
                     f'<td>{a["accuracy"]*100:.2f}% → {b["accuracy"]*100:.2f}%</td>'
                     f'<td>{a["ng_not_detected"]}/{a["per_class"]["ng"]["support"]} → '
                     f'{b["ng_not_detected"]}/{b["per_class"]["ng"]["support"]}</td>'
                     f'<td>{a["no_disk_as_ok"]} → {b["no_disk_as_ok"]}</td></tr>')
    cards = []
    for r in sorted(records, key=lambda row: -row['predictions']['candidate']['confidence']):
        predictions = r['predictions']
        if all(p['label'] == r['label'] for p in predictions.values()):
            continue
        domain = 'legacy_val' if r['id'].startswith('legacy_') else 'ha_'+r['split']
        native = f'errors/{domain}/{r["id"]}.png'
        target = output/native
        target.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(r['path'])
        if image is None or not cv2.imwrite(str(target), image):
            raise ValueError('Unreadable native crop')
        candidate, previous_pred = predictions['candidate'], predictions['previous']
        miss = r['label'] == 'ng' and candidate['label'] != 'ng'
        regression = previous_pred['label'] == r['label'] and candidate['label'] != r['label']
        fixed = previous_pred['label'] != r['label'] and candidate['label'] == r['label']
        details = ''.join(f'<br>{name}: {predictions[key]["label"]} · {predictions[key]["confidence"]:.6f}'
                          for key, name in [('candidate', 'Sau train mới'), ('previous', 'Trước train mới / backup'),
                                            ('baseline', 'Model production gốc')])
        cards.append(f'<article data-domain="{domain}" data-miss="{int(miss)}" '
                     f'data-error="{int(candidate["label"]!=r["label"])}" data-regression="{int(regression)}" '
                     f'data-fixed="{int(fixed)}"><a href="{html.escape(native)}" target="_blank">'
                     f'<img loading="lazy" src="{html.escape(native)}"></a>'
                     f'<p><b>GT đã rà: {r["label"]}</b> · {domain}{details}</p>'
                     f'<small>{html.escape(r["id"])}</small><details><summary>Nhãn trước review / nguồn</summary>'
                     f'{r["original_label_before_review"]}<br>{html.escape(r["path"])}</details></article>')
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'per_image.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['domain', 'id', 'corrected_ground_truth', 'ground_truth_before_review'] + [
        f'{model}_{field}' for model in ('baseline', 'previous', 'candidate') for field in ('label', 'confidence')]
    with (output/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for r in records:
            row = {'domain': 'legacy_val' if r['id'].startswith('legacy_') else 'ha_'+r['split'],
                   'id': r['id'], 'corrected_ground_truth': r['label'],
                   'ground_truth_before_review': r['original_label_before_review']}
            for name, prediction in r['predictions'].items():
                row[name+'_label'], row[name+'_confidence'] = prediction['label'], prediction['confidence']
            writer.writerow(row)
    document = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Classifier — trước / sau train lại</title>
<style>:root{color-scheme:dark}body{background:#171b22;color:#eef1f6;font:15px system-ui;margin:24px}p{line-height:1.5}a{color:#9ed4ff}table{border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #657084;text-align:left}.controls{display:flex;flex-wrap:wrap;gap:14px;margin:20px 0}select{padding:8px;background:#293444;color:white}.grid{display:flex;flex-wrap:wrap;gap:14px}article{width:270px;background:#232d3c;padding:12px}article[hidden]{display:none}img{width:100%;height:210px;object-fit:contain}small,details{overflow-wrap:anywhere;color:#c0cddd}</style>
<h1>Classifier — trước / sau train lại bằng nhãn đã sửa</h1>
<p><b>Cùng một bộ nhãn đã rà</b> cho cả model trước và sau. Nhãn thay đổi: 37; chỉ 4 correction HA train vào học, correction validation/test giữ riêng. Model trước đợt train này đã backup riêng; <b>chưa triển khai model mới</b>.</p>
<p><b>Lưu ý:</b> HA test đã được xem và sửa nhãn, cùng phiên chụp; đây là đánh giá lại holdout đã rà, <b>không phải test mù độc lập</b>. 157 ảnh chỉ model cũ lệch chưa được rà lại; không phải audit toàn bộ dataset. NG bỏ sót = ng→ok hoặc ng→no_disk, tính cấp crop, không phải tỷ lệ PASS sai cả khay.</p>
<table><tr><th>Tập</th><th>Crop</th><th>Accuracy trước → sau</th><th>NG bỏ sót trước → sau</th><th>no_disk→ok trước → sau</th></tr>TABLE</table>
<p><a href="report.json">Báo cáo</a> · <a href="predictions.csv">CSV dự đoán + confidence</a> · <a href="http://127.0.0.1:8777/">Review trước (giữ nguyên)</a></p>
<div class="controls"><label>Xem <select id="mode"><option value="error">Tất cả lỗi model sau train mới</option><option value="miss">NG bị bỏ sót sau train mới</option><option value="regression">Trước đúng, sau sai</option><option value="fixed">Trước sai, sau đúng</option><option value="all">Ảnh bất kỳ model nào lệch nhãn đã rà</option></select></label><label>Tập <select id="domain"><option value="">Tất cả</option><option value="ha_test">HA test đã rà</option><option value="ha_val">HA validation</option><option value="legacy_val">Validation cũ</option><option value="ha_train">HA train (in-sample)</option></select></label><span id="count"></span></div>
<p>Chỉ xem kết quả, không sửa nhãn / train lại. Confidence model sau train mới xếp cao → thấp; bấm ảnh để xem PNG native.</p><div class="grid">CARDS</div>
<script>function filter(){let count=0;const mode=document.getElementById('mode').value,domain=document.getElementById('domain').value;for(const card of document.querySelectorAll('article')){card.hidden=!!((domain&&card.dataset.domain!==domain)||(mode!=='all'&&card.dataset[mode]!=='1'));if(!card.hidden)count++;}document.getElementById('count').textContent=count+' ảnh';}document.getElementById('mode').onchange=filter;document.getElementById('domain').onchange=filter;filter();</script></html>'''
    (output/'index.html').write_text(document.replace('TABLE', ''.join(table)).replace('CARDS', ''.join(cards)), encoding='utf-8')
    print('Saved before/after gallery', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.evaluation.resolve(), args.previous.resolve(), args.output.resolve())
