"""Read-only comparison gallery, defaulting to true NG missed by the new model."""
import argparse
import html
import json
from pathlib import Path


def render(root):
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    records=json.loads((root/'per_image.json').read_text(encoding='utf-8'))
    table=[]
    for domain in ('ha_val','ha_test','legacy_val'):
        if domain not in report['domains']:continue
        values=report['domains'][domain]
        old,new=values['baseline'],values['candidate']
        table.append(f'<tr><td>{domain}</td><td>{values["count"]}</td>'
                     f'<td>{old["accuracy"]*100:.2f}% → {new["accuracy"]*100:.2f}%</td>'
                     f'<td>{old["ng_as_ok"]+old["ng_as_no_disk"]} → {new["ng_as_ok"]+new["ng_as_no_disk"]}</td>'
                     f'<td>{new["ng_as_ok"]} / {new["ng_as_no_disk"]}</td></tr>')
    cards=[]
    records.sort(key=lambda r:-r['predictions']['candidate']['confidence'])
    for r in records:
        old,new=r['predictions']['baseline'],r['predictions']['candidate']
        if old['label']==r['label'] and new['label']==r['label']:continue
        domain='legacy_val' if r['id'].startswith('legacy_') else 'ha_'+r['split']
        native=f'errors/{domain}/{r["id"]}.png'
        assert (root/native).exists()
        miss=r['label']=='ng' and new['label']!='ng'
        cards.append(f'<article data-domain="{domain}" data-miss="{int(miss)}" data-error="{int(new["label"]!=r["label"])}">'
                     f'<a href="{html.escape(native)}" target="_blank"><img loading="lazy" src="{html.escape(native)}"></a>'
                     f'<p><b>GT: {r["label"]}</b> · {domain}<br>Model mới: {new["label"]} · {new["confidence"]:.6f}<br>'
                     f'Model cũ: {old["label"]} · {old["confidence"]:.6f}</p><small>{html.escape(r["id"])}</small>'
                     f'<details><summary>Nguồn / scores model mới</summary>{html.escape(r["path"])}<br>'
                     f'{html.escape(json.dumps(new["probabilities"]))}</details></article>')
    document='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Classifier HA — kết quả training</title>
<style>:root{color-scheme:dark}body{background:#171b22;color:#eee;font:15px system-ui;margin:24px}a{color:#9ed4ff}p{line-height:1.5}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #657084;text-align:left}select{padding:8px;background:#2b3442;color:white}.grid{display:flex;flex-wrap:wrap;gap:14px;margin-top:20px}article{width:270px;background:#232c38;padding:12px}img{width:100%;height:200px;object-fit:contain}small,details{overflow-wrap:anywhere;color:#c7d2e3}.controls{margin:18px 0;display:flex;gap:16px;align-items:center;flex-wrap:wrap}article[hidden]{display:none}</style>
<h1>Classifier mới — ưu tiên chống lọt NG</h1>
<p>Ảnh và confidence là kết quả model. GT = nhãn HA mày đã xác nhận hoặc nhãn dataset cũ. Trang này <b>chỉ đọc</b>, không sửa nhãn / không train lại.</p>
<p><b>NG bị bỏ sót</b> = GT ng nhưng model đoán ok hoặc no_disk. APP hiện gửi OK cho PLC khi WARNING, nên ng → no_disk cũng là nguy cơ. Đây là thống kê <b>cấp crop</b>, không phải tỷ lệ PASS sai của cả khay.</p>
<table><tr><th>Tập</th><th>Crop</th><th>Accuracy cũ → mới</th><th>NG bỏ sót cũ → mới</th><th>Model mới: ng→ok / ng→no_disk</th></tr>TABLE</table>
<p>HA test: bốn ảnh gốc giữ riêng, cùng phiên chụp; chưa phải test từ phiên mới độc lập. Tập cũ là validation và đã tham gia chọn model. Model <b>chưa được triển khai</b>.</p>
<p><a href="report.json">Báo cáo đầy đủ</a> · <a href="predictions.csv">CSV dự đoán + confidence</a></p>
<div class="controls"><label>Xem <select id="mode"><option value="miss">NG bị model mới bỏ sót</option><option value="error">Tất cả ảnh model mới đoán sai</option><option value="all">Ảnh một trong hai model đoán sai</option></select></label><label>Tập <select id="domain"><option value="">Tất cả</option><option value="ha_test">HA test</option><option value="ha_val">HA validation</option><option value="legacy_val">Validation cũ</option><option value="ha_train">HA train (in-sample)</option></select></label><span id="count"></span></div>
<p>Confidence model mới xếp cao → thấp để nhìn các lỗi tự tin sai trước. Ảnh PNG giữ kích thước gốc; bấm để xem ảnh native.</p><div class="grid">CARDS</div>
<script>function filter(){let count=0;const mode=document.getElementById('mode').value,domain=document.getElementById('domain').value;for(const card of document.querySelectorAll('article')){card.hidden=!!((domain&&card.dataset.domain!==domain)||(mode==='miss'&&card.dataset.miss!=='1')||(mode==='error'&&card.dataset.error!=='1'));if(!card.hidden)count++;}document.getElementById('count').textContent=count+' ảnh';}document.getElementById('mode').onchange=filter;document.getElementById('domain').onchange=filter;filter();</script></html>'''
    (root/'index.html').write_text(document.replace('TABLE',''.join(table)).replace('CARDS',''.join(cards)),encoding='utf-8')
    print('Read-only gallery:',root/'index.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    render(parser.parse_args().evaluation.resolve())
