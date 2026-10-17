"""Render offline UV evaluation images/diagnostics, without claiming unlabeled counts as recall."""
import argparse
import html
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--middle-regions', action='store_true', help='Label the new middle-support UV geometry')
    a = p.parse_args()
    first = json.loads((a.root / 'diagnostics_first/report.json').read_text(encoding='utf-8'))
    last = json.loads((a.root / 'v_candidates_last/report.json').read_text(encoding='utf-8'))
    candidate = 'h105_125_s35_v60'
    esc = html.escape
    parts = ['''<!doctype html><html lang="vi"><meta charset="utf-8"><title>UV HSV - HA 2026-10-06</title>
<style>body{font:16px system-ui;margin:24px;background:#171b22;color:#e6edf3}a{color:#7ac8ff}h1{font-size:26px}h2{margin-top:36px}.warning{border-left:5px solid #f1bb50;background:#332c20;padding:16px}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{padding:10px;border:1px solid #475260;text-align:left}figure{margin:12px 0;padding:12px;background:#252d38}img{width:100%;height:auto}figcaption{margin:7px 0;color:#bbcbdc}code{color:#96e2af}summary{cursor:pointer;padding:12px}.note{color:#acb9c9}</style>
<h1>Đánh giá bắt đĩa UV — HA</h1>
<div class="warning">Chỉ có 2 ảnh UV đơn lẻ. Geometry lấy từ ảnh White KHÁC thời điểm, cùng camera.<br>
Người dùng đã xác nhận cả hai ảnh là MIXING. Chưa có số lượng UV thật từng band để tính recall.<br>
Các số bên dưới là <b>cặp cạnh caliper thô</b>, không phải ground truth và không bị ép thành 25.<br>
HSV thử nghiệm chưa thay vào production; White/caliper dùng chung được giữ nguyên.</div>
<p>Ngưỡng cũ: <code>(107,35,40)..(119,150,250)</code>.<br>
Ngưỡng thử cân bằng: <code>(105,35,60)..(125,150,255)</code>. H rộng hơn, bỏ nền tối bằng V≥60, không cắt pixel V251–255. V≥80 làm mất nhiều cạnh mờ.</p>
<p>Mask xanh lá: pixel qua HSV và lọc nhiễu. Vạch vàng: scanline. Chấm/số đỏ: cặp cạnh được caliper đếm.
Những chấm bám nhiễu, hai chấm trên một dải hay dải xanh không có chấm cần rà soát riêng.</p>
<h2>Kết quả thay ngưỡng và thay White tham chiếu</h2><table><tr><th>Camera</th><th>White tham chiếu</th><th>HSV cũ band1 / band2</th><th>HSV thử V60 band1 / band2</th><th>V80 band1 / band2</th></tr>''']
    banner = ''
    if a.middle_regions:
        banner = ('<div class="warning"><b>VÙNG MỚI: hai phía hàng rãnh GIỮA theo yêu cầu người dùng.</b> '
                     'Band1 phía trên và Band2 phía dưới hàng giữa trong ảnh sau xoay/rectify; '
                     'tương ứng hai phía thanh đỡ giữa trong ảnh RAW. Không lấy thanh đỡ, không đổi HSV production. '
                     '<a href="../">Xem phép thử vùng ngoài cũ</a></div>')
    elif (a.root / 'middle_regions/index.html').exists():
        banner = ('<div class="warning">Đây là báo cáo VÙNG NGOÀI CŨ. '
                     '<a href="middle_regions/">Mở báo cáo mới: hai vùng GIỮA theo yêu cầu</a></div>')
    parts[0] = parts[0].replace('<h1>Đánh giá bắt đĩa UV — HA</h1>', '<h1>Đánh giá bắt đĩa UV — HA</h1>' + banner)
    for group, report in [('Đầu tiên', first), ('Gần UV nhất', last)]:
        for camera in report['cameras']:
            count = lambda key: ' / '.join(str(b['raw_pairs']) for b in camera['candidates'][key]['bands'])
            parts.append(f'<tr><td>Cam{camera["camera"]}</td><td>{group}: {esc(Path(camera["reference"]["white_path"]).name)}</td>'
                         f'<td>{count("legacy")}</td><td>{count(candidate)}</td><td>{count("h105_125_s35_v80")}</td></tr>')
    parts.append('''</table><p class="note">Không chọn threshold bằng cách làm count bằng số Mid1/Mid2 của ảnh White mượn.
Ảnh White và UV có thể khác số/loại/vị trí đĩa. Chỉ xem đây là thử khả năng tách màu và độ nhạy hình học.</p>''')
    for camera in first['cameras']:
        n = camera['camera']
        prefix = f'diagnostics_first/Cam{n}'
        parts.append(f'<h2>Camera {n}</h2><p>UV: {esc(Path(camera["uv_path"]).name)}<br>White tham chiếu: {esc(Path(camera["reference"]["white_path"]).name)}</p>')
        region = f'regions/first/Cam{n}'
        parts.append(f'<h3>Vùng đang được đánh giá trên ảnh UV TOÀN KHUNG</h3>'
                     f'<p>Viền cyan: CropBox mượn từ White. Xanh lá: Band1. Tím: Band2. Vàng: đường quét đếm thực tế. '
                     f'<b>Chỉ hai band này được kiểm tra, không phải toàn bộ vùng cyan.</b></p>'
                     f'<figure><figcaption>Ảnh UV RAW nguyên kích thước — click để phóng to</figcaption>'
                     f'<a href="{region}/uv_full_raw_regions.png" target="_blank"><img src="{region}/uv_full_raw_regions.png"></a></figure>'
                     f'<p><a href="{region}/uv_full_normalized_regions.png" target="_blank">Ảnh toàn khung sau xoay (tọa độ AI)</a> · '
                     f'<a href="{region}/uv_full_rectified_regions.png" target="_blank">Ảnh toàn vùng sau rectify</a> · '
                     f'<a href="regions/last/Cam{n}/uv_full_raw_regions.png" target="_blank">So sánh vùng dùng White tham chiếu cuối</a></p>')
        parts.append(f'<p><a href="{prefix}/uv_rectified.png" target="_blank">Xem toàn ảnh UV đã crop/rectify</a> · '
                     f'<a href="{prefix}/uv_normalized.png" target="_blank">Ảnh UV chuẩn hóa</a> · '
                     f'<a href="{prefix}/white_result.png" target="_blank">Kết quả White mượn geometry</a></p>')
        for band in (1, 2):
            parts.append(f'<h3>Band {band}</h3>')
            for key, caption in [('raw', 'Ảnh UV band gốc'), ('legacy', 'HSV cũ + caliper hiện tại'),
                                 (candidate, 'HSV thử V60 + CÙNG caliper hiện tại')]:
                path = f'{prefix}/band{band}_raw.png' if key == 'raw' else f'{prefix}/{key}_band{band}_overlay.png'
                detail = '' if key == 'raw' else f' — {camera["candidates"][key]["bands"][band-1]["raw_pairs"]} cặp cạnh'
                parts.append(f'<figure><figcaption>{caption}{detail} — click để xem pixel gốc</figcaption><a href="{path}" target="_blank"><img src="{path}" loading="lazy"></a></figure>')
        parts.append('<details><summary>So sánh tất cả 18 ngưỡng HSV</summary><table><tr><th>Candidate / HSV</th><th>Band1 / band2 thô</th><th>Mask + ảnh so sánh</th></tr>')
        for name, result in camera['candidates'].items():
            counts = ' / '.join(str(b['raw_pairs']) for b in result['bands'])
            parts.append(f'<tr><td>{esc(name)}<br>{result["lower"]}..{result["upper"]}</td><td>{counts}</td>'
                         f'<td><a target="_blank" href="{prefix}/{name}_bands_comparison.png">Raw / mask / scanline</a></td></tr>')
        parts.append('</table></details>')
    parts.append('''<h2>Vì sao chỉ thay HSV chưa đủ?</h2><ul>
<li>Nền xanh tối cùng hue với đĩa tạo các mảnh mask và cặp cạnh giả. Hạ S/V để bắt mờ làm nhiễu tăng.</li>
<li>Scanline chỉ ở giữa band; đĩa cong/giao nhau, lỗ phản xạ và mảnh màu có thể sinh nhiều cặp từ một dải.</li>
<li>Khoảng cách cạnh tối thiểu hiện là5px (band1) /4px (band2), khó với cạnh UV mảnh; giảm xuống1px cũng làm nhiễu tăng.</li>
<li>Quét95% bề ngang, hai band dùng tham số khác nhau. Thay White tham chiếu làm số đếm đổi.</li>
<li>Luật hiện tại chấp nhận khay khi bắt0 UV, hoặc số UV từng band bằng số White; khay được chấp nhận trả CountUvDisk=0.
Vì vậy không thể dùng CountUvDisk=0 hay OK API làm bằng chứng đã bắt đúng.</li></ul>
<p><b>Hướng xử lý tiếp:</b> bộ đếm riêng cho UV với độ liên tục dải trên nhiều scanline và gộp mảnh cùng cạnh;
đánh giá trên ảnh White+UV cùng khay, có cả đĩa UV, không UV và mixing. Không sửa caliper White và không ép đủ25.</p>
<p><a href="diagnostics_first/report.json">JSON phép thử chính / ablations</a> ·
<a href="v_candidates_last/report.json">JSON thử White gần UV nhất</a> · <a href="hsv_candidate.env">Ngưỡng thử OFFLINE</a></p></html>''')
    (a.root / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print(a.root / 'index.html')


if __name__ == '__main__':
    main()
