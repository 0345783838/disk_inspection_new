"""Offline report for UV-only rim tracking; no model loading or production writes."""
import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.tools.uv_rim_counter import count_uv_rims, load_rim_settings


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config' / 'config.env')
    args = parser.parse_args()
    settings = load_rim_settings(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    for reference, folder in (('first', 'diagnostics_first'), ('last', 'v_candidates_last')):
        prior = json.loads((args.input_root / folder / 'report.json').read_text(encoding='utf-8'))
        for camera in prior['cameras']:
            for band in (1, 2):
                raw_path = args.input_root / folder / f'Cam{camera["camera"]}' / f'band{band}_raw.png'
                image = cv2.imread(str(raw_path))
                result, images = count_uv_rims(image, band, settings)
                name = f'{reference}_Cam{camera["camera"]}_band{band}'
                output = args.output / name
                save(output / 'raw.png', image)
                for key, value in images.items():
                    save(output / f'{key}.png', value)
                result.update(camera=camera['camera'], band=band, reference=reference, folder=name,
                              input_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                              legacy_count=camera['candidates']['legacy']['bands'][band-1]['raw_pairs'],
                              v60_caliper_count=camera['candidates']['h105_125_s35_v60']['bands'][band-1]['raw_pairs'])
                stability = []
                if reference == 'first':
                    for label, variant in (
                        ('BGR-20', np.clip(image.astype(np.int16)-20, 0, 255).astype(np.uint8)),
                        ('BGR+20', np.clip(image.astype(np.int16)+20, 0, 255).astype(np.uint8)),
                        ('blur sigma0.6', cv2.GaussianBlur(image, (3, 3), .6))):
                        alternative, _ = count_uv_rims(variant, band, settings)
                        stability.append({'change': label, 'count': alternative['count'],
                                          'uncertain': alternative['uncertain_count']})
                result['synthetic_sensitivity_not_independent_accuracy'] = stability
                records.append(result)
                print(name, 'count', result['count'], 'uncertain', result['uncertain_count'], 'sensitivity', stability, flush=True)
    report = {'experimental_only': True, 'production_deployed': False, 'ground_truth_verified': False,
              'scope': '2 real UV captures, first/last unpaired White geometry; alternate references are NOT extra UV data',
              'settings': asdict(settings), 'records': records,
              'config_path': str(args.config.resolve()),
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'seconds': time.perf_counter()-started,
              'counter_sha256': hashlib.sha256(Path('src/tools/uv_rim_counter.py').read_bytes()).hexdigest()}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    esc = html.escape
    parts = ['''<!doctype html><html lang="vi"><meta charset="utf-8"><title>UV rim tracking — thử nghiệm</title>
<style>body{font:16px system-ui;background:#151b23;color:#e6edf3;margin:24px}a{color:#81caff}img{width:100%;height:auto}figure{margin:14px 0;padding:12px;background:#222d39}figcaption{margin-bottom:8px}table{border-collapse:collapse;margin:16px 0;width:100%}td,th{border:1px solid #4d6175;padding:9px}.note{background:#342e20;padding:16px;border-left:4px solid #eac052}h2{margin-top:40px}summary{cursor:pointer;padding:12px}</style>
<h1>Đếm đĩa UV theo cạnh dài — THỬ NGHIỆM, chưa deploy</h1>
<div class="note">Hai ảnh UV là MIXING, chưa có nhãn/số lượng UV thật từng band. Số bên dưới là kết quả thuật toán, KHÔNG phải accuracy.
White mượn khác thời điểm. Không ép25 đĩa, không dùng số Mid của White để chọn ngưỡng.
Xanh lá + số vàng: cạnh được đếm. Đường cam ? là vệt cần xác nhận, chưa cộng vào count.
Dấu x đỏ là ứng viên bị loại; không phải khẳng định mọi vệt đỏ đều là bóng.</div>
<p>__RIM_SETTINGS__ Không dùng contour-area để xóa cạnh1px.
Sau đó căn hướng fan, theo dõi toàn chiều dài band, gộp hai biên cùng cạnh và loại nhánh ngắn/ngược hướng.
Vệt thiếu độ sáng hoặc thiếu độ liên tục vẫn được ghi lại để rà soát.</p>
<table><tr><th>Reference/camera/band</th><th>Caliper HSV cũ</th><th>Caliper V60</th><th>Rim tracking</th><th>Cần xác nhận thêm</th></tr>''']
    for r in records:
        parts.append(f'<tr><td><a href="#{r["folder"]}">{r["folder"]}</a></td><td>{r["legacy_count"]}</td><td>{r["v60_caliper_count"]}</td><td>{r["count"]}</td><td>{r["uncertain_count"]}</td></tr>')
    description = (f'Config: {esc(str(args.config.resolve()))}. '
                   f'Mask H{settings.hue_low}–{settings.hue_high}, S≥{settings.saturation_low}, V≥{settings.value_low}, '
                   f'contrast≥{settings.weak_contrast}. Lõi trắng sát halo xanh: S≤{settings.core_saturation_max}, V≥{settings.core_value_min}. '
                   f'Tín hiệu mạnh V≥{settings.strong_value}, contrast≥{settings.strong_contrast}. '
                   f'Coverage≥{settings.min_coverage}, span≥{settings.min_span}, bright coverage≥{settings.min_strong_coverage}; '
                   f'min separation={settings.min_separation}px.')
    parts[0] = parts[0].replace('__RIM_SETTINGS__', description)
    parts.append('</table><p>Đổi sáng/blur bên dưới chỉ là kiểm tra nhạy tham số trên cùng ảnh, không tạo thêm dữ liệu kiểm chứng độc lập.</p>')
    for r in records:
        name = r['folder']
        parts.append(f'<h2 id="{name}">{name}: {r["count"]} cạnh được đếm + {r["uncertain_count"]} vệt cần xem lại</h2>')
        for key, caption in [('raw', '1. Ảnh gốc'), ('permissive_mask', '2. Mask mở rộng, gồm cả nhánh bóng'),
                             ('kept_mask', '3. Mask quanh cạnh được giữ sau tracking'), ('overlay', '4. Kết quả đánh số cạnh')]:
            path = f'{name}/{key}.png'
            parts.append(f'<figure><figcaption>{caption}</figcaption><a href="{path}" target="_blank"><img src="{path}" loading="lazy"></a></figure>')
        parts.append(f'<details><summary>Mask lõi sáng / ảnh căn hướng / thông số từng cạnh</summary>'
                     f'<p><a href="{name}/bright_seeds.png">Lõi sáng</a> · <a href="{name}/aligned_mask.png">Mask sau căn hướng</a></p>'
                     '<table><tr><th>Cạnh</th><th>Coverage</th><th>Bright coverage</th><th>Span</th></tr>')
        for t in r['tracks']:
            parts.append(f'<tr><td>{t["index"]}</td><td>{t["coverage"]:.2f}</td><td>{t["strong_coverage"]:.2f}</td><td>{t["span"]:.2f}</td></tr>')
        parts.append('</table>')
        for t in r['uncertain']:
            parts.append(f'<p>Vệt ? x≈{t["center_x"]}: mask coverage {t["coverage"]:.2f}, bright coverage {t["strong_coverage"]:.2f}; chưa cộng vào count.</p>')
        parts.append('</details>')
        if r['synthetic_sensitivity_not_independent_accuracy']:
            parts.append('<p>Nhạy với ảnh biến đổi: '+esc(str(r['synthetic_sensitivity_not_independent_accuracy']))+'</p>')
    parts.append('<p><a href="report.json">Chi tiết JSON</a> · <a href="../">Về báo cáo vùng giữa</a></p></html>')
    (args.output / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print('Report:', args.output / 'index.html', 'seconds', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
