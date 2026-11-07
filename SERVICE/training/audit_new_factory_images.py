"""Replay individual factory White/UV frames with production settings, without hardware.

UV needs White geometry: nearest-time references are explicitly unverified pairs.
Outputs include every classifier crop, raw detections, masks and actual scan counts.
No thresholds, models or installed files are changed.
"""
import argparse
import ast
import base64
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import time

import cv2
import decouple
import numpy as np

from training.smoke_local_white import normalize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')


def stamp(path):
    return datetime.strptime(path.stem.removeprefix('Image_'), '%Y%m%d%H%M%S%f')


def load(path, camera, config):
    raw = cv2.imread(str(path))
    if raw is None:
        raise ValueError(f'Cannot read {path}')
    image = normalize(raw, camera, config)
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('JPEG encode failed')
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def decode(encoded):
    return cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)


def marked(image, pairs, center):
    image = image.copy()
    cv2.line(image, (0, center[1]), (image.shape[1]-1, center[1]), (255, 180, 0), 1)
    for index, pair in enumerate(pairs, 1):
        p = np.rint((np.array(pair['e1']['point'])+pair['e2']['point'])/2).astype(int)
        cv2.circle(image, tuple(p), 3, (0, 0, 255), -1)
        cv2.putText(image, str(index), (int(p[0])-5, max(13, int(p[1])-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
    return image


class Capture:
    """Wrap only this serial diagnostic process; preserve original inference returns."""
    def __init__(self, service):
        self.service = service
        self.reset()
        detector = service.disk_point_detect_model
        original_detect = detector.detect_objects_debug
        def detect(*args, **kwargs):
            result = original_detect(*args, **kwargs)
            self.detected = [x.copy() for x in result]
            return result
        detector.detect_objects_debug = detect
        original_rects = service.get_line_boxes_ratio_shift
        def rects(*args, **kwargs):
            result = original_rects(*args, **kwargs)
            self.rects.append(result)
            return result
        service.get_line_boxes_ratio_shift = rects
        original_crop = service.crop_boxes
        def crops(*args, **kwargs):
            result = original_crop(*args, **kwargs)
            self.crops.append([x.copy() for x in result])
            return result
        service.crop_boxes = crops
        classifier = service.point_classification_model
        original_run = classifier.ort_sess.run
        def run(*args, **kwargs):
            result = original_run(*args, **kwargs)
            self.scores.extend(result[0].copy().tolist())
            return result
        classifier.ort_sess.run = run
        original_predict = classifier.predict_batch
        def predict(*args, **kwargs):
            result = original_predict(*args, **kwargs)
            self.classified.append(result)
            return result
        classifier.predict_batch = predict
        original_segment = service.disk_segmentor_yolo.segment_large_image_debug
        def segment(image, *args, **kwargs):
            result = original_segment(image, *args, **kwargs)
            self.bands.append({'image': image.copy(), 'raw': result[0].copy()})
            return result
        service.disk_segmentor_yolo.segment_large_image_debug = segment
        original_clean = service.clean_mask
        def clean(*args, **kwargs):
            result = original_clean(*args, **kwargs)
            next(b for b in self.bands if 'filtered' not in b)['filtered'] = result.copy()
            return result
        service.clean_mask = clean
        original_measure = service.caliper.measure_with_params
        def measure(mask, center, *args, **kwargs):
            result = original_measure(mask, center, *args, **kwargs)
            self.measures.append({'center': list(center), 'pairs': result['pairs']})
            return result
        service.caliper.measure_with_params = measure
        original_uv = service.preprocess_uv_image_hsv
        def uv(image, *args, **kwargs):
            result = original_uv(image, *args, **kwargs)
            self.uv_bands.append({'image': image.copy(), 'raw': result.copy()})
            return result
        service.preprocess_uv_image_hsv = uv
        original_uv_clean = service.remove_mask_noise_uv
        def uv_clean(*args, **kwargs):
            result = original_uv_clean(*args, **kwargs)
            next(b for b in self.uv_bands if 'filtered' not in b)['filtered'] = result.copy()
            return result
        service.remove_mask_noise_uv = uv_clean

    def reset(self):
        self.detected = None
        self.rects, self.crops, self.classified, self.scores = [], [], [], []
        self.bands, self.measures, self.uv_bands = [], [], []


def contact_sheet(crops, labels, confidences):
    sheet = np.full((5*125, 5*150, 3), 26, np.uint8)
    for i, (crop, label, confidence) in enumerate(zip(crops, labels, confidences)):
        x, y = i%5*150, i//5*125
        if i >= 25:
            break
        h, w = crop.shape[:2]
        scale = min(144/w, 96/h)
        thumb = cv2.resize(crop, (max(1, round(w*scale)), max(1, round(h*scale))))
        sheet[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
        color = (70, 70, 255) if label == 'ng' else ((0, 180, 255) if label == 'no_disk' else (90, 230, 100))
        cv2.putText(sheet, f'{i+1}: {label} {confidence:.3f}', (x+2, y+116), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--app-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_root, runtime, app_config, output = (p.resolve() for p in (args.input, args.runtime, args.app_config, args.output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config_path = runtime/'config/config.env'
    config = decouple.Config(decouple.RepositoryEnv(str(config_path)))
    decouple.config = config
    protected = [config_path, app_config, runtime/'main.exe']
    protected += [runtime/config(key) for key in ('DISK_SEGMENT_MODEL_PATH', 'DISK_POINT_DETECT_MODEL_PATH', 'POINT_CLASSIFY_MODEL_PATH')]
    sources = sorted(source_root.glob('*/*.bmp'))
    before = {str(p): digest(p) for p in protected+sources}
    os.chdir(runtime)
    from src.service.check_disk_service_yolo import DiskCheckingService
    from src.tools.uv_rim_counter import count_uv_rims, load_rim_settings
    service = DiskCheckingService()
    capture = Capture(service)
    params = service._default_white_params()
    white, uv = [], []
    responses = {}
    directions = ('R1_bottom', 'R2_top', 'R2_bottom', 'R3_top')
    for camera in (1, 2):
        for path in sorted((source_root/f'white_cam{camera}').glob('*.bmp')):
            started = time.perf_counter()
            capture.reset()
            image = load(path, camera, app_config)
            folder = output/f'white_cam{camera}'/path.stem
            save(folder/'normalized.jpg', image)
            response = service.check_disk_white(image.copy())
            responses[str(path)] = response
            save(folder/'result.png', decode(response.ResImg))
            record = {'camera': camera, 'source': str(path), 'folder': folder.relative_to(output).as_posix(),
                      'response': response.model_dump(exclude={'ResImg'}), 'classifications': [], 'bands': []}
            boxes, scores, classes = capture.detected
            record['raw_detections'] = [{'box': b.tolist(), 'confidence': float(c)} for b,c in zip(boxes,scores)]
            record['raw_point_count'] = len(boxes)
            detector_image = image.copy()
            service.draw_detected_boxes(detector_image, boxes, scores, classes)
            for inferred in response.InferredSlots:
                x1,y1,x2,y2 = map(int,inferred['box'])
                cv2.rectangle(detector_image, (x1,y1), (x2,y2), (0,215,255), 3)
                cv2.putText(detector_image, f"INFER R{inferred['row']} S{inferred['slot']}", (x1,max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX,.5,(0,215,255),1)
            save(folder/'points.jpg', detector_image)
            offset = 0
            for face, rects, crops, (labels, confs) in zip(directions, capture.rects, capture.crops, capture.classified):
                save(folder/f'{face}.jpg', contact_sheet(crops, labels, confs))
                for slot, (rect, crop, label, conf) in enumerate(zip(rects,crops,labels,confs),1):
                    name = f'crops/{face}_{slot:02}_{label}.jpg'
                    save(folder/name, crop)
                    record['classifications'].append({'face':face, 'slot':slot, 'box':rect, 'label':label,
                        'confidence':conf, 'scores':dict(zip(service.point_classification_model.labels,capture.scores[offset])), 'crop':name})
                    offset += 1
            record['class_counts'] = dict(Counter(c['label'] for c in record['classifications']))
            for i, band in enumerate(capture.bands):
                measurements = capture.measures[i*2:i*2+2]
                save(folder/f'band{i+1}_raw.png', band['image'])
                save(folder/f'band{i+1}_mask.png', band['filtered'])
                save(folder/f'band{i+1}_raw_mask.png', band['raw'])
                overlay = band['image'].copy()
                overlay[band['filtered']>0] = (overlay[band['filtered']>0]*.45+np.array([0,215,255])*.55).astype(np.uint8)
                for m in measurements:
                    overlay = marked(overlay,m['pairs'],m['center'])
                save(folder/f'band{i+1}_overlay.jpg', overlay)
                record['bands'].append({'counts':[len(m['pairs']) for m in measurements], 'measurements':measurements})
            record['seconds'] = time.perf_counter()-started
            write_json(folder/'details.json',record)
            white.append(record)
            print('WHITE', camera,path.name,'points',len(boxes),'inferred',len(response.InferredSlots),'classes',record['class_counts'],'counts',[b['counts'] for b in record['bands']], flush=True)
    settings = load_rim_settings(config_path)
    for camera in (1,2):
        references = [r for r in white if r['camera']==camera and r['response']['CropBox']]
        for path in sorted((source_root/f'uv_cam{camera}').glob('*.bmp')):
            capture.reset()
            image = load(path,camera,app_config)
            folder = output/f'uv_cam{camera}'/path.stem
            save(folder/'normalized.jpg',image)
            ranked = sorted(references, key=lambda r:abs((stamp(path)-stamp(Path(r['source']))).total_seconds()))
            if not ranked:
                raise ValueError(f'No usable White geometry for camera {camera}')
            reference = ranked[0]
            white_response = responses[reference['source']]
            values = [getattr(white_response,k) for k in ('CropBox','UvBox1','UvBox2','Mid1','Mid2')]
            response = service.check_disk_uv(image.copy(),*values)
            save(folder/'result.png',decode(response.ResImg))
            record = {'camera':camera,'source':str(path),'folder':folder.relative_to(output).as_posix(),
                      'white_reference':reference['folder'],'pair_verified':False,
                      'time_delta_uv_minus_white_seconds':(stamp(path)-stamp(Path(reference['source']))).total_seconds(),
                      'response':response.model_dump(exclude={'ResImg'}),'bands':[]}
            mids = [ast.literal_eval(v) for v in values[-2:]]
            for i,band in enumerate(capture.uv_bands):
                # UV uses measure_debug, rather than measure_with_params; recover exact params here.
                kwargs = dict(length_rate=.95, max_edge_distance=50,min_edge_distance=5,thickness_list=[1,3]) if i==0 else dict(length_rate=service.caliper.length_rate,max_edge_distance=service.caliper.pair_max_gap,min_edge_distance=service.caliper.min_edge_distance,thickness_list=service.caliper.thickness_list)
                center = (band['image'].shape[1]//2,band['image'].shape[0]//2)
                measurement = service.get_caliper_result_debug(band['filtered'],center,**kwargs)
                pairs = measurement['pairs']
                xs = [float((p['e1']['point'][0]+p['e2']['point'][0])/2) for p in pairs]
                white_x = np.array(mids[i])[:,0] if mids[i] else np.array([])
                mapping = [int(np.argmin(np.abs(white_x-x)))+1 for x in xs] if len(white_x) else []
                save(folder/f'band{i+1}_raw.png',band['image'])
                save(folder/f'band{i+1}_mask.png',band['filtered'])
                save(folder/f'band{i+1}_raw_mask.png',band['raw'])
                save(folder/f'band{i+1}_overlay.jpg',marked(band['image'],pairs,center))
                rim, rim_images = count_uv_rims(band['image'],i+1,settings)
                for key,value in rim_images.items():
                    save(folder/f'band{i+1}_experimental_{key}.png',value)
                write_json(folder/f'band{i+1}_experimental_rims.json',rim)
                record['bands'].append({'white_count':len(mids[i]),'uv_count':len(pairs),'uv_x':xs,
                    'nearest_white_indices':mapping, 'duplicate_display_indices':{str(k):v for k,v in Counter(mapping).items() if v>1},
                    'caliper_params':kwargs,'measurement':measurement,'experimental_rim_count':rim['count'],
                    'experimental_uncertain_count':rim['uncertain_count']})
            record['alternative_white_references'] = []
            # Same UV signal and counts, alternative White counts: isolate reference sensitivity.
            for alt in ranked[:3]:
                counts = [len(ast.literal_eval(alt['response'][k])) for k in ('Mid1','Mid2')]
                record['alternative_white_references'].append({'white':alt['folder'], 'seconds':(stamp(path)-stamp(Path(alt['source']))).total_seconds(), 'white_counts':counts,
                    'count_policy_with_fixed_uv_counts':sum(b['uv_count'] for b in record['bands'])==0 or counts==[b['uv_count'] for b in record['bands']],
                    'note':'Count-only sensitivity; geometry NOT rerun with this alternative'})
            write_json(folder/'details.json',record)
            uv.append(record)
            print('UV',camera,path.name,'white',[b['white_count'] for b in record['bands']],'uv',[b['uv_count'] for b in record['bands']],'rim',[b['experimental_rim_count'] for b in record['bands']],'pass',response.Result,flush=True)
    after = {str(p):digest(p) for p in protected+sources}
    report = {'input':str(source_root),'runtime':str(runtime),'app_config':str(app_config),'params':params.model_dump(),
              'pairing':'Nearest timestamp within same camera; unverified, may be a different tray. UV verdicts conditional on this geometry.',
              'ground_truth_verified':False,'hardware_calls':False,'production_files_modified':before!=after,
              'hashes_before':before,'hashes_after':after,'white':white,'uv':uv,
              'summary':{'white_images':len(white),'uv_images':len(uv),'white_results':dict(Counter(r['response']['Result'] for r in white)),
                         'white_class_counts':dict(sum((Counter(r['class_counts']) for r in white),Counter())),
                         'white_images_with_raw_missing_points':sum(r['raw_point_count']<75 for r in white),
                         'uv_conditional_failures':sum(not r['response']['Result'] for r in uv)}}
    write_json(output/'report.json',report)
    render(output,report)
    assert before==after,'Protected files changed'


def render(output, report):
    esc = html.escape
    parts = ['''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra ANH_18_10</title>
<style>body{background:#141b24;color:#e6edf3;font:16px system-ui;margin:24px;max-width:1500px}a{color:#8bcdff}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #506174;padding:8px}img{max-width:100%;height:auto}.note{padding:16px;background:#493a20}details{background:#202c39;padding:12px;margin:12px 0}summary{cursor:pointer}figure{margin:14px 0}.faces{display:grid;grid-template-columns:1fr 1fr;gap:12px}.bad{color:#ff9595}</style>
<h1>Ảnh thực tế 18/10 — White / UV riêng từng ảnh</h1>
<p class="note">Chạy source pipeline với config/models trong bộ cài hiện tại. Xoay theo cấu hình APP, JPEG bằng OpenCV (không khẳng định giống từng byte với Emgu).
Chưa có nhãn ground truth: NG là dự đoán model, chưa xác nhận NG ảo. UV cần hình học White; ghép ảnh gần giờ cùng camera chỉ là GIẢ ĐỊNH, không xác nhận cùng khay.
Số rim tracking là thử nghiệm, chưa dùng cho kết luận production. CountUvDisk trả về 0 khi PASS theo quy ước UI; bảng ghi số đo thô thực tế.</p>''']
    parts.append('<p><a href="report.json">JSON đầy đủ / hash / tham số</a></p><pre>'+esc(json.dumps(report['summary'],ensure_ascii=False,indent=2))+'</pre>')
    parts.append('<h2>White</h2><table><tr><th>Ảnh</th><th>Result</th><th>Point raw / suy diễn</th><th>Classify</th><th>Đếm 4 scan</th></tr>')
    for r in report['white']:
        f=r['folder']
        parts.append(f'<tr><td><a href="#{f}">{f}</a></td><td>{r["response"]["Result"]} / {r["response"]["ErrorCode"]}</td><td>{r["raw_point_count"]} / {len(r["response"]["InferredSlots"])}</td><td>{esc(str(r["class_counts"]))}</td><td>{[b["counts"] for b in r["bands"]]}</td></tr>')
    parts.append('</table><h2>UV — kết quả phụ thuộc ảnh White tham chiếu</h2><table><tr><th>Ảnh</th><th>White tham chiếu / lệch giây</th><th>White / UV từng band</th><th>Result</th><th>Rim thử nghiệm</th></tr>')
    for r in report['uv']:
        f=r['folder']
        parts.append(f'<tr><td><a href="#{f}">{f}</a></td><td>{r["white_reference"]} / {r["time_delta_uv_minus_white_seconds"]:.1f}</td><td>{[(b["white_count"],b["uv_count"]) for b in r["bands"]]}</td><td>{r["response"]["Result"]}</td><td>{[(b["experimental_rim_count"],b["experimental_uncertain_count"]) for b in r["bands"]]}</td></tr>')
    parts.append('</table>')
    for kind in ('white','uv'):
        for r in report[kind]:
            f=r['folder']
            parts.append(f'<details id="{f}"><summary>{f} — {esc(str(r["response"].get("ErrorCode")))}</summary><p><a href="{f}/normalized.jpg">Ảnh xoay</a> · <a href="{f}/result.png">Kết quả pipeline</a> · <a href="{f}/details.json">Chi tiết</a></p>')
            if kind=='white':
                parts.append(f'<p><a href="{f}/points.jpg">Point detect: xanh = raw, vàng = suy diễn</a></p><div class="faces">')
                for face in ('R1_bottom','R2_top','R2_bottom','R3_top'):
                    if r['classifications']:
                        parts.append(f'<figure><figcaption>{face}, slot 1–25 từ trái sang phải</figcaption><img loading="lazy" src="{f}/{face}.jpg"></figure>')
                parts.append('</div><p>Crop NG/no_disk (click ảnh để xem):</p>')
                for c in r['classifications']:
                    if c['label']!='ok':
                        parts.append(f'<a href="{f}/{c["crop"]}">{c["face"]} S{c["slot"]}: {c["label"]} {c["confidence"]:.3f}</a> · ')
            else:
                parts.append(f'<p>White reference: {r["white_reference"]}; Δt={r["time_delta_uv_minus_white_seconds"]:.1f}s, chưa xác nhận cùng khay.</p>')
            for i,b in enumerate(r['bands'],1):
                parts.append(f'<figure><figcaption>Band {i}: '+(str(b['counts']) if kind=='white' else f'White {b["white_count"]} / UV {b["uv_count"]}; số thứ tự ảnh pipeline bị lặp: {b["duplicate_display_indices"]}')+f'</figcaption><a href="{f}/band{i}_raw.png">Gốc</a> · <a href="{f}/band{i}_mask.png">Mask</a><img loading="lazy" src="{f}/band{i}_overlay.jpg"></figure>')
                if kind=='uv':
                    parts.append(f'<details><summary>Rim tracking thử nghiệm: {b["experimental_rim_count"]} + {b["experimental_uncertain_count"]} chưa chắc</summary><img loading="lazy" src="{f}/band{i}_experimental_overlay.png"></details>')
            parts.append('</details>')
    parts.append('</html>')
    (output/'index.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':
    main()
