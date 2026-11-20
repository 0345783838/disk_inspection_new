"""Inventory every factory image and capture unreviewed masks through real White runtime.

This preparation does not train, assign truth from inspection results, or change
runtime/canonical datasets. All White sources remain accounted for, including
model-correct images, aliases and geometry failures.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)), encoding='utf-8')
    os.replace(temporary, path)


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Smoke in a separate root only; never silently limit a full run')
    args = parser.parse_args()
    source_root, run = args.source.resolve(), args.root.resolve()
    if not source_root.is_relative_to(PROJECT) or not run.is_relative_to(PROJECT):
        raise ValueError('Source and output must be inside the project')
    if run.exists():
        raise FileExistsError('Use a new root; existing proposals/labels are never overwritten: ' + str(run))
    review = run / 'review'
    assets = review / 'assets'
    assets.mkdir(parents=True)
    (run / 'code').mkdir()
    shutil.copy2(__file__, run / 'code' / Path(__file__).name)
    old_review = PROJECT / '.work/segmentation_supplementary_20261023'
    shutil.copy2(old_review / 'serve_review.py', run / 'serve_review.py')
    html = (old_review / 'review/index.html').read_text(encoding='utf-8').replace('ANH_18_10', 'Nhà máy · NEW_IMAGES_2')
    # Reuse the reviewed editor's exact drawing/save semantics; add navigation only.
    html = html.replace('<div id="list"></div>', '<select id="sourceFilter"><option value="all">Mọi đợt ảnh</option></select><select id="bandFilter"><option value="all">Cả hai band</option><option>band1</option><option>band2</option></select><a href="overview.html">Gallery toàn bộ ảnh / lỗi geometry</a><div id="list"></div>')
    html = html.replace("const r=state.bands[v.id],s=r?.status||'pending';", "const r=state.bands[v.id],s=r?.status||'pending';if($('sourceFilter').value!=='all'&&$('sourceFilter').value!==v.source_batch)return;if($('bandFilter').value!=='all'&&$('bandFilter').value!==v.band)return;")
    html = html.replace("$('statusFilter').onchange=list;", "$('statusFilter').onchange=list;$('sourceFilter').onchange=list;$('bandFilter').onchange=list;")
    html = html.replace("payload=await p.json();state=await s.json();", "payload=await p.json();state=await s.json();for(const batch of [...new Set(payload.items.map(v=>v.source_batch))]){const o=document.createElement('option');o.value=o.textContent=batch;$('sourceFilter').append(o)}")
    # A gallery can link straight to any band, including model-correct cases.
    html = html.replace("$('saveState').textContent='Lưu tự động sau mỗi thay đổi';show()", "const target=new URLSearchParams(location.search).get('item');if(target){const n=payload.items.findIndex(v=>v.id===target);if(n>=0)current=n}$('saveState').textContent='Lưu tự động sau mỗi thay đổi';show()")
    (review / 'index.html').write_text(html, encoding='utf-8')
    service_root = PROJECT / 'SERVICE'
    config_path = service_root / 'config/config.env'
    app_config = PROJECT / 'APP/DiskInspection/bin/Debug/DiskInspection.exe.config'
    protected_paths = [config_path, app_config, PROJECT / 'DATASET_CLASSIFIER/manifest.json',
                       PROJECT / '.work/ACTIVE_CLASSIFIER_DATASET.json']
    protected_paths += list((service_root / 'src').rglob('*.py'))
    protected_paths += [p for p in (service_root / 'config/models').iterdir() if p.is_file()]
    protected = {str(p): sha(p) for p in protected_paths}
    write(run / 'protected_before.json', protected)
    write(run / 'status.json', {'phase': 'inventory', 'training_started': False})
    pattern = re.compile(r'^(\d{8})_(\d{6})_(\d+)_Cam([12])_(White|UV)_(Raw|Result)$')
    inventory = []
    for path in sorted(p for p in source_root.rglob('*') if p.is_file()):
        match = pattern.fullmatch(path.stem)
        row = {'path': str(path), 'relative_path': path.relative_to(source_root).as_posix(),
               'bytes': path.stat().st_size, 'sha256': sha(path), 'parsed': bool(match)}
        if match:
            date, clock, fraction, camera, light, role = match.groups()
            row.update(camera='Cam' + camera, light=light, role=role,
                       timestamp=datetime.strptime(date + clock, '%d%m%Y%H%M%S').isoformat() + '.' + fraction,
                       capture_id=date + '_' + clock + '_' + fraction,
                       source_batch=path.relative_to(source_root).parts[0])
        inventory.append(row)
    write(run / 'inventory.json', {'source': str(source_root), 'files': inventory,
                                   'unparsed': [r['path'] for r in inventory if not r['parsed']]})
    whites = [r for r in inventory if r.get('light') == 'White' and r.get('role') == 'Raw']
    if args.limit:
        whites = whites[:args.limit]
    if not whites:
        raise ValueError('No White Raw files found')
    # Group both cameras from the same capture, nearby captures, and exact decoded
    # aliases. These are provisional leakage constraints, not known tray identities.
    captures = sorted({r['capture_id']: r for r in whites}.values(), key=lambda r: r['timestamp'])
    parent = {r['capture_id']: r['capture_id'] for r in captures}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        parent[find(b)] = find(a)

    for a, b in zip(captures, captures[1:]):
        ta, tb = datetime.fromisoformat(a['timestamp'][:26]), datetime.fromisoformat(b['timestamp'][:26])
        if (tb - ta).total_seconds() <= 120:
            join(a['capture_id'], b['capture_id'])
    decoded_lookup, thumbs = {}, {}
    for row in whites:
        image = cv2.imread(row['path'])
        if image is None:
            raise ValueError('Cannot decode ' + row['path'])
        row['shape_hw'] = list(image.shape[:2])
        row['decoded_sha256'] = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        key = (row['camera'], row['decoded_sha256'])
        if key in decoded_lookup:
            join(row['capture_id'], decoded_lookup[key]['capture_id'])
        else:
            decoded_lookup[key] = row
        thumbs[row['path']] = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 64))
    for row in whites:
        row['capture_group'] = find(row['capture_id'])
    near = []
    for n, a in enumerate(whites):
        for b in whites[n + 1:]:
            if a['camera'] != b['camera'] or a['decoded_sha256'] == b['decoded_sha256']:
                continue
            mae = float(np.abs(thumbs[a['path']].astype(float) - thumbs[b['path']]).mean())
            if mae <= 3:
                near.append({'a': a['path'], 'b': b['path'], 'thumbnail_mae': mae,
                             'same_provisional_group': a['capture_group'] == b['capture_group'],
                             'status': 'appearance_candidate_not_confirmed_same_tray'})
    write(run / 'grouping_draft.json', {'status': 'no_split_assigned_pending_geometry_labels_and_tray_review',
        'white_sources': whites, 'capture_groups': len({r['capture_group'] for r in whites}),
        'near_duplicate_candidates': near, 'policy': 'Paired cameras + <=120s adjacent captures + exact decoded aliases. No claim of independent tray identity. All bands/tiles/augmentations from a family must stay together.'})
    sys.path[:0] = [str(service_root), str(service_root / 'training')]
    import decouple
    decouple.config = decouple.Config(decouple.RepositoryEnv(str(config_path)))
    os.chdir(service_root)
    from smoke_local_white import normalize
    from src.communication.modbus_client import MbClient

    def forbidden_hardware(*a, **kw):
        raise AssertionError('Hardware calls are forbidden during image preparation')

    for method in ['connect', 'disconnect', 'check_connection', '_MbClient__write_bit', '_MbClient__read_bit']:
        setattr(MbClient, method, forbidden_hardware)
    from src.service.check_disk_service_yolo import DiskCheckingService
    service = DiskCheckingService()
    segmentor = service.disk_segmentor_yolo
    assert (segmentor.preprocess_mode, segmentor.rgb_input, segmentor.mask_decode, segmentor.tile_merge) == ('letterbox', True, 'native', 'core')
    model = (service_root / decouple.config('DISK_SEGMENT_MODEL_PATH')).resolve()
    model_hash = sha(model)
    context, bands = {}, []
    original_rectify, original_crop = service.full_rectify_pipeline, service.crop_between_rows
    original_segment, original_tile = segmentor.segment_large_image_debug, segmentor.segment_objects_debug

    def rectify(*a, **kw):
        result = original_rectify(*a, **kw)
        context.update(rectified=result[0].copy(), homography=result[1].tolist(), quad=result[3].tolist())
        return result

    def crop(*a, **kw):
        result = original_crop(*a, **kw)
        context.setdefault('crop_boxes', []).append(result[1])
        return result

    def segment(image, conf, iou):
        calls = []

        def tile(*a, **kw):
            output = original_tile(*a, **kw)
            calls.append(output)
            return output

        segmentor.segment_objects_debug = tile
        try:
            result = original_segment(image, conf, iou)
        finally:
            segmentor.segment_objects_debug = original_tile
        h, w = image.shape[:2]
        _, positions = segmentor.split_image(image)
        assert len(calls) == len(positions)
        union = np.zeros((h, w), np.uint8)
        instances, exact_masks = [], []
        for ti, (output, (x1, x2)) in enumerate(zip(calls, positions)):
            _, scores, _, masks = output
            for pi, (mask, score) in enumerate(zip(masks, scores)):
                binary = cv2.resize(mask.astype(np.uint8), (x2 - x1, h), interpolation=cv2.INTER_NEAREST) * 255
                placed = np.zeros((h, w), np.uint8)
                segmentor.paste_tile_mask(placed, binary, x1, x2, ti)
                union = np.maximum(union, placed)
                if not np.any(placed):
                    continue
                # Preserve holes as opposite-winding subpaths in Canvas2D.
                contours, _ = cv2.findContours(placed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                parts = []
                for contour in contours:
                    if len(contour) >= 3 and cv2.contourArea(contour) > 0:
                        parts.append(contour.reshape(-1, 2).tolist())
                    else:
                        # Keep zero-area one-pixel core-boundary slivers. This
                        # fractional envelope preserves the original pixel centres.
                        component = np.zeros_like(placed)
                        cv2.drawContours(component, [contour], -1, 1, 1)
                        expanded = cv2.resize(component, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
                        outline, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for shape in outline:
                            points = shape.reshape(-1, 2)
                            envelope = points // 8 + .1 + .3 * (points % 8) / 7
                            envelope[:, 0] = np.clip(envelope[:, 0], 0, w - 1)
                            envelope[:, 1] = np.clip(envelope[:, 1], 0, h - 1)
                            parts.append(envelope.tolist())
                reconstructed = np.zeros_like(placed)
                cv2.fillPoly(reconstructed, [np.rint(p).astype(np.int32) for p in parts], 255)
                overlap = np.count_nonzero((placed > 0) & (reconstructed > 0)) / max(1, np.count_nonzero((placed > 0) | (reconstructed > 0)))
                instances.append({'id': f't{ti}p{pi}', 'kind': 'polygon', 'parts': parts,
                    'score': float(score), 'tile': ti, 'model_proposal': True,
                    'mask_pixels': int(np.count_nonzero(placed)), 'polygon_pixel_iou': overlap,
                    'center_x': float(np.where(placed)[1].mean())})
                exact_masks.append(placed > 0)
        assert np.array_equal(union, result[0]), 'Proposal capture changed runtime mask'
        order = sorted(range(len(instances)), key=lambda n: instances[n]['center_x'])
        bands.append({'image': image.copy(), 'mask': result[0].copy(),
                      'instances': [instances[n] for n in order],
                      'exact_masks': [exact_masks[n] for n in order], 'positions': positions})
        return result

    service.full_rectify_pipeline, service.crop_between_rows = rectify, crop
    segmentor.segment_large_image_debug = segment
    frames, items, failures, seen = [], [], [], {}
    for n, row in enumerate(whites, 1):
        started = time.perf_counter()
        key = (row['camera'], row['decoded_sha256'])
        if key in seen:
            frame = seen[key]
            frame['sources'].append(row['path'])
            frame['capture_ids'].append(row['capture_id'])
            for item in items:
                if item['frame_id'] == frame['id']:
                    item['sources'].append(row['path'])
            continue
        camera = int(row['camera'][-1])
        frame_id = row['camera'] + '_' + row['decoded_sha256'][:16]
        frame = {**row, 'id': frame_id, 'sources': [row['path']], 'capture_ids': [row['capture_id']], 'bands': []}
        context.clear()
        bands.clear()
        raw = cv2.imread(row['path'])
        normalized = normalize(raw, camera, app_config)
        ok, encoded = cv2.imencode('.jpg', normalized)
        assert ok
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        frame['normalization'] = 'Raw camera rotated/ROI once per current Debug APP config; OpenCV default JPEG roundtrip, not claimed Emgu-byte-identical.'
        normalized_path = assets / (frame_id + '_normalized.png')
        save_image(normalized_path, image)
        frame['normalized'] = 'assets/' + normalized_path.name
        field_result = Path(row['path'].replace('_Raw.png', '_Result.jpg'))
        if field_result.exists():
            field_image = cv2.imread(str(field_result))
            field_image = cv2.resize(field_image, (700, round(field_image.shape[0] * 700 / field_image.shape[1])))
            save_image(assets / (frame_id + '_field_result.jpg'), field_image)
            frame['field_result'] = f'assets/{frame_id}_field_result.jpg'
        try:
            response = service.check_disk_white(image.copy())
            value = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
            rendered = value.pop('ResImg', None)
            frame.update(response=value, geometry_status=response.PointGeometryStatus,
                         inferred_slots=response.InferredSlots)
            if rendered:
                rendered_image = cv2.imdecode(np.frombuffer(base64.b64decode(rendered), np.uint8), cv2.IMREAD_COLOR)
                rendered_image = cv2.resize(rendered_image, (700, round(rendered_image.shape[0] * 700 / rendered_image.shape[1])))
                save_image(assets / (frame_id + '_runtime_result.jpg'), rendered_image)
                frame['runtime_result'] = f'assets/{frame_id}_runtime_result.jpg'
            if len(bands) != 2:
                raise ValueError(f'Runtime extracted {len(bands)} bands; geometry requires review')
            rectified = context['rectified'].copy()
            for bi, box in enumerate(context['crop_boxes'], 1):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(rectified, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
                cv2.putText(rectified, f'band{bi}', (x1 + 10, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
            save_image(assets / (frame_id + '_context.jpg'), rectified)
            for bi, band in enumerate(bands, 1):
                identifier = frame_id + f'_band{bi}'
                image_path, mask_path = assets / (identifier + '.png'), assets / (identifier + '_model_mask.png')
                save_image(image_path, band['image'])
                save_image(mask_path, band['mask'])
                exact_path = assets / (identifier + '_exact_proposals.npz')
                h, w = band['mask'].shape
                np.savez_compressed(exact_path, masks=np.asarray(band['exact_masks'], np.uint8) if band['exact_masks'] else np.empty((0, h, w), np.uint8))
                overlay = band['image'].copy()
                overlay[band['mask'] > 0] = (0, 220, 255)
                display = cv2.addWeighted(band['image'], .5, overlay, .5, 0)
                stack = np.vstack([band['image'], display])
                save_image(assets / (identifier + '_preview.jpg'), cv2.resize(stack, (1000, round(stack.shape[0] * 1000 / stack.shape[1]))))
                item = {'id': identifier, 'frame_id': frame_id, 'camera': row['camera'], 'band': f'band{bi}',
                    'sources': frame['sources'].copy(), 'source_sha256': row['sha256'], 'source_batch': row['source_batch'],
                    'capture_group': row['capture_group'], 'capture_id': row['capture_id'], 'filename': Path(row['path']).name,
                    'width': w, 'height': h, 'image': 'assets/' + image_path.name, 'image_sha256': sha(image_path),
                    'model_mask': 'assets/' + mask_path.name, 'model_mask_sha256': sha(mask_path),
                    'exact_proposals': 'assets/' + exact_path.name, 'exact_proposals_sha256': sha(exact_path),
                    'preview': f'assets/{identifier}_preview.jpg', 'context': f'assets/{frame_id}_context.jpg',
                    'normalized': frame['normalized'], 'geometry_status': frame['geometry_status'],
                    'inferred_slots': frame['inferred_slots'], 'band_xyxy': context['crop_boxes'][bi - 1],
                    'homography': context['homography'], 'instances': copy.deepcopy(band['instances']),
                    'tile_positions': band['positions'], 'label_status': 'model_proposal_unreviewed',
                    'runtime_mask_capture_exact': True, 'split': 'unassigned'}
                items.append(item)
                frame['bands'].append(identifier)
            frame['status'] = 'proposals_unreviewed'
        except Exception as error:
            frame.update(status='geometry_or_runtime_failure_retained', failure=str(error))
            failures.append(frame)
        frame['seconds'] = time.perf_counter() - started
        frames.append(frame)
        seen[key] = frame
        write(run / 'status.json', {'phase': 'proposals', 'processed': n, 'white_sources': len(whites),
            'unique_frames': len(frames), 'bands': len(items), 'failures': len(failures), 'training_started': False})
        write(run / 'frames_checkpoint.json', frames)
        print(f'{n}/{len(whites)} {row["camera"]} {Path(row["path"]).name} {frame["status"]} bands={len(frame["bands"])} {frame["seconds"]:.2f}s', flush=True)
    after = {str(p): sha(p) for p in protected_paths}
    source_after = {r['path']: sha(r['path']) for r in inventory}
    assert after == protected, 'Protected runtime/canonical/source files changed during preparation'
    assert all(source_after[r['path']] == r['sha256'] for r in inventory), 'Factory input changed'
    write(run / 'protected_after.json', after)
    payload = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'awaiting_human_review_no_training', 'source_files': len(whites), 'unique_frames': len(frames),
        'duplicate_files': len(whites) - len(frames), 'model_path': str(model), 'model_sha256': model_hash,
        'scope': 'Every new White Raw included, irrespective of model result. White-only segmentation; UV/results are provenance/context, never pixel labels. Native 512 RGB/core conf0.5 NMS0.8. Predictions and geometry are unreviewed; no fixed disk count or extrapolation. No classifier edits, split assignment, training or deployment.',
        'python_executable': sys.executable, 'opencv_version': cv2.__version__,
        'hardware_blocked': True, 'frames': frames, 'geometry_failures': failures, 'items': items}
    write(review / 'items.json', payload)
    (review / 'overview.html').write_text(OVERVIEW, encoding='utf-8')
    summary = {'all_factory_files': len(inventory), 'white_sources': len(whites), 'capture_sets': len(captures),
        'unique_frames': len(frames), 'bands': len(items), 'tile_proposals_for_later_export': sum(len(i['tile_positions']) for i in items),
        'duplicate_files': len(whites) - len(frames), 'geometry_failures': len(failures),
        'source_accounting_complete': sum(len(f['sources']) for f in frames) == len(whites),
        'by_camera': dict(Counter(f['camera'] for f in frames)),
        'by_batch': dict(Counter(r['source_batch'] for r in whites)),
        'model_result_not_used_for_exclusion': True, 'protected_unchanged': True,
        'reviewed_bands': 0, 'training_started': False, 'split_assigned': False,
        'items_sha256': sha(review / 'items.json')}
    write(review / 'proposal_summary.json', summary)
    write(run / 'status.json', {'phase': 'awaiting_label_and_geometry_review', **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


OVERVIEW = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Segmentation · Toàn bộ ảnh nhà máy mới</title><style>
*{box-sizing:border-box}body{margin:0;background:#0c1320;color:#e5edf8;font:15px system-ui}header{padding:24px 4vw;background:#142135;position:sticky;top:0;z-index:1;border-bottom:1px solid #35465e}h1{font-size:23px;margin:0 0 8px}p{line-height:1.6;margin:8px 0}.muted{color:#afbed1}a{color:#86d9c5}button,select{background:#1d2c42;color:#fff;border:1px solid #50627b;padding:9px;border-radius:6px;font:inherit;margin:4px}main{padding:20px 4vw}article{background:#142135;border:1px solid #35465e;border-radius:10px;margin-bottom:20px;padding:18px}h2{font-size:16px;overflow-wrap:anywhere}.bands{display:grid;grid-template-columns:1fr;gap:12px}.bands img{width:100%;display:block}.tag{padding:4px 8px;background:#244538;border-radius:5px;font-size:12px}.fail{background:#572b33}.context{display:flex;gap:12px;flex-wrap:wrap}.context img{width:280px;max-width:100%}details{margin-top:12px}pre{white-space:pre-wrap;font-size:12px}#list{max-width:1500px;margin:auto}.notice{background:#3a331e;padding:12px;border-radius:6px;color:#ffe7a5}.controls{display:flex;align-items:center;flex-wrap:wrap}.bandLink{display:block;padding:8px 0}.footer{display:flex;justify-content:center;gap:12px;padding:16px}#summary{font-weight:600}</style>
<header><h1>Segmentation cạnh đĩa · NEW_IMAGES_2</h1><div id="summary">Đang tải…</div><p class="muted">Toàn bộ White của cả hai camera, kể cả ảnh model đang làm đúng. Ảnh gốc ở trên / mask model ở dưới mỗi band.</p><div class="controls"><select id="camera"><option value="all">Cả hai camera</option><option>Cam1</option><option>Cam2</option></select><select id="batch"><option value="all">Mọi đợt ảnh</option></select><select id="kind"><option value="all">Mọi kết quả geometry</option><option value="ready">Có hai band để sửa nhãn</option><option value="failure">Cần xử lý geometry / runtime</option></select><a href="index.html">Mở trình sửa mask →</a></div></header>
<main><p class="notice">Mask và kết quả máy là gợi ý chưa xác nhận, không phải nhãn thật. Màu vàng là mask thô trước lọc diện tích. Kết quả OK/NG của cả luồng không xác nhận segmentation đúng. Chưa train hoặc thay model.</p><div id="list"></div><div class="footer"><button id="prev">← Trước</button><span id="page"></span><button id="next">Tiếp →</button></div></main><script>
let data,page=0;const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function show(){const rows=data.frames.filter(f=>($('camera').value==='all'||f.camera===$('camera').value)&&($('batch').value==='all'||f.source_batch===$('batch').value)&&($('kind').value==='all'||($('kind').value==='ready'?(f.bands.length===2):(f.bands.length!==2))));const pages=Math.max(1,Math.ceil(rows.length/8));page=Math.min(page,pages-1);$('page').textContent=`Trang ${page+1}/${pages} · ${rows.length} ảnh`;$('prev').disabled=page===0;$('next').disabled=page===pages-1;const items=Object.fromEntries(data.items.map(i=>[i.id,i]));$('list').innerHTML=rows.slice(page*8,page*8+8).map(f=>`<article><h2>${esc(f.camera)} · ${esc(f.source_batch)} · ${esc(f.filename||f.relative_path)}</h2><span class="tag ${f.bands.length!==2?'fail':''}">${esc(f.geometry_status||f.failure||f.status)}</span><p class="muted">${esc(f.timestamp)} · ${f.sources.length} file nguồn · nhóm tạm ${esc(f.capture_group)}</p><div class="bands">${f.bands.map(id=>{const i=items[id];return `<div><a class="bandLink" href="index.html?item=${encodeURIComponent(id)}">${esc(i.band)} · ${i.instances.length} gợi ý / mảnh mask · Mở sửa nhãn →</a><a href="index.html?item=${encodeURIComponent(id)}"><img loading="lazy" src="${esc(i.preview)}" alt="Ảnh band gốc và mask model"></a></div>`}).join('')}</div><details><summary>Xem ảnh camera, ảnh kết quả tại nhà máy và replay hiện tại</summary><div class="context">${[['normalized','White sau chuẩn hóa'],['field_result','Kết quả đã lưu tại nhà máy'],['runtime_result','Replay bằng SERVICE hiện tại']].filter(([k])=>f[k]).map(([k,t])=>`<a href="${esc(f[k])}" target="_blank"><p>${t}</p><img loading="lazy" src="${esc(f[k])}" alt="${t}"></a>`).join('')}</div><pre>${esc(JSON.stringify({sources:f.sources,response:f.response,failure:f.failure},null,2))}</pre></details></article>`).join('')||'<p>Không có ảnh phù hợp.</p>'}
for(const id of ['camera','batch','kind'])$(id).onchange=()=>{page=0;show()};$('prev').onclick=()=>{page--;show();scrollTo(0,0)};$('next').onclick=()=>{page++;show();scrollTo(0,0)};(async()=>{data=await(await fetch('/api/items')).json();for(const b of [...new Set(data.frames.map(f=>f.source_batch))]){const o=document.createElement('option');o.value=o.textContent=b;$('batch').append(o)}$('summary').textContent=`${data.source_files} White nguồn · ${data.unique_frames} ảnh khác nhau · ${data.items.length} band · ${data.geometry_failures.length} ảnh cần xử lý geometry/runtime`;show()})().catch(e=>{$('summary').textContent=e.message});
</script></html>'''


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
