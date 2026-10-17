"""Export classifier inputs by observing the real White inspection flow.

No production files or parameters are changed. The segmentation call is stopped
after classification. Frames rejected by the existing point-count guard are
kept separate: their optional diagnostic crops are NOT production inputs.
"""
import argparse
import hashlib
import html
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'SERVICE'))
from src.service.check_disk_service_yolo import DiskCheckingService
from src.tools.normalize_factory_frames import IMAGE_EXTENSIONS, parse_roi

REGIONS = ('row1_bottom', 'row2_top', 'row2_bottom', 'row3_top')
COLORS = ((70, 210, 70), (255, 180, 40), (60, 180, 255), (220, 80, 220))


class ExportComplete(Exception):
    pass


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(str(path))
    # Native crops must remain pixel-exact, including the top-face 180 rotation.
    assert np.array_equal(cv2.imread(str(path)), image), str(path)


def normalize(image, camera, settings):
    rotation = settings.get(f'Cam{camera}FrameRotation',
                            'Clockwise90' if camera == 1 else 'CounterClockwise90')
    modes = {'Clockwise90': cv2.ROTATE_90_CLOCKWISE,
             'CounterClockwise90': cv2.ROTATE_90_COUNTERCLOCKWISE,
             'Rotate180': cv2.ROTATE_180}
    if rotation == 'None':
        result = image.copy()
    else:
        result = cv2.rotate(image, modes[rotation])
    roi = parse_roi(settings.get(f'Cam{camera}NormalizedRoi', ''))
    if roi:
        x, y, w, h = roi
        if x + w > result.shape[1] or y + h > result.shape[0]:
            raise ValueError('Configured ROI exceeds image')
        result = result[y:y+h, x:x+w].copy()
    return result, {'rotation': rotation, 'roi': roi}


def tile_sheet(crops, records):
    columns, cell_w, cell_h = 8, 160, 210
    canvas = np.full((((len(crops)+columns-1)//columns)*cell_h,
                      columns*cell_w, 3), 24, np.uint8)
    for i, (crop, record) in enumerate(zip(crops, records)):
        x, y = (i % columns)*cell_w, (i // columns)*cell_h
        scale = min(144/crop.shape[1], 162/crop.shape[0])
        preview = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)),
                                   max(1, round(crop.shape[0]*scale))))
        px, py = x+(cell_w-preview.shape[1])//2, y+25
        canvas[py:py+preview.shape[0], px:px+preview.shape[1]] = preview
        cv2.putText(canvas, f"#{i+1:02d} {crop.shape[1]}x{crop.shape[0]}",
                    (x+5, y+18), cv2.FONT_HERSHEY_SIMPLEX, .45, (235,235,235), 1)
        label = record['label'] or 'diagnostic only'
        if record['confidence'] is not None:
            label += f" {record['confidence']:.3f}"
        cv2.putText(canvas, label, (x+5, y+201), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (130,220,255), 1)
    return canvas


def export_frame(service, image, target, diagnostic, state):
    original_detect = service.disk_point_detect_model.detect_objects_debug
    original_rectify = service.full_rectify_pipeline
    original_rects = service.get_line_boxes_ratio_shift
    original_crops = service.crop_boxes
    original_predict = service.point_classification_model.predict_batch
    original_segment = service.disk_segmentor_yolo.segment_large_image_debug
    groups, detected, geometry, pending = [], {}, {}, {}

    def capture_detect(*args, **kwargs):
        result = original_detect(*args, **kwargs)
        detected['boxes'] = result[0].copy()
        return result

    def capture_rectify(*args, **kwargs):
        result = original_rectify(*args, **kwargs)
        geometry.update(image=result[0].copy(), matrix=result[1].copy(), quad=result[3].copy())
        return result

    def capture_rects(*args, **kwargs):
        result = original_rects(*args, **kwargs)
        pending.update(rects=result, points=np.asarray(args[1]).copy(), direction=args[2])
        return result

    def capture_crops(*args, **kwargs):
        result = original_crops(*args, **kwargs)
        pending['crops'] = [crop.copy() for crop in result]
        return result

    def capture_predict(crops):
        assert all(np.array_equal(a,b) for a,b in zip(crops,pending['crops']))
        labels, confidences = original_predict(crops)
        if labels is None or confidences is None:
            raise RuntimeError('Classifier failed')
        groups.append(dict(pending, labels=labels, confidences=confidences))
        return labels, confidences

    def stop_segmentation(*args, **kwargs):
        raise ExportComplete()

    service.disk_point_detect_model.detect_objects_debug = capture_detect
    service.full_rectify_pipeline = capture_rectify
    service.get_line_boxes_ratio_shift = capture_rects
    service.crop_boxes = capture_crops
    service.point_classification_model.predict_batch = capture_predict
    service.disk_segmentor_yolo.segment_large_image_debug = stop_segmentation
    try:
        response = service.check_disk_white(image.copy())
        state['flow_error'] = response.ErrorCode
        state['status'] = 'production_rejected_before_segmentation'
    except ExportComplete:
        assert len(groups) == 4
        state['status'] = 'production_classifier_inputs'
    finally:
        service.disk_point_detect_model.detect_objects_debug = original_detect
        service.full_rectify_pipeline = original_rectify
        service.get_line_boxes_ratio_shift = original_rects
        service.crop_boxes = original_crops
        service.point_classification_model.predict_batch = original_predict
        service.disk_segmentor_yolo.segment_large_image_debug = original_segment

    boxes = detected.get('boxes', np.empty((0,4), np.float32))
    state['detected_points'] = len(boxes)
    rows = None
    try:
        rows = service.split_rows(boxes, expected_points_per_row=service.point_slot_count)
        state['row_point_counts'] = [len(row) for row in rows]
    except ValueError:
        state['row_point_counts'] = []
    state['production_classifier_calls'] = len(groups)
    state['diagnostic_only'] = False

    if not groups and diagnostic and rows is not None:
        # Explicitly separate hypothetical geometry for frames the real guard rejects.
        result = original_rectify(image, rows[0], rows[2], expand_ratio_x=.2, expand_ratio_y=.1)
        geometry.update(image=result[0].copy(), matrix=result[1], quad=result[3])
        warped_rows = [service.update_boxes_after_warp(row.copy(), result[1]) for row in rows]
        for row, direction in ((warped_rows[0],'bottom'), (warped_rows[1],'top'),
                               (warped_rows[1],'bottom'), (warped_rows[2],'top')):
            rects = original_rects(result[0], row, direction)
            crops = original_crops(result[0], rects, direction)
            if any(crop is None or crop.size == 0 for crop in crops):
                raise ValueError('Empty diagnostic crop')
            groups.append(dict(rects=rects, points=row, direction=direction,
                               crops=crops, labels=[None]*len(crops),
                               confidences=[None]*len(crops)))
        state['diagnostic_only'] = True

    target.mkdir(parents=True, exist_ok=True)
    detected_preview = image.copy()
    for box in boxes:
        x1,y1,x2,y2 = map(int,box)
        cv2.rectangle(detected_preview, (x1,y1), (x2,y2), (0,190,255), 2)
    write_image(target/'detected_points.png', detected_preview)
    all_records = []
    if groups:
        overlay = geometry['image'].copy()
        normalized_overlay = image.copy()
        state['rectified_shape'] = list(overlay.shape)
        state['source_quad'] = geometry['quad'].tolist()
        state['warp_matrix'] = geometry['matrix'].tolist()
        inv = np.linalg.inv(geometry['matrix'])
        for group_index, group in enumerate(groups):
            region = REGIONS[group_index]
            records = []
            for index, (crop, box, point, label, confidence) in enumerate(zip(
                    group['crops'], group['rects'], group['points'],
                    group['labels'], group['confidences']), 1):
                crop_path = f'crops/{region}/{index:02d}.png'
                write_image(target/crop_path, crop)
                x1,y1,x2,y2 = map(int,box)
                effective = [min(max(x1,0),overlay.shape[1]), min(max(y1,0),overlay.shape[0]),
                             min(max(x2,0),overlay.shape[1]), min(max(y2,0),overlay.shape[0])]
                ex1,ey1,ex2,ey2 = effective
                assert crop.shape[:2] == (ey2-ey1, ex2-ex1)
                polygon = np.array([[[ex1,ey1],[ex2,ey1],[ex2,ey2],[ex1,ey2]]],np.float32)
                source_polygon = cv2.perspectiveTransform(polygon,inv)[0]
                record = dict(region=region,index=index,crop=crop_path,
                              direction=group['direction'],rotated_180=group['direction']=='top',
                              requested_box=box,effective_box=effective,point_box=point.tolist(),
                              normalized_polygon=source_polygon.tolist(),shape=list(crop.shape),
                              label=str(label) if label is not None else None,
                              confidence=float(confidence) if confidence is not None else None,
                              production_input=not state['diagnostic_only'])
                records.append(record)
                color = COLORS[group_index]
                cv2.rectangle(overlay,(ex1,ey1),(ex2-1,ey2-1),color,2)
                cv2.putText(overlay,f'{group_index+1}:{index:02d}',(ex1,max(15,ey1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
                cv2.polylines(normalized_overlay,[np.rint(source_polygon).astype(np.int32)],True,color,2)
                sx,sy = map(int,source_polygon[0])
                cv2.putText(normalized_overlay,f'{group_index+1}:{index:02d}',(sx,max(15,sy-4)),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
            write_image(target/f'{region}_sheet.png', tile_sheet(group['crops'],records))
            all_records.extend(records)
        # Draw the actual full input rectangles, not production display's 30% vertical trim.
        write_image(target/'rectified_boxes.png',overlay)
        write_image(target/'normalized_boxes.png',normalized_overlay)
    state['exported_crops'] = len(all_records)
    state['crops'] = all_records
    (target/'manifest.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    warning = ('DIAGNOSTIC ONLY: luong that dung truoc classifier; cac crop duoi day '
               'chi de xem hinh hoc, KHONG phai input da duoc classify.'
               if state['diagnostic_only'] else 'Crop chinh xac da dua vao classifier cua luong hien tai.')
    content = [f'<h1>{html.escape(target.name)}</h1><p>{warning}</p>',
               f"<p>Point rows: {state['row_point_counts']}; actual classifier calls: {state['production_classifier_calls']}</p>",
               '<p>1=row1 bottom; 2=row2 top; 3=row2 bottom; 4=row3 top. '
               'Top crops da xoay 180 do nhu input classifier. PNG goc khong resize. '
               'Nhan va confidence tren sheet la du doan model hien tai, khong phai ground truth.</p>',
               '<p><a href="manifest.json">Toa do / nhan model / crop files</a></p>']
    if groups:
        content += ['<h2>Full classify rectangles on rectified image</h2><a href="rectified_boxes.png"><img src="rectified_boxes.png"></a>',
                    '<h2>Mapped back onto normalized image</h2><a href="normalized_boxes.png"><img src="normalized_boxes.png"></a>']
        for region in REGIONS[:len(groups)]:
            content += [f'<h2>{region}</h2><a href="{region}_sheet.png"><img src="{region}_sheet.png"></a>',
                        '<div class="crops">']
            for r in (r for r in all_records if r['region']==region):
                content += [f'<a href="{r["crop"]}"><img src="{r["crop"]}" loading="lazy"><br>#{r["index"]:02d}</a>']
            content += ['</div>']
    else:
        content += ['<img src="detected_points.png">']
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'
        'body{background:#17191d;color:#eee;font:16px Arial;margin:24px}a{color:#8fd1ff}'
        'img{max-width:100%;height:auto}.crops{display:flex;flex-wrap:wrap;gap:12px}'
        '.crops a{text-align:center}.crops img{height:160px;width:auto}</style>'+''.join(content),encoding='utf-8')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ha',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--include-blocked-proposals',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing results are not overwritten.')
    active_config = ROOT/'APP/DiskInspection/bin/BroSight_Disinspector/plugin/config/config.env'
    source_config = ROOT/'SERVICE/config/config.env'
    assert active_config.read_bytes()==source_config.read_bytes(), 'Active/source config differ'
    active_models = active_config.parent/'models'
    env_values = dict(line.split('=',1) for line in source_config.read_text().splitlines() if '=' in line and not line.lstrip().startswith('#'))
    env_values = {k.strip():v.strip() for k,v in env_values.items()}
    for name in (Path(env_values['DISK_POINT_DETECT_MODEL_PATH']).name,'c_p.onnx'):
        assert hashlib.sha256((active_models/name).read_bytes()).digest()==hashlib.sha256(
            (source_config.parent/'models'/name).read_bytes()).digest(), name
    app_config = ROOT/'APP/DiskInspection/bin/BroSight_Disinspector/DiskInspection.exe.config'
    settings = {item.attrib['key']:item.attrib['value'] for item in ET.parse(app_config).findall('./appSettings/add')}
    service = DiskCheckingService()
    assert service.plc_controller.client is None
    args.output.mkdir(parents=True)
    states = []
    for camera in (1,2):
        for source in sorted((args.ha/f'Cam{camera}').iterdir()):
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            raw = cv2.imread(str(source))
            if raw is None:
                raise ValueError(str(source))
            image,normalization = normalize(raw,camera,settings)
            state = dict(source=str(source.resolve()),camera=camera,normalization=normalization,
                         normalized_shape=list(image.shape))
            # The directory is determined from the actual guard, not a fixed camera assumption.
            temporary = args.output/'frames'/f'Cam{camera}_{source.stem}'
            export_frame(service,image,temporary,args.include_blocked_proposals,state)
            state['review'] = temporary.relative_to(args.output).as_posix()+'/index.html'
            states.append(state)
            print(temporary.name,state['status'],'rows',state['row_point_counts'],
                  'actual_calls',state['production_classifier_calls'],'crops',state['exported_crops'],
                  'diagnostic_only',state['diagnostic_only'],flush=True)
    report = dict(point_threshold=service.disk_point_detect_model.conf_threshold,
                  classify_crop_width_ratio=service.classify_crop_width_ratio,
                  point_model=env_values['DISK_POINT_DETECT_MODEL_PATH'],
                  detector_roi=service.disk_point_detect_model.input_roi,
                  slot_completion=service.point_slot_completion,
                  point_iou=service.disk_point_detect_model.iou_threshold,
                  required_points_per_row=service.point_slot_count,
                  production_frames=sum(s['production_classifier_calls']>0 for s in states),
                  diagnostic_frames=sum(s['diagnostic_only'] for s in states),
                  production_crops=sum(s['exported_crops'] for s in states if not s['diagnostic_only']),
                  diagnostic_crops=sum(s['exported_crops'] for s in states if s['diagnostic_only']),
                  frames=states,PLC_connected=False)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    rows = []
    for s in states:
        kind = 'DIAGNOSTIC ONLY - guard rejected' if s['diagnostic_only'] else s['status']
        rows.append(f'<tr><td><a href="{s["review"]}">Cam{s["camera"]} {html.escape(Path(s["source"]).stem)}</a></td>'
                    f'<td>{kind}</td><td>{s["row_point_counts"]}</td><td>{s["exported_crops"]}</td></tr>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'
        'body{background:#17191d;color:#eee;font:16px Arial;margin:30px}a{color:#8fd1ff}'
        'td,th{padding:12px;border-bottom:1px solid #444;text-align:left}table{border-collapse:collapse}</style>'
        '<h1>HA - classify crops - current runtime</h1><p>Khong sua luong hay tham so. '
        'Native crop PNG + full rectangles + contact sheets.</p>'
        '<p>DIAGNOSTIC ONLY: luong that da dung vi thieu point. Cac crop nay chi la de xuat hinh hoc, '
        'khong duoc nham voi crop classifier da nhan.</p>'
        f'<p>{report["production_frames"]} production frames / {report["production_crops"]} inputs; '
        f'{report["diagnostic_frames"]} rejected frames / {report["diagnostic_crops"]} diagnostic crops.</p>'
        '<table><tr><th>Frame</th><th>Runtime status</th><th>Points by row</th><th>Crops</th></tr>'
        +''.join(rows)+'</table>',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))


if __name__=='__main__':
    main()
