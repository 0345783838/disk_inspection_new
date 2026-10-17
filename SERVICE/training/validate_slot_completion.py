"""Model-only HA audit, followed by controlled deletion of detected slot boxes."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'SERVICE'))
from src.service.check_disk_service_yolo import DiskCheckingService
from src.tools.normalize_factory_frames import normalize_frame
from src.tools.slot_geometry import complete_slot_rows, centers


def main():
    output=ROOT/'.work/detector_deploy_20261001/geometry_validation'
    output.mkdir(parents=True,exist_ok=True)
    service=DiskCheckingService()
    patterns=([12],[0],[24],[0,24],[0,1,2,23,24],[4,8,12,16,20],list(range(5)),list(range(20,25)))
    records=[]
    for camera in (1,2):
        for path in sorted(Path('D:/huynhvc/OTHERS/disk_checking/disk_checking/new_data/HA',f'Cam{camera}').glob('*.bmp')):
            image=normalize_frame(cv2.imread(str(path)),camera)
            boxes,scores,_=service.disk_point_detect_model(image)
            truth=service.split_rows(boxes,expected_points_per_row=25)
            complete,meta=complete_slot_rows(truth,image.shape)
            assert not meta and all(np.array_equal(a,b) for a,b in zip(complete,truth))
            for row in range(3):
                for missing in patterns:
                    rows=[r.copy() for r in truth]
                    rows[row]=np.delete(rows[row],missing,axis=0)
                    record=dict(camera=camera,source=path.name,row=row+1,removed=[i+1 for i in missing])
                    try:
                        recovered,meta=complete_slot_rows(rows,image.shape)
                        error=np.linalg.norm(centers(recovered[row])[missing]-centers(truth[row])[missing],axis=1)
                        record.update(accepted=True,max_center_error=float(max(error)),inferred=len(meta))
                        assert len(meta)==len(missing)
                        if row==0 and missing==[0,1,2,23,24]:
                            preview=image.copy()
                            for r in rows:
                                for box in r:
                                    x1,y1,x2,y2=map(int,box)
                                    cv2.rectangle(preview,(x1,y1),(x2,y2),(0,200,0),2)
                            for item in meta:
                                x1,y1,x2,y2=map(int,item['box'])
                                cv2.rectangle(preview,(x1,y1),(x2,y2),(0,215,255),3)
                                cv2.putText(preview,f"INFER {item['slot']}",(x1,max(15,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,215,255),1)
                            cv2.putText(preview,'CONTROLLED BOX DELETION - not a real NG scene',(20,2550),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,215,255),2)
                            cv2.imwrite(str(output/f'Cam{camera}_{path.stem}_infer5.png'),preview)
                    except ValueError as ex:
                        record.update(accepted=False,reason=str(ex))
                    records.append(record)
    accepted=[r for r in records if r['accepted']]
    summary=dict(cases=len(records),accepted=len(accepted),rejected=len(records)-len(accepted),
                 max_center_error=max(r['max_center_error'] for r in accepted),
                 p95_center_error=float(np.percentile([r['max_center_error'] for r in accepted],95)),
                 note='Deleted model boxes on HA images, not independent occlusion/NG scenes.')
    (output/'report.json').write_text(json.dumps(dict(summary=summary,cases=records),indent=2))
    links=''.join(f'<p>{p.stem}</p><a href="{p.name}"><img src="{p.name}"></a>' for p in sorted(output.glob('*_infer5.png')))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'
        'body{background:#17191d;color:#eee;font:16px Arial;margin:25px}img{max-width:600px;width:100%}'
        'a{color:#9cf}</style><h1>Controlled slot recovery</h1><p>Green: retained detections. '
        'Yellow: geometry-inferred. Five original detector boxes were deliberately removed; '
        'these are not real occlusion/NG scenes.</p><a href="report.json">Detailed report</a>'+links)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
