"""Compare native crop exports, proving widening retains the old crop pixels."""
import argparse
import html
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def href(path, parent):
    return Path(os.path.relpath(path,parent)).as_posix()


def preview(image, max_width=120, max_height=162):
    scale=min(max_width/image.shape[1],max_height/image.shape[0])
    return cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--before',required=True,type=Path)
    parser.add_argument('--after',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    before=json.loads((args.before/'manifest.json').read_text(encoding='utf-8-sig'))
    after=json.loads((args.after/'manifest.json').read_text(encoding='utf-8-sig'))
    old_frames={f['source']:f for f in before['frames']}
    args.output.mkdir(parents=True)
    total,changes,neighbor_hits,neighbor_checks=0,0,0,0
    rows=[]; records=[]
    for frame in after['frames']:
        old=old_frames[frame['source']]
        frame_name=Path(frame['review']).parent.name
        target=args.output/frame_name
        target.mkdir()
        old_root=args.before/Path(old['review']).parent
        new_root=args.after/Path(frame['review']).parent
        old_crops={(c['region'],c['index']):c for c in old['crops']}
        content=[f'<h1>{html.escape(frame_name)}</h1><p>TRAI: 1.5; PHAI: 1.8. '
                 'Cung chieu cao, cung ranhs; PNG native giu nguyen pixel. '
                 '0 dia = no_disk; 1 dia = ok; nhieu hon 1 dia tai ranhs muc tieu = ng. '
                 'Dia cua ranhs ben canh khong tinh vao nhan. Du doan model khong phai ground truth.</p>']
        for region in ('row1_bottom','row2_top','row2_bottom','row3_top'):
            crops=[c for c in frame['crops'] if c['region']==region]
            columns,cell_w,cell_h=5,280,215
            sheet=np.full((((len(crops)+columns-1)//columns)*cell_h,columns*cell_w,3),22,np.uint8)
            content += [f'<h2>{region}</h2><a href="{region}_comparison.png"><img src="{region}_comparison.png"></a>',
                        '<details><summary>Tung crop native: cu / moi</summary><div class="pairs">']
            for i,c in enumerate(crops):
                b=old_crops[(region,c['index'])]
                old_image=cv2.imread(str(old_root/b['crop']))
                new_image=cv2.imread(str(new_root/c['crop']))
                old_native=target/'native_before'/region/f'{c["index"]:02d}.png'
                new_native=target/'native_after'/region/f'{c["index"]:02d}.png'
                old_native.parent.mkdir(parents=True,exist_ok=True)
                new_native.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(old_root/b['crop'],old_native)
                shutil.copy2(new_root/c['crop'],new_native)
                assert b['effective_box'][1::2]==c['effective_box'][1::2]
                assert np.allclose(b['point_box'],c['point_box'],atol=1e-5)
                left=(c['effective_box'][2]-b['effective_box'][2] if c['rotated_180']
                      else b['effective_box'][0]-c['effective_box'][0])
                assert left>=0 and new_image.shape[1]>=old_image.shape[1]
                assert np.array_equal(new_image[:,left:left+old_image.shape[1]],old_image)
                total+=1; changes+=int(b['label']!=c['label'])
                x,y=(i%columns)*cell_w,(i//columns)*cell_h
                for j,(image,record) in enumerate(((old_image,b),(new_image,c))):
                    view=preview(image); px=x+j*140+(140-view.shape[1])//2
                    sheet[y+25:y+25+view.shape[0],px:px+view.shape[1]]=view
                    text=f"{image.shape[1]}x{image.shape[0]} {record['label']}"
                    cv2.putText(sheet,text,(x+j*140+3,y+204),cv2.FONT_HERSHEY_SIMPLEX,.37,(230,230,230),1)
                cv2.putText(sheet,f"#{c['index']:02d} BEFORE / AFTER",(x+8,y+18),cv2.FONT_HERSHEY_SIMPLEX,.45,(100,220,255),1)
                nx1,_,nx2,_=c['effective_box']
                for j in (i-1,i+1):
                    if 0<=j<len(crops):
                        point=crops[j]['point_box']; ncx=(point[0]+point[2])/2
                        neighbor_hits+=int(nx1<=ncx<nx2); neighbor_checks+=1
                records.append(dict(frame=frame_name,region=region,index=c['index'],
                                    before_width=old_image.shape[1],after_width=new_image.shape[1],
                                    old_pixels_retained=True,y_unchanged=True,
                                    before_label=b['label'],after_label=c['label']))
                content += [f'<div>#{c["index"]:02d}<br><a href="{href(old_native,target)}"><img src="{href(old_native,target)}"></a>'
                            f'<a href="{href(new_native,target)}"><img src="{href(new_native,target)}"></a></div>']
            if not cv2.imwrite(str(target/f'{region}_comparison.png'),sheet): raise OSError(target)
            content+=['</div></details>']
        (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{background:#181a1e;color:#eee;font:16px Arial;margin:25px}'
            'a{color:#9cf}img{max-width:100%}.pairs{display:flex;gap:12px;flex-wrap:wrap}.pairs img{height:160px;width:auto}</style>'+''.join(content),encoding='utf-8')
        rows.append(f'<p><a href="{frame_name}/index.html">{frame_name}</a></p>')
    summary=dict(crops_compared=total,label_changes=changes,old_pixels_retained=total,
                 neighbor_centers_inside_after=neighbor_hits,neighbor_checks=neighbor_checks,
                 before_width_median=float(np.median([r['before_width'] for r in records])),
                 after_width_median=float(np.median([r['after_width'] for r in records])),
                 note='Label changes are old-model prediction drift, not an accuracy improvement metric.')
    (args.output/'report.json').write_text(json.dumps(dict(summary=summary,crops=records),indent=2))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{background:#181a1e;color:#eee;font:16px Arial;margin:30px}a{color:#9cf}</style>'
        '<h1>Classify crop: 1.5 vs 1.8</h1><p>Trai: cu. Phai: moi. Khong thay height hay tam ranhs. '
        'Nhan classifier: 0 dia = no_disk; 1 dia = ok; >1 dia tai cung ranhs = ng. '
        'Khong dem dia cua ranhs ben canh. Nhan hien tren sheet chi la du doan model cu.</p>'
        '<a href="report.json">Pixel/geometry audit</a>'+''.join(rows),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
