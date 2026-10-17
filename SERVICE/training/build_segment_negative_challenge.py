"""Make held-out edge-erased stress tests, not real empty-tray ground truth."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from curate_segment_masks import band_image


def main():
    p=argparse.ArgumentParser()
    for name in ('point-dataset','curated','crops','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output/'images/val').mkdir(parents=True)
    (output/'labels/val').mkdir(parents=True)
    points=json.loads((args.point_dataset/'manifest.json').read_text())['records']
    lookup={r['source']:r for r in points if r['origin']=='factory'}
    masks=json.loads((args.curated/'manifest.json').read_text())
    crops=json.loads((args.crops/'manifest.json').read_text())
    crop_lookup={(r['source'],r['band'],r['tile']):r for r in crops}
    records=[]
    for r in masks:
        if r['point_split']!='val':
            continue
        band=band_image(lookup[r['source']],r['band'])
        mask=cv2.imread(r['curated_mask'],0).squeeze()
        for tile in (2,3):
            crop=crop_lookup[(r['source'],r['band'],tile)]
            x1,_,x2,_=crop['rectified_band_xyxy']
            erase=cv2.dilate(mask[:,x1:x2],np.ones((5,7),np.uint8))
            negative=cv2.inpaint(band[:,x1:x2],erase,5,cv2.INPAINT_TELEA)
            stem=f"{r['camera']}_{Path(r['source']).stem}_{r['band']}_tile{tile}_negative"
            image_path=output/'images/val'/(stem+'.png')
            label_path=output/'labels/val'/(stem+'.txt')
            cv2.imwrite(str(image_path),negative)
            label_path.write_text('')
            records.append({'image':str(image_path),'label':str(label_path),'source':r['source'],
                            'split':'val','origin':'synthetic_negative','objects':0,
                            'scenario':'erased_edge_negative','source_group':Path(r['source']).stem,
                            'label_status':'constructed_negative_not_real_empty_tray'})
    (output/'manifest.json').write_text(json.dumps(records,indent=2))
    print('Held-out constructed negatives:',len(records))


if __name__=='__main__':
    main()
