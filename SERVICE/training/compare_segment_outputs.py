"""Visual before/after gallery; inference folders are preserved."""
import argparse
import html
from pathlib import Path

import cv2
import numpy as np

from infer_segmentor_ha import titled


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--before-title',default='BEFORE / old candidate320 / legacy decoder / conf0.1')
    p.add_argument('--after-title',default='AFTER / refined512 / native RGB decoder / conf0.5')
    p.add_argument('--description',default='Different model, resolution, decoder and color order; not an isolated decoder-only comparison.')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sections=[]
    for original_path in sorted((args.after/'band_original').glob('*.png')):
        original=cv2.imread(str(original_path))
        before=cv2.imread(str(args.before/'band_overlay'/original_path.name))
        after=cv2.imread(str(args.after/'band_overlay'/original_path.name))
        if before.shape!=original.shape or after.shape!=original.shape:
            raise ValueError(original_path)
        image=np.concatenate([titled(original,original_path.stem+' / ORIGINAL'),
            titled(before,args.before_title),
            titled(after,args.after_title)])
        cv2.imwrite(str(args.output/original_path.name),image)
        name=html.escape(original_path.name)
        sections.append(f'<h2>{name}</h2><a href="{name}"><img loading="lazy" src="{name}"></a>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>HA before after</title><style>body{background:#17191d;color:white;font-family:sans-serif}'
        'img{width:100%}h2{font-size:16px}</style><h1>HA before / after</h1>'
        '<p>'+html.escape(args.before_title)+' / '+html.escape(args.after_title)+'. '
        +html.escape(args.description)+' Geometry uses prepared point labels.</p>'+''.join(sections),encoding='utf-8')
    print('Comparison bands:',len(sections))


if __name__=='__main__':
    main()
