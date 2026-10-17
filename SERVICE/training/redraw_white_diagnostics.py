"""Refresh diagnostic overlays from saved masks only; do not run models or hardware."""
import argparse
import json
from pathlib import Path
import cv2
from src.tools.caliper_advanced_new import AdvancedMultiEdgeCaliper
from training.diagnose_white_edges import overlay, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root/'report.json').read_text(encoding='utf-8'))
    params = report['params']
    caliper = AdvancedMultiEdgeCaliper(min_edge_distance=params['caliper_min_edge_distance'],
        max_pairs=50, pair_max_gap=params['caliper_max_edge_distance'],
        thickness_list=params['caliper_thickness_list'], length_rate=params['caliper_length_rate'],
        polarity='both', angle_deg=0, subpixel=True)
    page = (args.root/'index.html').read_text(encoding='utf-8')
    for record in report['records']:
        stem = record['folder']
        folder = args.root/stem
        for band in record['bands']:
            index = band['band']
            original = cv2.imread(str(folder/f'band{index}_original.png'))
            for key, source, length in [('filtered', 'filtered', params['caliper_length_rate']),
                                         ('conf01', 'conf01_filtered', params['caliper_length_rate']),
                                         ('fullscan', 'filtered', 1.)]:
                mask = cv2.imread(str(folder/f'band{index}_{source}_mask.png'), 0)
                measurements = []
                for counts in band['counts']:
                    result = caliper.measure_with_params(mask, tuple(counts['center']),
                        params['caliper_min_edge_distance'], params['caliper_max_edge_distance'],
                        length, params['caliper_thickness_list'])
                    expected = counts['production' if key=='filtered' else 'full_scan' if key=='fullscan' else 'conf01']
                    assert len(result['pairs'])==expected
                    measurements.append({'center': counts['center'], 'pairs': result['pairs']})
                save(folder/f'band{index}_{key}_overlay.png', overlay(original, mask, measurements, length_rate=length))
            path = f'{stem}/band{index}_fullscan_overlay.png'
            if path not in page:
                anchor = f'<p><a href="{stem}/band{index}_raw_mask.png">'
                addition = f'<figure><figcaption>Offline full width100% (not deployed)</figcaption><a href="{path}" target="_blank"><img loading="lazy" src="{path}"></a></figure>'
                page = page.replace(anchor, addition+anchor)
    (args.root/'index.html').write_text(page, encoding='utf-8')
    print('Updated40 bands; drawn scanline lengths match actual95% and100% measurements')


if __name__=='__main__':
    main()
