"""Interpolate two full-HA single-network checkpoints; never ensemble or alter thresholds."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from train_classifier_fullha import digest, dump, evaluate


def interpolate(warm, cold, fraction):
    if not 0 < fraction < 1 or set(warm) != set(cold):
        raise ValueError('Same state keys and fraction strictly between zero and one required')
    result = {}
    for key, value in warm.items():
        other = cold[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Incompatible parameter/buffer: ' + key)
        if torch.is_floating_point(value):
            blended = value*(1-fraction)+other*fraction
            if not torch.isfinite(blended).all():
                raise ValueError('Nonfinite blended weights')
            result[key] = blended
        else:
            result[key] = (value if fraction < .5 else other).clone()
    return result


def blend(root, warm_pt, cold_pt, fraction):
    name = f'blend_cold{int(fraction*100):02d}'
    folder = root/'training'/name/'weights'
    if folder.exists():
        raise FileExistsError('Blended checkpoint never overwritten')
    warm, cold = YOLO(str(warm_pt)), YOLO(str(cold_pt))
    if warm.names != cold.names or [warm.names[i] for i in range(3)] != ['ng','no_disk','ok']:
        raise ValueError('Class order/topology differs')
    warm.model = warm.model.cpu().float().eval()
    cold.model = cold.model.cpu().float().eval()
    warm.model.load_state_dict(interpolate(warm.model.state_dict(), cold.model.state_dict(), fraction), strict=True)
    metadata = {'strategy': 'single-network weight/float-buffer interpolation, NOT prediction ensemble',
                'cold_fraction': fraction, 'warm_parent': str(warm_pt), 'cold_parent': str(cold_pt),
                'warm_parent_sha256': digest(warm_pt), 'cold_parent_sha256': digest(cold_pt),
                'both_parents_trained_all_2000_ha': True, 'no_new_labels_or_threshold_changes': True}
    warm.ckpt.update({'train_metrics': None, 'train_results': None, 'best_fitness': None,
                      'blend_metadata': metadata, 'epoch': -1, 'optimizer': None})
    folder.mkdir(parents=True)
    pt = folder/'best.pt'
    warm.save(str(pt))
    model = Path(YOLO(str(pt)).export(format='onnx', dynamic=True, simplify=True, opset=13,
                                    imgsz=224, batch=1, device='cpu'))
    dump(folder.parent/'blend_metadata.json', metadata)
    report, _ = evaluate(root/'datasets'/'full_balanced', model, root/'evaluations'/name, full=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    torch.set_num_threads(4)
    original = json.loads((root/'release_decision.json').read_text(encoding='utf-8'))
    target = root/'release_decision_v2.json'
    if target.exists():
        raise FileExistsError('Blend release decision is never overwritten')
    candidates = original['candidates'][:]
    warm_pt = root/'training'/'full_from_backup'/'weights'/'best.pt'
    cold_pt = root/'training'/'full_from_original'/'weights'/'best.pt'
    for fraction in (.25,.50):
        candidates.append(blend(root, warm_pt, cold_pt, fraction))
    eligible = [c for c in candidates if c['eligible']]
    chosen = max(eligible, key=lambda c:c['safety_rank'])
    source = Path(chosen['model'])
    folder = root/'selected_v2'; folder.mkdir()
    model = folder/'c_p_ha_all_reviewed_20261005.onnx'
    shutil.copy2(source, model); shutil.copy2(source.with_suffix('.pt'), model.with_suffix('.pt'))
    decision = dict(original, chosen_source=str(source), selected_onnx=str(model), selected_pt=str(model.with_suffix('.pt')),
                    onnx_sha256=digest(model), pt_sha256=digest(model.with_suffix('.pt')), candidates=candidates,
                    selection_includes_single_network_weight_interpolations=True,
                    original_release_decision_retained=True, production_deployed=False)
    dump(target, decision)
    print('BLEND SELECTION', source, json.dumps(chosen['domains']), flush=True)


if __name__=='__main__':
    main()
