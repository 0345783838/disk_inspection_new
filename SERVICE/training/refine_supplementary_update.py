"""Evaluate smaller update fractions after CV showed retention loss at 25%.

Uses only grouped CV for fraction selection; final fitting metrics cannot choose
the fraction. Keeps the first protocol/decision and all prior artifacts intact.
"""
from pathlib import Path
import argparse
import json
import shutil
import time

from train_classifier_supplementary import read, dump, blend, evaluate, population_metrics, render, digest
from evaluate_classifier_ha import torch_onnx_parity


def wait_for(path):
    while True:
        if path.exists():
            try:
                return read(path)
            except json.JSONDecodeError:
                pass
        time.sleep(15)


def main(root, fractions=(.05, .10, .15), reason=None):
    if not fractions or any(not 0 < fraction < .25 for fraction in fractions):
        raise ValueError('Fine grid fractions must be strictly between zero and .25')
    dump(root / 'fine_grid_protocol.json', {
        'fractions': fractions, 'reason': reason or 'First grouped CV fold lost legacy retention and one reviewed NG at 25% or higher.',
        'exploratory_adaptation_of_cv_grid': True, 'final_fit_results_not_used_to_select_fraction': True,
        'no_additional_training_or_label_changes': True})
    protocol = read(root / 'protocol.json')
    results = {str(f): [] for f in fractions}
    for fold in range(3):
        while True:
            progress = wait_for(root / 'cv_progress.json')
            if len(progress['0.0']) > fold:
                break
            time.sleep(15)
        data = root / 'datasets' / f'group_cv_{fold}'
        records = [dict(r, path=str(data / r['image'])) for r in read(data / 'manifest.json')['records'] if r['split'] == 'val']
        pt = root / 'training' / f'group_cv_{fold}' / 'weights' / 'best.pt'
        for fraction in fractions:
            name = f'fine_cv{fold}_fraction{fraction}'
            model = blend(root, pt, fraction, name)
            folder = root / 'evaluations' / name
            result = evaluate(model, records, folder, protocol['gates'])
            result['predictions_file'] = str(folder / 'predictions.json')
            results[str(fraction)].append(result)
            dump(root / 'fine_grid_progress.json', results)
            print(name, 'human NG misses', result['metrics']['new_human']['ng_not_detected'],
                  'false NG', result['metrics']['new_human']['false_ng'], 'retention', result['retention_failures'], flush=True)
    first = wait_for(root / 'cv_selection.json')
    summaries = first['summaries'][:]
    for fraction in fractions:
        folds = results[str(fraction)]
        rows = [r for f in folds for r in read(f['predictions_file']) if r['origin'] == 'new_factory']
        if len(rows) != 2800 or len({r['id'] for r in rows}) != 2800:
            raise ValueError('Incomplete OOF coverage')
        metrics = population_metrics(rows, [r['prediction'] for r in rows])
        m = metrics['new_human']
        eligible = all(f['eligible'] for f in folds)
        key = (eligible, -m['ng_not_detected'], -m['false_ng'], -metrics['new_all']['errors'],
               -max(f['metrics']['old_ha']['errors'] for f in folds), -fraction)
        summaries.append({'fraction': fraction, 'eligible_all_folds': eligible, 'metrics': metrics, 'ranking': key,
                          'legacy_retention_per_fold': [f['metrics']['legacy_val'] for f in folds],
                          'old_ha_retention_per_fold': [f['metrics']['old_ha'] for f in folds]})
        dump(root / 'oof' / f'fraction{fraction}.json', rows)
    chosen = max(summaries, key=lambda r: tuple(r['ranking']))
    dump(root / 'cv_selection_v2.json', {'fraction': chosen['fraction'], 'summaries': summaries,
        'selection_population': 'Grouped new-collection CV plus old retention only',
        'grid_adapted_after_first_CV_fold': True, 'final_fit_predictions_not_used_for_selection': True})
    initial_release = wait_for(root / 'release_decision.json')
    pool = Path(protocol['pool'])
    records = [dict(r, path=str(pool / r['image'])) for r in read(pool / 'manifest.json')['records'] if r['origin'] != 'old_legacy_train']
    fraction = chosen['fraction']
    pt = root / 'training' / 'final_all_new' / 'weights' / 'best.pt'
    model = Path(protocol['baseline_onnx']) if fraction == 0 else pt.with_suffix('.onnx') if fraction == 1 else blend(root, pt, fraction, f'fine_final_fraction{fraction}')
    final = evaluate(model, records, root / 'evaluations' / 'final_v2', protocol['gates'])
    parity = torch_onnx_parity(model.with_suffix('.pt'), model, records[:32])
    before, after = initial_release['baseline']['metrics']['new_human'], final['metrics']['new_human']
    recommend = (final['eligible'] and after['ng_not_detected'] <= before['ng_not_detected']
                 and after['false_ng'] < before['false_ng'] and chosen['eligible_all_folds'])
    selected = root / 'selected_v2'
    selected.mkdir()
    artifact = selected / 'c_p_supplementary_20261020.onnx'
    shutil.copy2(model, artifact)
    shutil.copy2(model.with_suffix('.pt'), artifact.with_suffix('.pt'))
    release = dict(initial_release, selected_onnx=str(artifact), selected_pt=str(artifact.with_suffix('.pt')),
                   onnx_sha256=digest(artifact), pt_sha256=digest(artifact.with_suffix('.pt')),
                   cv_fraction=fraction, cv_selection=str(root / 'cv_selection_v2.json'),
                   candidate=final, onnx_pt_parity=parity, recommended_for_review=recommend,
                   fraction_grid_exploratory_extension=True, previous_release_retained=True)
    dump(root / 'release_decision_v2.json', release)
    # Keep the original report for audit; final report links to the new decision and artifacts.
    shutil.copy2(root / 'index.html', root / 'index_v1.html')
    render(root, release, chosen)
    page = (root / 'index.html').read_text(encoding='utf-8')
    page = page.replace('href="release_decision.json"', 'href="release_decision_v2.json"')
    page = page.replace('href="cv_selection.json"', 'href="cv_selection_v2.json"')
    page = page.replace('href="selected/', 'href="selected_v2/')
    # The shared report renderer's error gallery reads final/predictions; replace it with v2-specific errors.
    if final['metrics']['new_human']['errors'] != initial_release['candidate']['metrics']['new_human']['errors']:
        start = page.index('<h2>Lỗi còn lại trên 113 crop đã sửa</h2>')
        page = page[:start] + '<h2>Lỗi còn lại trên 113 crop đã sửa</h2><p>Xem danh sách đầy đủ trong <a href="evaluations/final_v2/predictions.json">kết quả từng crop</a>.</p></html>'
    (root / 'index.html').write_text(page, encoding='utf-8')
    print('FINE GRID COMPLETE', fraction, 'recommended', recommend, 'human', after, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fractions', default='.05,.10,.15')
    parser.add_argument('--reason')
    args = parser.parse_args()
    main(args.root.resolve(), tuple(float(value) for value in args.fractions.split(',')), args.reason)
