"""Version disagreement-only corrections; preserve every split and frozen original."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from classifier_metrics import LABELS, classification_metrics
from serve_classifier_result_review import review_payload


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rescore(records):
    domains = {}
    for domain in ('ha_train', 'ha_val', 'ha_test', 'legacy_val'):
        rows = [r for r in records if ('legacy_val' if r['id'].startswith('legacy_') else 'ha_'+r['split']) == domain]
        if not rows:
            continue
        metrics = {'count': len(rows), 'label_counts': dict(Counter(r['label'] for r in rows))}
        for name in rows[0]['predictions']:
            targets = [LABELS.index(r['label']) for r in rows]
            predictions = [LABELS.index(r['predictions'][name]['label']) for r in rows]
            result = classification_metrics(targets, predictions)
            result['confidence_below_0_5'] = sum(r['predictions'][name]['confidence'] < .5 for r in rows)
            result['confident_errors_0_5'] = sum(r['predictions'][name]['confidence'] >= .5 and
                                              r['predictions'][name]['label'] != r['label'] for r in rows)
            metrics[name] = result
        domains[domain] = metrics
    return domains


def build(data, evaluation, output):
    if output.exists():
        raise FileExistsError('Correction version is never overwritten')
    frozen_path = evaluation/'label_review'/'decisions.json'
    frozen_bytes = frozen_path.read_bytes()
    state = json.loads(frozen_bytes)
    expected_digest, eligible, original_report = review_payload(evaluation)
    if state['evaluation_sha256'] != expected_digest:
        raise ValueError('Corrections belong to another frozen evaluation')
    eligible = {r['id']: r for r in eligible}
    decisions = state['decisions']
    if not set(decisions).issubset(eligible):
        raise ValueError('Unknown correction ID')
    for identifier, decision in decisions.items():
        if decision['label'] not in LABELS or decision['original_label'] != eligible[identifier]['original_label']:
            raise ValueError('Invalid or stale correction')
    new_error_ids = {identifier for identifier, r in eligible.items()
                     if r['predictions']['candidate']['label'] != r['original_label']}
    if not new_error_ids.issubset(decisions):
        raise ValueError('Selected model disagreements remain unreviewed')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    evaluated = json.loads((evaluation/'per_image.json').read_text(encoding='utf-8'))
    labels = {r['id']: r['label'] for r in evaluated}
    records, audit, seen_hashes = [], [], {}
    for record in manifest['records']:
        r = dict(record)
        identifier = r['id']
        source = data/r['image']
        if digest(source) != r['sha256']:
            raise ValueError('Original native crop changed: ' + identifier)
        decision = decisions.get(identifier)
        if decision and labels[identifier] != r['label']:
            raise ValueError('Correction GT differs from original dataset: ' + identifier)
        if decision:
            r['post_evaluation_review'] = decision
            if r['label'] != decision['label']:
                audit.append({'id': identifier, 'origin': r.get('origin', 'ha'), 'split': r['split'],
                              'old_label': r['label'], 'new_label': decision['label'],
                              'image_sha256': r['sha256'], 'notes': decision.get('notes', '')})
                r['label_before_correction'] = r['label']
                r['label'] = decision['label']
        r['source_version_image'] = str(source.resolve())
        r['image'] = f'{r["split"]}/{r["label"]}/{identifier}{source.suffix.lower()}'
        previous_split = seen_hashes.setdefault(r['sha256'], r['split'])
        if previous_split != r['split']:
            raise ValueError('Exact crop duplicate crosses splits')
        records.append(r)
    if not set(decisions).issubset({r['id'] for r in records}):
        raise ValueError('Corrections not represented in the versioned dataset')
    for r in evaluated:
        decision = decisions.get(r['id'])
        r['original_label'] = r['label']
        if decision:
            r['label'] = decision['label']
            r['post_evaluation_review'] = decision
    corrected_domains = rescore(evaluated)
    legacy_anchor = corrected_domains['legacy_val']['candidate']
    # Retain NG/empty protection relative to the already selected model on SAME corrected GT.
    gates = {'ha_ok_recall_min': .90, 'legacy_ok_recall_min': .95,
             'legacy_accuracy_min': max(.97, legacy_anchor['accuracy']-.0025),
             'legacy_ng_misses_max': legacy_anchor['ng_not_detected'],
             'legacy_empty_as_ok_max': legacy_anchor['no_disk_as_ok']}
    protocol = {'reviewed_disagreements': len(decisions), 'changed_labels': len(audit),
                'changes_by_origin_split': dict(Counter(r['origin']+'_'+r['split'] for r in audit)),
                'remaining_old_only_disagreements_not_reviewed': len(eligible)-len(decisions),
                'source_evaluation_sha256': expected_digest, 'correction_sha256': hashlib.sha256(frozen_bytes).hexdigest(),
                'revision': state['revision'], 'correction_audit_scope': 'prediction-visible original disagreements only',
                'test_previously_viewed_and_labels_reviewed': True,
                'test_limitation': 'same-session reviewed holdout, NOT a new blind or independent factory test',
                'test_images_in_training': False, 'test_used_for_checkpoint_selection': False,
                'split_changes': 0, 'legacy_source_files_unchanged': True}
    if frozen_path.read_bytes() != frozen_bytes:
        raise ValueError('Human decisions changed during freeze; retry as a new version')
    output.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        for label in LABELS:
            (output/split/label).mkdir(parents=True)
    for r in records:
        target = output/r['image']
        shutil.copy2(r['source_version_image'], target)
        if digest(target) != r['sha256']:
            raise ValueError('Copy checksum failed')
    for filename in ('review_items_snapshot.json', 'review_decisions_snapshot.json'):
        if (data/filename).exists():
            shutil.copy2(data/filename, output/filename)
    (output/'post_evaluation_decisions_snapshot.json').write_bytes(frozen_bytes)
    (output/'correction_audit.json').write_text(json.dumps({'protocol': protocol, 'changes': audit},
                                                        ensure_ascii=False, indent=2), encoding='utf-8')
    manifest.update({'created_at': datetime.now(timezone.utc).isoformat(), 'records': records,
                     'source_dataset': str(data.resolve()), 'source_manifest_sha256': digest(data/'manifest.json'),
                     'label_correction_protocol': protocol, 'validation_safety_gates': gates,
                     'selection_policy': 'Corrected validation only; HA 75% / legacy 25% NG->non-NG risk; '
                                         'NG/empty gates relative to previous selected model on same corrected GT',
                     'counts': {s: dict(Counter(r['label'] for r in records if r['split'] == s))
                                for s in ('train', 'val', 'test')}})
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    original_report.update({'domains': corrected_domains, 'dataset': str(output.resolve()),
                            'label_correction_protocol': protocol,
                            'rescore_only_frozen_predictions': True,
                            'original_evaluation': str(evaluation.resolve())})
    baseline_folder = output.parent/'previous_models_corrected'
    baseline_folder.mkdir()
    (baseline_folder/'report.json').write_text(json.dumps(original_report, ensure_ascii=False, indent=2), encoding='utf-8')
    (baseline_folder/'per_image.json').write_text(json.dumps(evaluated, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'protocol': protocol, 'counts': manifest['counts'], 'validation_safety_gates': gates}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.data.resolve(), args.evaluation.resolve(), args.output.resolve())
