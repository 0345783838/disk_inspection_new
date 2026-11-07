"""HA/new fitted-data diagnostics, always joined to current canonical labels."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from .classifier_dataset import PROJECT, atomic_json, read
except ImportError:
    from classifier_dataset import PROJECT, atomic_json, read

REGISTRY = PROJECT / '.work/ACTIVE_CLASSIFIER_EVALUATION.json'
ORIGINS = ('old_ha', 'new_factory')


def live_diagnostics(manifest, dataset_root, registry=REGISTRY):
    if not registry.exists():
        return None
    config = read(registry)
    if Path(config['dataset_path']).resolve() != Path(dataset_root).resolve():
        raise ValueError('Evaluation scope belongs to another canonical dataset')
    if tuple(config['origins']) != ORIGINS:
        raise ValueError('Diagnostic scope must contain HA and new data only')
    source = read(config['predictions_path'])
    if source['origins'] != list(ORIGINS):
        raise ValueError('Prediction origins differ from diagnostic scope')
    model = Path(source['model_path'])
    if hashlib.sha256(model.read_bytes()).hexdigest() != source['model_sha256']:
        return dict(available=False, error='Model đã thay đổi; cần chạy lại dự đoán rà lỗi.')
    predictions = source['predictions']
    if len({p['id'] for p in predictions}) != len(predictions):
        raise ValueError('Duplicate diagnostic prediction ID')
    by_id = {p['id']: p for p in predictions}
    selected = [r for r in manifest['records'] if r['origin'] in ORIGINS]
    current, missing, stats = {}, [], {}
    for origin in ORIGINS:
        counts = Counter()
        for r in selected:
            if r['origin'] != origin:
                continue
            counts['count'] += 1
            p = by_id.get(r['id'])
            if p is None or p['sha256'] != r['sha256']:
                missing.append(r['id'])
                continue
            current[r['id']] = dict(label=p['label'], confidence=p['confidence'], probabilities=p['probabilities'])
            counts['scored'] += 1
            counts['errors'] += p['label'] != r['label']
            counts['ng_not_detected'] += r['label'] == 'ng' and p['label'] != 'ng'
            counts['no_disk_as_ok'] += r['label'] == 'no_disk' and p['label'] == 'ok'
            counts['false_ng'] += r['label'] != 'ng' and p['label'] == 'ng'
        stats[origin] = {k: counts[k] for k in ('count', 'scored', 'errors', 'ng_not_detected', 'no_disk_as_ok', 'false_ng')}
    return dict(available=True, origins=list(ORIGINS), count=len(selected), by_id=current,
                counts_by_origin=stats, missing_prediction_ids=missing,
                canonical_revision=manifest['canonical_revision'], prediction_source_revision=source['canonical_revision'],
                model_path=str(model), model_sha256=source['model_sha256'], generated_at=source['generated_at'],
                scope='All HA and new crops; fitted-data diagnostics, not independent validation. Labels remain canonical.')


def generate(output):
    # Inference only. No training authorization or labels are changed.
    if output.exists():
        raise FileExistsError('Diagnostic prediction snapshots are immutable: ' + str(output))
    try:
        from .classifier_dataset import REGISTRY as DATASET_REGISTRY
        from .evaluate_classifier_ha import predict
    except ImportError:
        from classifier_dataset import REGISTRY as DATASET_REGISTRY
        from evaluate_classifier_ha import predict
    root = Path(read(DATASET_REGISTRY)['path'])
    raw = (root / 'manifest.json').read_bytes()
    m = json.loads(raw)
    rows = [dict(r, path=str(root / r['image'])) for r in m['records'] if r['origin'] in ORIGINS]
    model = PROJECT / 'SERVICE/config/models/c_p.onnx'
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    for r in rows:
        if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Canonical crop checksum failed: ' + r['id'])
    _, outputs = predict(model, rows)
    if raw != (root / 'manifest.json').read_bytes():
        raise ValueError('Canonical labels changed during inference; rerun')
    if model_hash != hashlib.sha256(model.read_bytes()).hexdigest():
        raise ValueError('Runtime model changed during inference; rerun')
    output.mkdir(parents=True)
    (output / 'source_manifest.json').write_bytes(raw)
    atomic_json(output / 'predictions.json', dict(
        canonical_revision=m['canonical_revision'], manifest_sha256=hashlib.sha256(raw).hexdigest(),
        model_path=str(model), model_sha256=model_hash, origins=list(ORIGINS),
        generated_at=datetime.now(timezone.utc).isoformat(),
        predictions=[dict(id=r['id'], sha256=r['sha256'], **p) for r, p in zip(rows, outputs)]))
    atomic_json(REGISTRY, dict(schema_version=1, dataset_path=str(root), origins=list(ORIGINS),
        predictions_path=str((output / 'predictions.json').resolve()), split_changes=False,
        purpose='Current-label HA/new error review; fitted data, not independent holdout',
        user_instruction='Dùng toàn bộ HA + ảnh mới để rà lỗi, chưa chia lại train/val'))
    live = live_diagnostics(m, root)
    atomic_json(output / 'summary_at_generation.json', {k: v for k, v in live.items() if k != 'by_id'})
    print({k: v for k, v in live.items() if k not in ('by_id', 'scope')}, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    generate(parser.parse_args().output.resolve())
