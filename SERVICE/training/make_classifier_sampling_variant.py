"""Create an immutable sampling variant; native crops, labels and splits unchanged."""
import argparse
import json
import shutil
from pathlib import Path


def build(source, output, ha_probability):
    if output.exists():
        raise FileExistsError('Sampling variants are never overwritten')
    if not 0 < ha_probability < 1:
        raise ValueError('HA probability must be between zero and one')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['domain'] != 'HA_dual_validation_fault_replay':
        raise ValueError('Requires dual validation replay')
    manifest.update({'ha_sampling_probability': ha_probability,
                     'legacy_sampling_probability': 1-ha_probability,
                     'sampling_variant_source': str(source.resolve())})
    shutil.copytree(source, output)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'ha_probability': ha_probability,
                      'legacy_probability': 1-ha_probability}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ha-probability', type=float, required=True)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve(), args.ha_probability)
