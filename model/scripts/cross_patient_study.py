"""Verify published evidence or replay the frozen 24-run study on private data.

The publication contains anonymous metadata only. Replay always uses a new
output directory and never counts as a new independent-patient experiment.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

DEFAULT_CAPSULE = Path(__file__).resolve().parents[1] / 'experiments/cross_patient_20261022_01'
METRICS = ['dice', 'iou', 'precision', 'recall', 'hausdorff_95', 'mean_surface_distance', 'area_ratio']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stats(values):
    values = [v if v is not None and math.isfinite(v) else None for v in values]
    finite = [v for v in values if v is not None]
    return {'mean': statistics.mean(values) if len(values) == len(finite) and values else None,
            'sd': statistics.stdev(values) if len(values) == len(finite) and len(values) > 1 else None,
            'values': values, 'nonfinite_count': len(values) - len(finite),
            'finite_mean': statistics.mean(finite) if finite else None}


def validate_protocol(protocol):
    records = protocol['records']
    names = [r['filename'] for r in records]
    if len(names) != len(set(names)) or Counter(r['split'] for r in records) != {'train': 20, 'val': 5, 'test': 69}:
        raise ValueError('Missing, duplicate, or unexpected split records')
    patients = {s: {r['patient'] for r in records if r['split'] == s} for s in ['train', 'val', 'test']}
    if patients != {'train': {'patient_001'}, 'val': {'patient_001'}, 'test': {'patient_002'}}:
        raise ValueError('Patient leakage or unexpected patient identity')
    dev = [r for r in records if r['split'] != 'test']
    if any(r['phase'] != 'Z' for r in dev):
        raise ValueError('Non-Z development sample')
    if {r['source_group'] for r in dev if r['split'] == 'train'} & {r['source_group'] for r in dev if r['split'] == 'val'}:
        raise ValueError('Source group crosses train and validation')
    if len({r['decoded_image_sha256'] for r in records}) != len(records):
        raise ValueError('Duplicate decoded image pixels')
    groups = protocol['test_subgroups']
    test_names = {r['filename'] for r in records if r['split'] == 'test'}
    if set(groups['all_69_auxiliary']) != test_names:
        raise ValueError('Lost test samples')
    a, b = set(groups['Z_original_label']), set(groups['Z_instrument_occluded'])
    if len(a) != 12 or len(b) != 13 or a & b or a | b != set(groups['primary_Z_all']):
        raise ValueError('Primary Z strata mismatch')
    for group, filenames in groups.items():
        if len(filenames) != len(set(filenames)) or not set(filenames) <= test_names:
            raise ValueError('Invalid subgroup: ' + group)
    identities = [r['run_id'] for r in protocol['runs']]
    if len(set(identities)) != len(identities) or len(identities) != 24:
        raise ValueError('Missing or duplicate training runs')
    conditions = {r['condition_id'] for r in protocol['runs']}
    if len(conditions) != 8:
        raise ValueError('Expected eight conditions')
    for condition in conditions:
        runs = [r for r in protocol['runs'] if r['condition_id'] == condition]
        if sorted(r['seed'] for r in runs) != [42, 43, 44]:
            raise ValueError('Unmatched seeds')


def aggregate(protocol, evaluation_rows):
    """Macro average within seed, then mean/sample SD; never pool pixels."""
    result = {}
    for condition in dict.fromkeys(r['condition_id'] for r in protocol['runs']):
        runs = sorted([r for r in protocol['runs'] if r['condition_id'] == condition], key=lambda r: r['seed'])
        groups = {}
        for group, filenames in protocol['test_subgroups'].items():
            summary = {}
            for metric in METRICS:
                means = []
                for run in runs:
                    rows = evaluation_rows[run['run_id']]
                    values = [0.0 if metric == 'precision' and rows[n]['empty_prediction'] else rows[n][metric] for n in filenames]
                    means.append(statistics.mean(values) if values and all(v is not None and math.isfinite(v) for v in values) else None)
                summary[metric] = stats(means)
            summary['n_images'] = len(filenames)
            summary['empty_predictions_by_seed'] = [sum(evaluation_rows[r['run_id']][n]['empty_prediction'] for n in filenames) for r in runs]
            groups[group] = summary
        result[condition] = {'groups': groups}
    return result


def close(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def verify(capsule):
    capsule = capsule.resolve()
    manifest = load(capsule / 'manifest_sha256.json')
    for relative, expected in manifest.items():
        path = (capsule / relative).resolve()
        if not path.is_relative_to(capsule) or sha(path) != expected:
            raise ValueError('Published evidence hash mismatch: ' + relative)
    protocol = load(capsule / 'protocol.public.json')
    validate_protocol(protocol)
    raw = {}
    completed = load(capsule / 'run_summary.json')
    locked = load(capsule / 'checkpoint_lock.json')['checkpoints']
    test_names = set(protocol['test_subgroups']['all_69_auxiliary'])
    for run in protocol['runs']:
        run_id = run['run_id'];folder = capsule / 'runs' / run_id
        history = load(folder / 'checkpoints/history.json')
        record = completed[run_id]
        if len(history) != 120 or [e['epoch'] for e in history] != list(range(1, 121)):
            raise ValueError('Incomplete epoch history: ' + run_id)
        best = max(history, key=lambda e: e['val']['dice'])
        if best['epoch'] != record['best_epoch'] or not close(best['val']['dice'], record['best_validation_dice']):
            raise ValueError('Checkpoint not selected by validation: ' + run_id)
        if record['checkpoint_sha256'] != locked[run_id]:
            raise ValueError('Checkpoint identity mismatch')
        if load(folder / 'config.json') != run['config']:
            raise ValueError('Run configuration mismatch')
        init = load(folder / 'initialization.json')
        if init['initial_epoch'] != 0 or init['optimizer_initial_entries'] != 0 or init['loaded_prior_ICG_checkpoint']:
            raise ValueError('Run did not initialize fresh')
        rows = load(folder / 'test/evaluation.json')['per_sample']
        if len(rows) != 69 or {r['filename'] for r in rows} != test_names:
            raise ValueError('Lost or duplicate test predictions')
        raw[run_id] = {r['filename']: r for r in rows}
    recomputed = aggregate(protocol, raw)
    original = load(capsule / 'summary.json')['conditions']
    for condition, item in recomputed.items():
        for group, values in item['groups'].items():
            for metric in METRICS:
                expected = original[condition]['groups'][group][metric]
                actual = values[metric]
                for key in ['mean', 'sd', 'finite_mean']:
                    if not close(actual[key], expected[key]):
                        raise ValueError(f'Aggregate mismatch: {condition}/{group}/{metric}/{key}')
                if actual['nonfinite_count'] != expected['nonfinite_count'] or not all(close(a, b) for a, b in zip(actual['values'], expected['values'])):
                    raise ValueError('Per-seed metric mismatch')
    return {'passed': True, 'manifest_files_checked': len(manifest), 'runs_checked': 24,
            'test_frames_per_run': 69, 'primary_Z_frames': 25, 'test_patients': 1,
            'all_subgroup_mean_and_sample_SD_recomputed': True,
            'private_images_or_weights_required': False}


def replace_roots(value, data, output):
    if isinstance(value, dict):
        return {k: replace_roots(v, data, output) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_roots(v, data, output) for v in value]
    if isinstance(value, str):
        return value.replace('${DATA_ROOT}', str(data)).replace('${STUDY_ROOT}', str(output))
    return value


def prepare(capsule, data, output):
    capsule, data, output = capsule.resolve(), data.resolve(), output.resolve()
    verify(capsule)
    if output.exists():
        raise FileExistsError('Replay output must be a new directory')
    for source in (data, capsule):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Inputs, publication and outputs must be separate')
    public = load(capsule / 'protocol.public.json')
    # Presence checks use filesystem metadata; test bytes are only read in the
    # frozen final test stage. Do not add decoding or SHA checks here for test.
    for relative, expected in public['file_fingerprints'].items():
        path = (data / relative).resolve()
        if not path.is_relative_to(data) or not path.is_file():
            raise FileNotFoundError('Missing private paired data: ' + relative)
        if '/test/' not in relative and sha(path) != expected:
            raise ValueError('Development data fingerprint mismatch: ' + relative)
    protocol = replace_roots(copy.deepcopy(public), data, output)
    now = datetime.now(timezone(timedelta(hours=8)))
    protocol['study_id'] += '_replay_' + now.strftime('%Y%m%d_%H%M%S')
    protocol['frozen_at'] = now.isoformat()
    for run in protocol['runs']:
        run['config']['experiment']['study'] = protocol['study_id']
    output.mkdir(parents=True)
    shutil.copytree(capsule / 'source_snapshot', output / 'code_snapshot')
    protocol['code_fingerprints'] = {p.relative_to(output / 'code_snapshot').as_posix(): sha(p)
                                     for p in sorted((output / 'code_snapshot').rglob('*')) if p.is_file()}
    write(output / 'protocol.json', protocol)
    (output / 'protocol.sha256').write_text(sha(output / 'protocol.json'), encoding='ascii')
    plan = load(capsule / 'analysis_plan.original.json')
    plan['training_protocol_sha256'] = sha(output / 'protocol.json')
    plan['frozen_at'] = now.isoformat()
    write(output / 'analysis_plan.json', plan)
    (output / 'analysis_plan.sha256').write_text(sha(output / 'analysis_plan.json'), encoding='ascii')
    write(output / 'replay_provenance.json', {
        'original_private_protocol_sha256': public['publication']['original_protocol_sha256'],
        'public_derivative_sha256': sha(capsule / 'protocol.public.json'),
        'new_replay_protocol_sha256': sha(output / 'protocol.json'),
        'same_data_split_hyperparameters_and_run_order': True,
        'test_bytes_read_during_preparation': False, 'new_independent_validation': False,
        'original_evidence_modified': False,
    })
    return {'prepared': True, 'output': str(output), 'new_independent_validation': False}


def run(study):
    study = study.resolve()
    # Refuse before touching any archived environment/lock/test records.
    if (study / 'test_evaluation_started.json').exists() or (study / 'test_summary.json').exists():
        raise FileExistsError('Final test already started; retain evidence and do not rerun')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
        raise ValueError('Recorded CUBLAS_WORKSPACE_CONFIG is :4096:8')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    runner_path = study / 'code_snapshot/run_study.py'
    protocol = load(study / 'protocol.json')
    if sha(runner_path) != protocol['code_fingerprints']['run_study.py']:
        raise ValueError('Frozen runner hash mismatch')
    import torch
    weights = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    if not weights.is_file() or sha(weights) != protocol['official_pretraining_weight_sha256']:
        raise ValueError('Pre-cache the exact recorded official ImageNet ResNet18 weights before replay')
    spec = importlib.util.spec_from_file_location('frozen_cross_patient_runner', runner_path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.run(study)


def summarize(study, output):
    if output.exists():
        raise FileExistsError('Numerical reanalysis must not overwrite existing evidence')
    if load(study / 'progress.json')['stage'] != 'complete':
        raise ValueError('Wait for frozen final test completion')
    protocol = load(study / 'protocol.json');validate_protocol(protocol)
    raw = {r['run_id']: {x['filename']: x for x in load(study / 'runs' / r['run_id'] / 'test/evaluation.json')['per_sample']}
           for r in protocol['runs']}
    write(output, {'conditions': aggregate(protocol, raw), 'test_patients': 1,
                   'scope': 'descriptive numeric reanalysis, not new independent validation'})
    return {'summarized': True, 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('verify-public', help='Audit numeric archive without private data or ML packages')
    check.add_argument('--capsule-root', type=Path, default=DEFAULT_CAPSULE)
    prep = sub.add_parser('prepare-replay', help='Freeze a new derived study against approved private paired data')
    prep.add_argument('--capsule-root', type=Path, default=DEFAULT_CAPSULE)
    prep.add_argument('--data-root', type=Path, required=True)
    prep.add_argument('--output-root', type=Path, required=True)
    train = sub.add_parser('run-replay', help='Run 24 fresh trainings, then the exclusively locked final test')
    train.add_argument('--study-root', type=Path, required=True)
    analysis = sub.add_parser('summarize', help='Recompute numbers only, without patient images or chart dependencies')
    analysis.add_argument('--study-root', type=Path, required=True)
    analysis.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'verify-public':result = verify(args.capsule_root)
    elif args.command == 'prepare-replay':result = prepare(args.capsule_root, args.data_root, args.output_root)
    elif args.command == 'run-replay':run(args.study_root);result = {'completed': True}
    else:result = summarize(args.study_root, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':main()
