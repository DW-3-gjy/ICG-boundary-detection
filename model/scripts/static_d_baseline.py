"""Prepare and use the original, frozen November D-only training baseline."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from statistics import median
import time

MODEL = Path(__file__).resolve().parents[1]
LOGS = MODEL / 'logs'
ORIGINAL = LOGS / 'patient_boundary_20261105_01'
DEFAULT_STUDY = LOGS / 'static_d_baseline_20261123_01'
SEEDS = (42, 43, 44)
FROZEN_METADATA = ('protocol.json', 'candidate_manifest.csv', 'cache_manifest.json', 'source_snapshot_manifest.json')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_file(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Frozen file changed: {path}')
    return actual


def clone_file(source, destination, expected):
    verify_file(source, expected)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f'{destination.name}.{os.getpid()}.part')
        shutil.copyfile(source, temporary)
        verify_file(temporary, expected)
        temporary.replace(destination)
    verify_file(destination, expected)


def verified_files():
    frozen = read_json(ORIGINAL / 'freeze.json')
    files = {'freeze.json': digest(ORIGINAL / 'freeze.json')}
    for name in FROZEN_METADATA:
        files[name] = verify_file(ORIGINAL / name, frozen[f'{Path(name).stem}_sha256'])
    for name, expected in read_json(ORIGINAL / 'source_snapshot_manifest.json').items():
        files[f'code/{name}'] = expected
    dev_cache = {name: expected for name, expected in read_json(ORIGINAL / 'cache_manifest.json').items()
                 if name.startswith('data/development/')}
    if len(dev_cache) != 106:
        raise ValueError('The frozen D baseline requires 106 development frames')
    return files | dev_cache


def original_checkpoints():
    lock = read_json(ORIGINAL / 'checkpoint_lock.json')
    if lock['protocol_sha256'] != digest(ORIGINAL / 'protocol.json'):
        raise ValueError('Original checkpoint lock does not match its protocol')
    checkpoints = {}
    for entry in lock['checkpoints']:
        if entry['group'] != 'D':
            continue
        path = ORIGINAL / 'runs' / entry['run'] / 'best.pt'
        if (entry['fold'], entry['updates'], entry['best_threshold'], entry['training_candidates']) != ('refit', 2160, .8, 106):
            raise ValueError('Original D checkpoint does not match the frozen recipe')
        checkpoints[str(entry['seed'])] = dict(path=path.relative_to(MODEL).as_posix(),
                                                sha256=verify_file(path, entry['checkpoint_sha256']))
    if set(checkpoints) != {str(seed) for seed in SEEDS}:
        raise ValueError('All three original D reference checkpoints are required')
    return checkpoints


def verify(study):
    for name, expected in verified_files().items():
        verify_file(study / name, expected)
    with (study / 'candidate_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        development = [row for row in csv.DictReader(stream) if row['planned_partition'] == 'development']
    patients = {row['patient_source_id'] for row in development}
    if len(development) != 106 or len(patients) != 9:
        raise ValueError('Unexpected D cohort')
    if set(row['planned_validation_fold'] for row in development) != {'fold_1', 'fold_2', 'fold_3'}:
        raise ValueError('Unexpected D development folds')
    receipt = read_json(study / 'baseline.json')
    if receipt['reference_checkpoints'] != original_checkpoints() or receipt['protocol_sha256'] != digest(study / 'protocol.json'):
        raise ValueError('Active D references do not match the frozen archive')
    return development, receipt


def prepare(study):
    study.mkdir(exist_ok=True)
    for name, expected in verified_files().items():
        clone_file(ORIGINAL / name, study / name, expected)
    receipt = {'group': 'D', 'original_study': ORIGINAL.name, 'protocol_sha256': digest(study / 'protocol.json'),
               'development_frames': 106, 'development_patients': 9, 'augmentation_and_selection': 'original frozen protocol',
               'reference_checkpoints': original_checkpoints(), 'fixed_display_seed': 42,
               'archived_separate_experiment': 'static_icg_20261123_01'}
    path = study / 'baseline.json'
    if path.exists() and read_json(path) != receipt:
        raise ValueError('An existing baseline receipt has a different identity')
    if not path.exists():
        temporary = path.with_name(f'baseline.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    return verify(study)


def load_runner(study):
    path = study / 'code/scripts/patient_boundary_study.py'
    spec = importlib.util.spec_from_file_location('frozen_patient_boundary_study', path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure()
    module.forbid_test_access(study)
    module.forbid_test_access(ORIGINAL)
    return module


def smoke(study, development, receipt):
    runner = load_runner(study)
    protocol = read_json(study / 'protocol.json')
    encoder = Path(runner.torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    verify_file(encoder, protocol['official_encoder_weights_sha256'])
    sid = development[0]['sample_id']
    with runner.np.load(study / 'data/development' / (sid + '.npz'), allow_pickle=False) as sample:
        image = runner.normalized_image(sample['image'])[None]
    device = 'cuda' if runner.torch.cuda.is_available() else 'cpu'
    image = image.to(device)
    for seed, entry in receipt['reference_checkpoints'].items():
        checkpoint = runner.torch.load(MODEL / entry['path'], map_location='cpu', weights_only=True)
        model = runner.MixedBoundaryUNet(True, head_seed=0, encoder_weights=None)
        model.load_state_dict(checkpoint['model'], strict=True)
        if checkpoint['threshold'] != .8 or checkpoint['update'] != 2160:
            raise ValueError('Unexpected D checkpoint settings')
        model = model.to(device).eval()
        with runner.torch.inference_mode():
            output = model(image)
        for head in ('logits', 'boundary_logits'):
            if output[head].shape[-2:] != (432, 768) or not runner.torch.isfinite(output[head]).all():
                raise ValueError(f'Invalid restored D output: {seed}/{head}')
    print(f'All three D checkpoints passed a development-frame forward check on {device}.')


@contextmanager
def exclusive_training(study):
    with (study / '.train.lock').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def train(study, development, receipt):
    with exclusive_training(study):
        runner = load_runner(study)
        if not runner.torch.cuda.is_available():
            raise RuntimeError('CUDA is required by the frozen D training runner')
        protocol = read_json(study / 'protocol.json')
        group = next(g for g in protocol['groups'] if g['id'] == 'D')
        if (study / 'final_test_started.json').exists():
            raise ValueError('This study has already started final evaluation')
        started = study / 'training_started.json'
        if not started.exists():
            runner.atomic_json(started, {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                         'protocol_sha256': receipt['protocol_sha256'], 'group': 'D'})

        def fit(seed, fold, updates, threshold=None):
            result = runner.train_one(study, protocol, development, group, seed, fold, updates, threshold, microbatch=4)
            verify_file(study / 'runs' / result['run'] / 'best.pt', result['checkpoint_sha256'])
            if (result['group'], result['seed'], result['fold'], result['updates']) != ('D', seed, fold, updates):
                raise ValueError('Unexpected training receipt')
            if threshold is not None and result['best_threshold'] != threshold:
                raise ValueError('Unexpected D refit threshold')
            return result

        rows = [fit(seed, fold, protocol['development_updates'])
                for fold in protocol['planned_development_validation_folds'] for seed in SEEDS]
        runner.atomic_json(study / 'development_summary.json', rows)
        updates = max(40, min(2400, 40 * round(median(r['best_update'] for r in rows) / 40)))
        threshold = float(median(r['best_threshold'] for r in rows))
        refits = [fit(seed, 'refit', updates, threshold) for seed in SEEDS]
        lock = study / 'checkpoint_lock.json'
        if not lock.exists():
            runner.atomic_json(lock, {'checkpoints': refits, 'protocol_sha256': receipt['protocol_sha256'],
                                      'locked_at': time.strftime('%Y-%m-%d %H:%M:%S')})
        elif read_json(lock)['checkpoints'] != refits:
            raise ValueError('Existing lock has different refit checkpoints')
        print('Nine D development runs and three refits complete and locked.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'train'))
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--smoke', action='store_true', help='Load the original D checkpoints and forward one development frame')
    args = parser.parse_args()
    study = args.study.resolve()
    if study.parent != LOGS.resolve() or study == ORIGINAL.resolve():
        parser.error('Use a distinct study directory directly under model/logs')
    if args.command == 'prepare':
        development, receipt = prepare(study)
    else:
        development, receipt = verify(study)
    print(f'D baseline ready: {len(development)} development frames from {receipt["development_patients"]} patients.')
    if args.smoke:
        smoke(study, development, receipt)
    if args.command == 'train':
        train(study, development, receipt)


if __name__ == '__main__':
    main()
