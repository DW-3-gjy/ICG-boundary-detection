"""Sequential, frozen-config comparison with a separately invoked test stage."""
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import ICGDataset
from .eval import evaluate_source_split
from .training import Trainer
from .utils import load_config, write_json


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    records = manifest.get('records', [])
    if not records:
        raise ValueError('Split manifest requires records with filename, split, source_batch')
    groups, filenames = {}, set()
    for record in records:
        split = record['split']
        filename = record['filename']
        group = record['source_batch']
        if split not in ('train', 'val', 'test') or not group:
            raise ValueError('Invalid split or missing source batch')
        if filename in filenames:
            raise ValueError(f'Duplicate/cross-split filename: {filename}')
        filenames.add(filename)
        if group in groups and groups[group] != split:
            raise ValueError(f'Source group crosses splits: {group}')
        groups[group] = split
    if {r['split'] for r in records} != {'train', 'val', 'test'}:
        raise ValueError('Manifest must define train, val, and test')


def source_fingerprints(config, manifest, splits=('train', 'val')):
    """Training preflight never enumerates or reads test images."""
    fingerprints = {}
    for split in splits:
        dataset = ICGDataset(config['data'][f'{split}_data_dir'], is_train=False)
        expected = {r['filename'] for r in manifest['records'] if r['split'] == split}
        actual = {image.name for image, _ in dataset.pairs}
        if actual != expected:
            raise ValueError(f'{split} image names do not match the frozen manifest')
        fingerprints[split] = {image.name: {'image_sha256': digest_file(image), 'mask_sha256': digest_file(mask)}
                               for image, mask in dataset.pairs}
    return fingerprints


def assert_frozen_sources(config, manifest, expected, splits=('train', 'val')):
    label = 'Development' if splits == ('train', 'val') else 'Test'
    try:
        actual = source_fingerprints(config, manifest, splits)
    except (OSError, ValueError) as exc:
        raise ValueError(f'{label} data changed or became unavailable after the experiment was frozen') from exc
    if actual != expected:
        raise ValueError(f'{label} data changed after the experiment was frozen; results cannot be accepted')


def summarize_runs(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(f"{run['architecture']}/{run['variant']}", []).append(run)
    result = {}
    for key, entries in grouped.items():
        metrics = {}
        for metric in entries[0]['metrics']:
            if not metric.startswith('mean_'):
                continue
            values = np.array([r['metrics'].get(metric) if r['metrics'].get(metric) is not None
                               else float('inf') for r in entries], dtype=float)
            metrics[metric] = float(values.mean())
            metrics[f'seed_std_{metric}'] = float(values.std()) if np.isfinite(values).all() else None
        result[key] = {'seeds': [r['seed'] for r in entries], 'metrics': metrics, 'promising': None}
    for key, entry in result.items():
        architecture, variant = key.split('/')
        control = result.get(f'{architecture}/control')
        if control and variant != 'control' and entry['seeds'] == control['seeds']:
            a, b = entry['metrics'], control['metrics']
            entry['promising'] = bool(
                a['mean_hausdorff_95'] < b['mean_hausdorff_95']
                and a['mean_dice'] >= b['mean_dice']-.02
                and a['mean_recall'] >= b['mean_recall']-.02)
    return result


def write_metrics_csv(path, rows):
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(config_paths, seeds, data_root, manifest_path, output_root, epochs=None, device=None):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_path = output_root / 'frozen_experiments.json'
    if frozen_path.exists():
        raise FileExistsError(f'Use a new output root; experiment already frozen: {frozen_path}')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    validate_manifest(manifest)
    definitions, identities = [], set()
    for config_path in config_paths:
        template = load_config(str(config_path))
        architecture = template['model'].get('name', 'unet')
        variant = template.get('experiment', {}).get('variant',
                   {'dice_ce': 'control', 'tversky_ce': 'tversky', 'tversky_ce_boundary': 'combined'}[
                       template['training']['loss_function']])
        for seed in seeds:
            identity = (architecture, variant, int(seed))
            if identity in identities:
                raise ValueError(f'Duplicate experiment: {identity}')
            identities.add(identity)
            config = copy.deepcopy(template)
            run_dir = output_root / architecture / variant / f'seed_{seed}'
            config['data'].update({f'{split}_data_dir': str((Path(data_root) / split).resolve())
                                   for split in ('train', 'val', 'test')})
            config['training'].update(seed=int(seed), checkpoint_dir=str(run_dir / 'checkpoints'))
            if epochs is not None:
                config['training']['epochs'] = int(epochs)
                config['training']['scheduler_params']['T_max'] = int(epochs)
            if device is not None:
                config['device']['use_cuda'] = device == 'cuda'
            config['inference'].update(model_path=str(run_dir / 'checkpoints/best_model.pt'),
                                        output_dir=str(run_dir / 'predictions'))
            config['logging']['log_dir'] = str(run_dir / 'logs')
            config.setdefault('experiment', {})['split_manifest'] = manifest
            config['experiment']['split_manifest_sha256'] = digest_file(manifest_path)
            definitions.append({'architecture': architecture, 'variant': variant, 'seed': int(seed),
                                'run_dir': str(run_dir), 'config': config})
    if not definitions:
        raise ValueError('Select at least one configuration and seed')
    fingerprints = source_fingerprints(definitions[0]['config'], manifest)
    frozen = {'manifest': manifest, 'manifest_sha256': digest_file(manifest_path),
              'development_fingerprints': fingerprints, 'runs': definitions,
              'synthetic_is_not_clinical_validation': True}
    write_json(frozen_path, frozen)
    freeze_hash = digest_file(frozen_path)
    (output_root / 'frozen_experiments.sha256').write_text(freeze_hash, encoding='ascii')
    runs = []
    for definition in definitions:
        config, run_dir = definition['config'], Path(definition['run_dir'])
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / 'config.json', config)
        assert_frozen_sources(config, manifest, fingerprints)
        trainer = Trainer(config)
        trainer.train()
        assert_frozen_sources(config, manifest, fingerprints)
        write_json(run_dir / 'provenance.json', trainer.metadata | {'optimizer_updates': trainer.optimizer_updates,
                    'frozen_experiments_sha256': freeze_hash})
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        validation = evaluate_source_split(config, 'val', output_dir=run_dir / 'val')
        assert_frozen_sources(config, manifest, fingerprints)
        write_json(run_dir / 'val_evaluation.json', validation)
        write_metrics_csv(run_dir / 'val_per_image.csv', validation['per_sample'])
        run = {k: definition[k] for k in ('architecture', 'variant', 'seed', 'run_dir')}
        run.update(metrics=validation['aggregated'], checkpoint_sha256=digest_file(config['inference']['model_path']))
        runs.append(run)
        write_json(output_root / 'validation_summary.json', {'runs': runs, 'comparison': summarize_runs(runs),
                    'frozen_experiments_sha256': freeze_hash, 'complete': len(runs) == len(definitions)})
    return {'runs': runs, 'comparison': summarize_runs(runs), 'frozen_experiments_sha256': freeze_hash, 'complete': True}


def run_frozen_test(output_root):
    """Explicit final stage: no training or model/threshold selection here."""
    output_root = Path(output_root).resolve()
    frozen_path = output_root / 'frozen_experiments.json'
    expected_hash = (output_root / 'frozen_experiments.sha256').read_text(encoding='ascii').strip()
    if digest_file(frozen_path) != expected_hash:
        raise ValueError('Frozen experiment definitions changed; test evaluation refused')
    frozen = json.loads(frozen_path.read_text(encoding='utf8'))
    validation = json.loads((output_root / 'validation_summary.json').read_text(encoding='utf8'))
    if not validation.get('complete') or len(validation['runs']) != len(frozen['runs']):
        raise ValueError('Complete every frozen validation run before evaluating test')
    if validation.get('frozen_experiments_sha256') != expected_hash:
        raise ValueError('Validation summary belongs to a different frozen experiment')
    assert_frozen_sources(frozen['runs'][0]['config'], frozen['manifest'], frozen['development_fingerprints'])
    for definition, record in zip(frozen['runs'], validation['runs']):
        if digest_file(definition['config']['inference']['model_path']) != record['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed after validation')
    # Exclusive marker also prevents accidental repeated or concurrent test access.
    with (output_root / 'test_evaluation_started.json').open('x', encoding='utf8') as stream:
        json.dump({'frozen_experiments_sha256': expected_hash}, stream)
    test_fingerprints = source_fingerprints(frozen['runs'][0]['config'], frozen['manifest'], splits=('test',))
    write_json(output_root / 'test_fingerprints.json', test_fingerprints)
    runs = []
    for definition in frozen['runs']:
        run_dir = Path(definition['run_dir'])
        assert_frozen_sources(definition['config'], frozen['manifest'], test_fingerprints, splits=('test',))
        result = evaluate_source_split(definition['config'], 'test', output_dir=run_dir / 'test')
        assert_frozen_sources(definition['config'], frozen['manifest'], test_fingerprints, splits=('test',))
        write_json(run_dir / 'test_evaluation.json', result)
        write_metrics_csv(run_dir / 'test_per_image.csv', result['per_sample'])
        runs.append({k: definition[k] for k in ('architecture', 'variant', 'seed', 'run_dir')} |
                    {'metrics': result['aggregated']})
        write_json(output_root / 'test_summary.json', {'runs': runs, 'complete': len(runs) == len(frozen['runs']),
                                                      'frozen_experiments_sha256': expected_hash})
    return {'runs': runs, 'complete': True, 'frozen_experiments_sha256': expected_hash}
