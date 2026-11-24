"""Export anonymous numeric evidence, never patient pixels/coordinates/weights.

This is a path-redacted derivative, not the original frozen protocol. The
original study is read only; an existing export directory is never overwritten.
"""
import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def redact(value, data_root, study_root):
    if isinstance(value, dict):
        return {k: redact(v, data_root, study_root) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, data_root, study_root) for v in value]
    if isinstance(value, str):
        for root, replacement in [(data_root, '${DATA_ROOT}'), (study_root, '${STUDY_ROOT}')]:
            value = value.replace(root, replacement).replace(root.replace('\\', '/'), replacement)
        return value.replace('\\', '/') if value.startswith('${') else value
    return value


def export(study, history, statistics_audit, output, analysis_script=None):
    study, history, output = study.resolve(), history.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('Use a new export directory; published evidence is not overwritten')
    for source in (study, history):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Export must be separate from private study records')
    protocol = load(study / 'protocol.json')
    original_sha = (study / 'protocol.sha256').read_text().strip()
    if sha(study / 'protocol.json') != original_sha:
        raise ValueError('Frozen protocol hash mismatch')
    if load(study / 'progress.json')['stage'] != 'complete':
        raise ValueError('Only completed studies can be exported')
    independent = load(study / 'analysis/independent_verification.json')
    audit = load(statistics_audit)
    if not independent['passed'] or not audit['passed']:
        raise ValueError('Independent audits have not passed')
    lock = load(study / 'checkpoint_lock.json')
    for run, expected in lock['checkpoints'].items():
        if sha(study / 'runs' / run / 'checkpoints/best_model.pt') != expected:
            raise ValueError('Checkpoint changed: ' + run)
    output.mkdir(parents=True)
    data_root, study_root = protocol['data_root'], protocol['study_root']
    public = redact(copy.deepcopy(protocol), data_root, study_root)
    public['publication'] = {
        'kind': 'path_redacted_derivative_not_original_frozen_protocol',
        'original_protocol_sha256': original_sha,
        'original_analysis_plan_sha256': sha(study / 'analysis_plan.json'),
        'original_source_fingerprints': protocol['code_fingerprints'],
        'private_data_included': False, 'weights_included': False,
        'replay_is_not_new_independent_validation': True,
    }
    # Archive exactly the training modules/configs and runner used. The private
    # intake/preparation script is intentionally excluded from this snapshot.
    for relative, expected in protocol['code_fingerprints'].items():
        if not (relative.startswith(('model/src/', 'model/configs/')) or relative == 'run_study.py'):
            continue
        source = study / 'code_snapshot' / relative
        if sha(source) != expected:
            raise ValueError('Frozen source changed: ' + relative)
        dest = output / 'source_snapshot' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    public['code_fingerprints'] = {
        p.relative_to(output / 'source_snapshot').as_posix(): sha(p)
        for p in sorted((output / 'source_snapshot').rglob('*')) if p.is_file()
    }
    write(output / 'protocol.public.json', public)
    plan = load(study / 'analysis_plan.json')
    analysis_source = analysis_script or (Path(__file__).resolve().parents[1] /
                                         'experiments/cross_patient_20261022_01/source_snapshot/analyze_study.py')
    if not analysis_source.exists() or sha(analysis_source) != plan['script_sha256']:
        raise ValueError('Exact original analysis source required for export')
    shutil.copyfile(analysis_source, output / 'source_snapshot/analyze_study.py')
    write(output / 'analysis_plan.original.json', plan)
    write(output / 'checkpoint_lock.json', lock)
    write(output / 'environment.json', load(study / 'environment.json'))
    write(output / 'summary.json', load(study / 'analysis/summary.json'))
    write(output / 'verification/independent_numeric_audit.json', independent)
    audit_keys = ['passed', 'reporting_revision_only', 'no_training_or_inference_executed',
                  'same_primary_test_filenames_verified', 'test_patients', 'test_frames',
                  'all_24_checkpoint_hashes_match_frozen_lock', 'all_24_run_csv_primary_aggregates_checked',
                  'maximum_csv_aggregate_error', 'independently_recomputed_saved_masks',
                  'independently_recomputed_metrics', 'maximum_absolute_mask_metric_error',
                  'primary_reporting_rule', 'precision_empty_prediction_convention',
                  'SD_is_not_patient_uncertainty', 'historical_only_one_run_no_SD',
                  'seed42_RB_is_highest_of_three_and_not_overall_estimate',
                  'historical_comparison_posthoc_no_new_blind_test', 'no_patient_level_significance_claim', 'deltas']
    write(output / 'verification/reporting_statistics_audit.json',
          {k: audit[k] for k in audit_keys} | {'private_audit_source_sha256': sha(statistics_audit)})
    # Store records without any original acquisition filename or directory.
    primary = set(protocol['test_subgroups']['primary_Z_all'])
    fields = ['filename', 'split', 'patient', 'source_group', 'phase', 'original_label',
              'occlusion', 'original_experiment_split', 'primary_test', 'image_sha256', 'mask_sha256']
    with (output / 'split_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in protocol['records']:
            row = {k: r.get(k, '') for k in fields}
            row['primary_test'] = r['filename'] in primary
            row['image_sha256'] = r['files']['fullres_images']['sha256']
            row['mask_sha256'] = r['files']['fullres_masks']['sha256']
            writer.writerow(row)
    run_summary = {r['run_id']: r for r in load(study / 'validation_summary.json')['runs']}
    prediction_hashes = {}
    for run in protocol['runs']:
        run_id = run['run_id']
        folder = study / 'runs' / run_id
        dest = output / 'runs' / run_id
        for name in ['config.json', 'initialization.json', 'provenance.json', 'checkpoints/history.json']:
            write(dest / name, redact(load(folder / name), data_root, study_root))
        for split in ['val', 'test']:
            # Only anonymous IDs and scalar metrics; no masks or overlays.
            write(dest / split / 'evaluation.json', load(folder / split / 'evaluation.json'))
            shutil.copyfile(folder / split / 'per_image.csv', dest / split / 'per_image.csv')
        prediction_hashes[run_id] = {
            p.name: sha(p) for p in sorted((folder / 'test/masks').glob('*.png'))
        }
    write(output / 'validation_summary.json', load(study / 'validation_summary.json'))
    write(output / 'test_prediction_hashes.json', prediction_hashes)
    write(output / 'run_summary.json', run_summary)
    hp = load(history / 'protocol.json')
    for key, record in hp['weights'].items():
        if sha(record['path']) != record['sha256']:
            raise ValueError('Historical comparison checkpoint changed: ' + key)
    write(output / 'historical_comparison/summary.json', load(history / 'summary.json'))
    write(output / 'historical_comparison/checkpoints.json', {
        key: {'sha256': record['sha256']} for key, record in hp['weights'].items()
    } | {'original_protocol_sha256': sha(history / 'protocol.json')})
    replay = load(history / 'historical_replay_audit.json')
    if not replay['all_five_historical_test_masks_reproduced_exactly']:
        raise ValueError('Historical replay failed')
    write(output / 'historical_comparison/replay_audit.json', {
        'replay_exact_old_five_masks': replay['all_five_historical_test_masks_reproduced_exactly'],
        'exact_replay_details': replay,
        'source_audit_sha256': sha(history / 'historical_replay_audit.json'),
        'mask_metric_audit': load(history / 'independent_verification.json'),
    })
    old_config = load(history / 'historical_checkpoint_config.json')
    path_fields = {'train_data_dir', 'val_data_dir', 'test_data_dir', 'checkpoint_dir',
                   'model_path', 'output_dir', 'log_dir', 'split_manifest_path', 'source_manifest'}
    old_config = {section: {k: v for k, v in values.items() if k not in path_fields}
                  for section, values in old_config.items()
                  if isinstance(values, dict) and section in {'model', 'data', 'training', 'inference', 'tracking', 'evaluation', 'device'}}
    old_config['publication'] = {'private_path_fields_omitted': True,
                                 'original_configuration_sha256': sha(history / 'historical_checkpoint_config.json')}
    write(output / 'historical_comparison/non_path_configuration.json', old_config)
    cases = []
    for row in load(history / 'per_sample_results.json'):
        cases.append({k: row[k] for k in ['filename', 'index', 'subgroup']} | {
            'seed': 42, 'models': {k: {m: v[m] for m in ['mask_sha256', 'metrics', 'empty_prediction']}
                                  for k, v in row['models'].items()}
        })
    write(output / 'historical_comparison/per_sample_results.json', cases)
    for path in output.rglob('*'):
        if path.is_file() and path.suffix not in {'.py', '.yaml', '.yml', '.json', '.csv'}:
            raise ValueError('Unexpected export file type: ' + path.name)
        if path.is_file() and path.suffix in {'.json', '.csv'}:
            content = path.read_text(encoding='utf-8-sig')
            # Path redaction is checked rather than assumed.
            if re.search(r'[A-Za-z]:[\\/]|/(?:home|Users)/|imageData|"points"\s*:', content):
                raise ValueError('Private path or annotation payload in export: ' + str(path))
    manifest = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output / 'manifest_sha256.json', manifest)
    print(json.dumps({'exported': True, 'files': len(manifest) + 1, 'runs': len(protocol['runs']),
                      'records': len(protocol['records']), 'original_protocol_sha256': original_sha,
                      'patient_pixels_coordinates_or_weights_included': False}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--historical-root', type=Path, required=True)
    parser.add_argument('--statistics-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--analysis-script', type=Path, help='Exact predeclared analysis source; defaults to the archived study script')
    args = parser.parse_args()
    export(args.study_root, args.historical_root, args.statistics_audit, args.output, args.analysis_script)
