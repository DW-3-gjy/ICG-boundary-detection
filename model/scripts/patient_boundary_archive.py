"""Verify anonymous study evidence with Python's standard library only.

This verifies recorded numbers and artifact identity. It cannot reproduce model
inference without the separately authorized private data and checkpoint files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / 'experiments/patient_boundary_20261105_01'
ALLOWED_SUFFIXES = {'.json', '.csv', '.md', '.py', '.yaml', '.yml', '.txt'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), 'Non-finite number: ' + label)
    require(abs(float(actual) - float(expected)) < 1e-11, 'Numeric mismatch: ' + label)


def privacy_scan(root):
    for path in Path(root).rglob('*'):
        if not path.is_file():
            continue
        require(path.suffix in ALLOWED_SUFFIXES, 'Non-public file type: ' + path.name)
        content = path.read_text(encoding='utf-8-sig')
        require(not re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home)/|xwechat_files|wxid_', content),
                'Local/private path found: ' + path.name)
        if path.suffix in {'.json', '.csv'}:
            require(not re.search(r'"(?:imageData|points|reference_points|packed_region)"\s*:', content),
                    'Image or coordinate payload found: ' + path.name)


def seal_archive(root):
    root = Path(root).resolve()
    privacy_scan(root)
    manifest = {p.relative_to(root).as_posix(): sha(p)
                for p in sorted(root.rglob('*'))
                if p.is_file() and p != root / 'manifest_sha256.json'}
    write(root / 'manifest_sha256.json', manifest)
    return {'files': len(manifest) + 1, 'bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}


def verify_hashes(root):
    root = Path(root).resolve()
    manifest = load(root / 'manifest_sha256.json')
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*')
                    if p.is_file() and p != root / 'manifest_sha256.json'}
    require(set(manifest) == actual_files, 'Manifest coverage differs from actual archive files')
    for relative, expected in manifest.items():
        target = (root / relative).resolve()
        require(target.is_relative_to(root), 'Manifest path escapes archive')
        require(sha(target) == expected, 'Archive hash mismatch: ' + relative)
    return len(manifest)


def verify_split(rows, protocol):
    require(len(rows) == 129, 'Expected 129 candidate frames')
    ids = [r['sample_id'] for r in rows]
    require(len(set(ids)) == len(ids), 'Duplicate anonymous sample ID')
    require(all(r['planned_partition'] in {'development', 'test'} for r in rows), 'Unknown partition')
    development = [r for r in rows if r['planned_partition'] == 'development']
    test = [r for r in rows if r['planned_partition'] == 'test']
    require((len(development), len(test)) == (106, 23), 'Incorrect development/test frame counts')
    dp = {r['patient_source_id'] for r in development}
    tp = {r['patient_source_id'] for r in test}
    require((len(dp), len(tp)) == (9, 3), 'Incorrect development/test patient counts')
    require(not dp & tp, 'Patient crosses development/test partitions')
    require(tp == set(protocol['fixed_new_batch_test_patient_sources']), 'Test patient identities differ from protocol')
    require(not {r['image_sha256'] for r in development} & {r['image_sha256'] for r in test},
            'Image hash crosses development/test partitions')
    for r in rows:
        for key in ('image_sha256', 'annotation_or_mask_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', r[key]) is not None, 'Invalid file hash: ' + r['sample_id'])
    folds = protocol['planned_development_validation_folds']
    require(set().union(*(set(v) for v in folds.values())) == dp, 'Validation folds do not cover development patients')
    require(sum(len(v) for v in folds.values()) == len(dp), 'Patient appears in multiple validation folds')
    for r in development:
        require(r['patient_source_id'] in folds[r['planned_validation_fold']], 'Frame assigned to wrong patient fold')
    require(Counter(r['task_stratum'] for r in test) == {'Z_label': 15, 'L_visible_fluorescence': 8},
            'Unexpected primary test strata')
    require(Counter(r['annotation_kind'] for r in test) == {'linestrip': 12, 'polygon': 11},
            'Unexpected test reference types')
    return development, test


def patient_macro(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None or value == '':
            continue
        value = float(value)
        require(math.isfinite(value), 'Non-finite metric: ' + metric)
        require(0 <= value <= 1, 'Out-of-range metric: ' + metric)
        grouped[row['patient']].append(value)
    require(bool(grouped), 'No evaluable references: ' + metric)
    means = {patient: statistics.mean(values) for patient, values in grouped.items()}
    return statistics.mean(means.values()), means


def eligible(rows, group, fold):
    return [r for r in rows if r['planned_partition'] == 'development'
            and (fold == 'refit' or r['planned_validation_fold'] != fold)
            and (group['include_old_training'] or r['batch'] == '1102')
            and (group['include_l_training'] or r['task_stratum'] == 'Z_label')]


def verify_archive(root):
    root = Path(root).resolve()
    privacy_scan(root)
    files = verify_hashes(root)
    p = load(root / 'protocol.public.json')
    freeze = load(root / 'freeze.public.json')
    original_protocol_sha = p['publication']['original_protocol_sha256']
    require(original_protocol_sha == freeze['protocol_sha256'], 'Original protocol identity mismatch')
    require(p['publication']['kind'] == 'path_redacted_derivative_not_original_frozen_protocol', 'Missing derivative disclosure')
    require(p['user_confirmations_20261105']['old_and_new_patients_all_distinct'], 'Missing identity confirmation')
    require(p['user_confirmations_20261105']['all_requested_visible_boundaries_annotated'], 'Missing reference confirmation')
    require(p['seeds'] == [42, 43, 44], 'Unexpected seed design')
    require(p['development_updates'] == 2400 and p['validation_interval_updates'] == 40, 'Unexpected training budget')
    require(sha(root / 'split_manifest.csv') == freeze['candidate_manifest_sha256'], 'Original anonymous candidate manifest identity changed')
    require(sha(root / 'source_snapshot_manifest.json') == freeze['source_snapshot_manifest_sha256'], 'Original source manifest identity changed')
    require(sha(root / 'cache_manifest.json') == freeze['cache_manifest_sha256'], 'Original anonymous cache manifest identity changed')
    records = read_csv(root / 'split_manifest.csv')
    development, test = verify_split(records, p)
    for rel, expected in load(root / 'source_snapshot_manifest.json').items():
        source = (root / 'source_snapshot' / rel).resolve()
        require(source.is_relative_to(root / 'source_snapshot'), 'Source path escapes snapshot')
        require(sha(source) == expected, 'Frozen source identity mismatch: ' + rel)
    run_records = {}
    shared = defaultdict(set)
    expected_ids = set()
    expected_folds = list(p['planned_development_validation_folds'])
    groups = {g['id']: g for g in p['groups']}
    require(set(groups) == set('ABCDE'), 'Unexpected experimental groups')
    for group in groups.values():
        for fold in expected_folds + ['refit']:
            for seed in p['seeds']:
                tag = f"{group['id']}_{fold}_s{seed}"
                expected_ids.add(tag)
                folder = root / 'runs' / tag
                config, complete = load(folder / 'config.json'), load(folder / 'complete.json')
                init, history = load(folder / 'initialization.json'), load(folder / 'history.json')
                require(config['protocol_sha256'] == original_protocol_sha, 'Run protocol mismatch: ' + tag)
                require(config['group'] == group and config['seed'] == seed and config['fold'] == fold, 'Run configuration identity mismatch: ' + tag)
                require(config['effective_batch'] == 4 and config['microbatch'] == 4, 'Changed batch size: ' + tag)
                require(complete['run'] == tag and complete['group'] == group['id'] and complete['fold'] == fold and complete['seed'] == seed,
                        'Completion identity mismatch: ' + tag)
                expected_training = {r['sample_id'] for r in eligible(records, group, fold)}
                require(set(config['training_ids']) == expected_training and len(config['training_ids']) == len(expected_training),
                        'Run training data mismatch: ' + tag)
                expected_validation = {r['sample_id'] for r in development if r['planned_validation_fold'] == fold}
                require(set(config['validation_ids']) == expected_validation and len(config['validation_ids']) == len(expected_validation),
                        'Run validation data mismatch: ' + tag)
                require(complete['training_candidates'] == len(expected_training), 'Training candidate count mismatch: ' + tag)
                require(init['seed'] == seed and not init['warm_start_icg_checkpoint'], 'Unexpected ICG warm start: ' + tag)
                require(init['encoder_weights_sha256'] == p['official_encoder_weights_sha256'], 'Encoder initialization mismatch: ' + tag)
                shared[seed].add(init['shared_parameter_sha256'])
                require(complete['updates'] == config['max_updates'], 'Completion budget mismatch: ' + tag)
                require([x['update'] for x in history] == list(range(40, complete['updates'] + 1, 40)), 'Incomplete history: ' + tag)
                for row in history:
                    require(math.isfinite(row['training_loss']) and math.isfinite(row['gradient_norm']), 'Non-finite training history: ' + tag)
                if fold != 'refit':
                    require(complete['updates'] == 2400, 'Development run shortened: ' + tag)
                    for row in history:
                        grid = {float(t): float(score) for t, score in row['validation_grid'].items()}
                        require(set(grid) == set(p['boundary_probability_threshold_grid']), 'Threshold grid changed: ' + tag)
                        chosen = max(grid, key=lambda t: (grid[t], -abs(t - .5), -t))
                        close(row['threshold'], chosen, tag + ' threshold choice')
                        close(row['validation_patient_macro_f1'], grid[chosen], tag + ' validation score')
                    first_best = max(history, key=lambda row: row['validation_patient_macro_f1'])
                    require(complete['best_update'] == first_best['update'], 'Selected update is not first validation maximum: ' + tag)
                    close(complete['best_threshold'], first_best['threshold'], tag + ' selected threshold')
                    close(complete['best_validation_f1'], first_best['validation_patient_macro_f1'], tag + ' selected validation score')
                else:
                    require(complete['best_validation_f1'] is None and not config['validation_ids'], 'Final refit selected on validation/test: ' + tag)
                    require(complete['best_update'] == complete['updates'], 'Final refit did not retain final update: ' + tag)
                run_records[tag] = complete
    actual_ids = {folder.name for folder in (root / 'runs').iterdir() if folder.is_dir()}
    require(actual_ids == expected_ids, 'Missing or extra experimental runs')
    require(all(len(hashes) == 1 for hashes in shared.values()), 'Shared initialization differs within a seed')
    lock = load(root / 'checkpoint_lock.public.json')
    require(lock['protocol_sha256'] == original_protocol_sha, 'Checkpoint lock protocol mismatch')
    require(len(lock['checkpoints']) == 15, 'Expected 15 final checkpoints')
    started = load(root / 'final_test_started.json')
    require(started['checkpoint_lock_sha256'] == lock['publication']['original_checkpoint_lock_sha256'], 'Test did not use original checkpoint lock')
    require(lock['locked_at'] < started['time'], 'Test began before final checkpoint lock')
    require(set(r['run'] for r in lock['checkpoints']) == {r for r in run_records if '_refit_' in r}, 'Locked final run identities differ')
    for group in groups:
        selected = [r for r in run_records.values() if r['group'] == group and r['fold'] != 'refit']
        budget = max(40, min(2400, round(statistics.median(r['best_update'] for r in selected) / 40) * 40))
        threshold = statistics.median(r['best_threshold'] for r in selected)
        for item in (r for r in lock['checkpoints'] if r['group'] == group):
            require(item == run_records[item['run']], 'Locked record differs from completed checkpoint: ' + item['run'])
            require(item['updates'] == budget, 'Refit did not use development median budget: ' + item['run'])
            close(item['best_threshold'], threshold, item['run'] + ' median threshold')
    final, analysis = load(root / 'final_summary.json'), load(root / 'analysis/report_data.public.json')
    test_by_id = {r['sample_id']: r for r in test}
    aggregate_values = defaultdict(lambda: defaultdict(list))
    strata_values = defaultdict(lambda: defaultdict(list))
    patient_values = defaultdict(lambda: defaultdict(list))
    for item in lock['checkpoints']:
        tag, group = item['run'], item['group']
        rows = read_csv(root / 'runs' / tag / 'test/per_image.csv')
        require(len(rows) == len(test) and {r['sample_id'] for r in rows} == set(test_by_id), 'Test frame set mismatch: ' + tag)
        for row in rows:
            record = test_by_id[row['sample_id']]
            require(row['patient'] == record['patient_source_id'] and row['stratum'] == record['task_stratum']
                    and row['reference_kind'] == record['annotation_kind'], 'Test frame identity mismatch: ' + tag)
            region = row['has_region_reference'] == 'True'
            require(region == (row['reference_kind'] == 'polygon'), 'Invalid region reference eligibility: ' + tag)
            require(region or (row['dice'] == '' and row['iou'] == ''), 'Invented region metric on an open curve: ' + tag)
            if row['empty_prediction'] == 'True':
                require(float(row['f1']) == 0 and row['hd95'] == '', 'Empty prediction not preserved: ' + tag)
        evaluation = load(root / 'runs' / tag / 'test/evaluation.json')
        evaluation_rows = {r['sample_id']: r for r in evaluation['per_image']}
        require(len(evaluation['per_image']) == len(test) and set(evaluation_rows) == set(test_by_id), 'Evaluation image set mismatch: ' + tag)
        for row in rows:
            expected = evaluation_rows[row['sample_id']]
            for field in ('patient', 'stratum', 'reference_kind'):
                require(row[field] == expected[field], 'Evaluation frame metadata differs: ' + tag)
            for field in ('has_region_reference', 'empty_prediction'):
                require(row[field] in {'True', 'False'} and (row[field] == 'True') == expected[field],
                        'Evaluation frame flag differs: ' + tag)
            for column in ('precision', 'recall', 'f1', 'f1_10px', 'f1_40px', 'hd95', 'assd',
                           'directed_reference_mean', 'directed_reference_p95', 'dice', 'iou'):
                if expected[column] is None:
                    require(row[column] == '', 'CSV missingness differs from evaluation: ' + tag)
                else:
                    close(row[column], expected[column], tag + ' per-image ' + column)
            precision, recall = float(row['precision']), float(row['recall'])
            require(0 <= precision <= 1 and 0 <= recall <= 1, 'Invalid precision/recall: ' + tag)
            close(row['f1'], 2 * precision * recall / (precision + recall) if precision + recall else 0,
                  tag + ' F1 harmonic mean')
            require(float(row['f1_10px']) <= float(row['f1']) + 1e-12 <= float(row['f1_40px']) + 2e-12,
                    'Boundary F1 tolerance ordering inconsistent: ' + tag)
            if row['dice'] != '':
                close(row['iou'], float(row['dice']) / (2 - float(row['dice'])), tag + ' Dice/IoU relation')
        f1, patients = patient_macro(rows, 'f1')
        close(f1, evaluation['patient_macro_f1'], tag + ' evaluation macro')
        require(set(patients) == set(evaluation['per_patient']), 'Evaluation patients differ: ' + tag)
        for patient, value in patients.items():
            close(value, evaluation['per_patient'][patient], tag + ' patient macro')
            patient_values[group][patient].append(value)
        original_run = next(r for r in final['groups'][group]['runs'] if r['run'] == tag)
        close(f1, original_run['boundary_f1'], tag + ' final summary')
        report_run = next(r for r in analysis['groups'][group]['runs'] if r['run'] == tag)
        empty = sum(r['empty_prediction'] == 'True' for r in rows)
        require(empty == report_run['empty_predictions'] == original_run['empty_predictions'], 'Empty prediction count mismatch: ' + tag)
        require(sum(r['hd95'] == '' for r in rows) == report_run['undefined_hd95'], 'Undefined distance count mismatch: ' + tag)
        require(sum(r['has_region_reference'] == 'True' for r in rows) == report_run['region_reference_frames'] == 11,
                'Region reference count mismatch: ' + tag)
        for metric, column in [('boundary_f1', 'f1'), ('f1_10px', 'f1_10px'), ('f1_40px', 'f1_40px'), ('dice', 'dice')]:
            value, _ = patient_macro(rows, column)
            aggregate_values[group][metric].append(value)
            close(value, report_run[metric], tag + ' analysis ' + metric)
        for stratum, selected_rows in {
                'Z_label': [r for r in rows if r['stratum'] == 'Z_label'],
                'L_visible_fluorescence': [r for r in rows if r['stratum'] == 'L_visible_fluorescence'],
                'linestrip': [r for r in rows if r['reference_kind'] == 'linestrip'],
                'region_reference': [r for r in rows if r['has_region_reference'] == 'True']}.items():
            value, _ = patient_macro(selected_rows, 'f1')
            strata_values[group][stratum].append(value)
    for group, metrics in aggregate_values.items():
        for metric, values in metrics.items():
            close(statistics.mean(values), analysis['groups'][group]['metrics'][metric]['mean'], group + ' mean ' + metric)
            close(statistics.stdev(values), analysis['groups'][group]['metrics'][metric]['seed_sd'], group + ' sample SD ' + metric)
        close(statistics.mean(metrics['boundary_f1']), final['groups'][group]['boundary_f1_mean'], group + ' final F1 mean')
        close(statistics.stdev(metrics['boundary_f1']), final['groups'][group]['boundary_f1_seed_sd'], group + ' final F1 SD')
        for stratum, values in strata_values[group].items():
            close(statistics.mean(values), analysis['groups'][group]['strata'][stratum]['mean'], group + ' stratum mean ' + stratum)
            close(statistics.stdev(values), analysis['groups'][group]['strata'][stratum]['seed_sd'], group + ' stratum sample SD ' + stratum)
        for patient, values in patient_values[group].items():
            close(statistics.mean(values), analysis['groups'][group]['per_patient_seed_mean'][patient], group + ' patient seed mean')
    for contrast in analysis['contrasts']:
        left, right = contrast['contrast'].split('-')
        close(contrast['mean_delta'], statistics.mean(aggregate_values[left]['boundary_f1']) - statistics.mean(aggregate_values[right]['boundary_f1']),
              contrast['contrast'] + ' contrast mean')
        for patient in patient_values[left]:
            close(contrast['per_patient_delta'][patient], statistics.mean(patient_values[left][patient]) - statistics.mean(patient_values[right][patient]),
                  contrast['contrast'] + ' patient contrast')
        require(contrast['patients_improved'] == sum(v > 0 for v in contrast['per_patient_delta'].values()), 'Incorrect improved patient count')
    return {'passed': True, 'hashed_files': files, 'candidate_frames': 129, 'development_patients': 9,
            'test_patients': 3, 'test_frames': 23, 'development_runs': 45, 'final_refits': 15,
            'test_csv_files_reaggregated': 15, 'region_reference_frames': 11,
            'scope': 'anonymous recorded evidence and arithmetic; no private pixels, checkpoints or inference replay'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify-public', 'seal-public'])
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = verify_archive(args.archive) if args.command == 'verify-public' else seal_archive(args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
