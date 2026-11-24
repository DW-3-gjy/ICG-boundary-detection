"""Evaluate frozen D region predictions with model, flip and scale averaging."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import torch

from static_d_baseline import (
    DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, load_runner, read_json, verify, verify_file,
)
from train_static_d_region_dice import coarse_checkpoint, get_region_probability
from train_static_d_region_refiner import (
    DEFAULT_STUDY as REGION_STUDY, FOLDS, SIZE, dice_at_original_size, read_frames,
)

SEEDS = (42, 43, 44)
HIGH_SIZE = (1024, 576)
ARMS = ('single', 'horizontal_flip', 'scale_and_flip', 'ensemble', 'ensemble_scale_and_flip')
ADAPTED_ARMS = ('adapted_high_ensemble', 'ensemble_with_adapted_high')


def flip_pair(net, image, runner, original=None):
    original = get_region_probability(net, image, runner) if original is None else original
    mirrored = get_region_probability(net, cv2.flip(image, 1), runner)[:, ::-1]
    return (original + mirrored) * .5


def predict_frame(net, frame, large, runner):
    native = get_region_probability(net, frame.old_rgb, runner)
    native_pair = flip_pair(net, frame.old_rgb, runner, native)
    high_pair = flip_pair(net, large, runner)
    multiscale = (cv2.resize(native_pair, SIZE, interpolation=cv2.INTER_LINEAR)
                  + cv2.resize(high_pair, SIZE, interpolation=cv2.INTER_LINEAR)) * .5
    return native, native_pair, multiscale


def adapted_checkpoint(study, fold, seed, coarse):
    protocol = read_json(study / 'protocol.json')
    folder = study / 'runs' / f'G_{fold}_s{seed}'
    receipt = read_json(folder / 'complete.json')
    path = folder / 'foreground.pt'
    if (protocol['seeds'] != [seed] or protocol['size'] != list(HIGH_SIZE) or
            protocol['base_checkpoints'][fold] != digest(coarse) or
            (receipt['seed'], receipt['fold'], receipt['updates']) != (seed, fold, protocol['development_updates'])):
        raise ValueError(f'Mismatched adapted checkpoint for {fold}, seed {seed}')
    verify_file(path, receipt['checkpoint_sha256'])
    return path


def evaluate_fold(runner, development, frames, large, net, fold, by_patient, adapted_studies):
    rows = [r for r in development if r['planned_validation_fold'] == fold and r['sample_id'] in frames]
    ensembles = ARMS[3:] + (ADAPTED_ARMS if adapted_studies else ())
    averages = {r['sample_id']: dict.fromkeys(ensembles, 0) for r in rows}
    adapted_net = runner.MixedBoundaryUNet(True, head_seed=0, encoder_weights=None).cuda().eval() if adapted_studies else None
    for seed in SEEDS:
        path = coarse_checkpoint(development, fold, seed)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
        if adapted_net is not None:
            new_path = adapted_checkpoint(adapted_studies[seed], fold, seed, path)
            adapted_net.load_state_dict(torch.load(new_path, map_location='cpu', weights_only=True)['model'])
        for row in rows:
            sample = row['sample_id']
            patient = row['patient_source_id']
            frame = frames[sample]
            predictions = predict_frame(net, frame, large[sample], runner)
            frame_ensemble = {'ensemble': predictions[0], 'ensemble_scale_and_flip': predictions[2]}
            if adapted_net is not None:
                probability = flip_pair(adapted_net, large[sample], runner)
                paired = (cv2.resize(predictions[1], SIZE, interpolation=cv2.INTER_LINEAR)
                          + cv2.resize(probability, SIZE, interpolation=cv2.INTER_LINEAR)) * .5
                frame_ensemble.update(adapted_high_ensemble=probability, ensemble_with_adapted_high=paired)
            for method, probability in frame_ensemble.items():
                averages[sample][method] += probability / len(SEEDS)
            for method, p in zip(ARMS[:3], predictions, strict=True):
                by_patient[method][patient].append(dice_at_original_size(p >= .5, frame))
    for row in rows:
        sample = row['sample_id']
        for method, p in averages[sample].items():
            by_patient[method][row['patient_source_id']].append(
                dice_at_original_size(p >= .5, frames[sample]))
    print(f'{fold}: evaluated {len(rows)} complete masks.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REGION_STUDY / 'report' / 'inference_checks.json')
    parser.add_argument('--adapted-study', type=Path, action='append', default=[],
                        help='Optional seed-specific whole-image studies, one per seed')
    args = parser.parse_args()
    output = args.output.resolve()
    if LOGS.resolve() not in output.parents or ORIGINAL.resolve() in output.parents:
        parser.error('Write inference comparisons under a separate model/logs study')
    adapted = {}
    for study in args.adapted_study:
        seed, = read_json(study / 'protocol.json')['seeds']
        if seed in adapted:
            parser.error('Pass one adapted study per seed')
        adapted[seed] = study.resolve()
    if adapted and set(adapted) != set(SEEDS):
        parser.error('Pass adapted studies for all three seeds')
    development, _ = verify(BASELINE)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    rows = [r for r in development if r['annotation_kind'] != 'linestrip']
    frames = read_frames(runner, rows)
    large = {sid: cv2.resize(fr.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA)
             for sid, fr in frames.items()}
    net = runner.MixedBoundaryUNet(True, head_seed=0, encoder_weights=None).cuda().eval()
    arms = ARMS + (ADAPTED_ARMS if adapted else ())
    values = {arm: defaultdict(list) for arm in arms}
    for fold in FOLDS:
        evaluate_fold(runner, development, frames, large, net, fold, values, adapted)
    n_patients = len({r['patient_source_id'] for r in rows})
    if len(rows) != 66 or n_patients != 5:
        raise ValueError('Frozen Dice cohort coverage changed')
    summary = {arm: {'dice': mean(mean(v) for v in patients.values()),
                    'by_patient': sorted(round(mean(v), 6) for v in patients.values())}
               for arm, patients in values.items()}
    if abs(summary['single']['dice'] - .8394367633807487) > .0001:
        raise ValueError('Original three-seed baseline did not reproduce')
    ordered_patients = sorted(values['single'], key=lambda pt: mean(values['single'][pt]))
    paired = [{arm: mean(values[arm][pt]) for arm in arms} for pt in ordered_patients]
    runner.atomic_json(output, {'scope': 'development-only exploratory fixed-rule scoring',
                       'seeds': SEEDS, 'patients': n_patients, 'frames': len(rows),
                       'region_threshold': .5, 'native_size': [768, 432], 'high_size': HIGH_SIZE,
                       'evaluator_sha256': digest(Path(__file__)),
                       'adapted_protocol_sha256': {seed: digest(study / 'protocol.json')
                                                    for seed, study in adapted.items()},
                       'methods': summary, 'patient_pairs_sorted_by_old_d_dice': paired})
    print(summary, flush=True)


if __name__ == '__main__':
    main()
