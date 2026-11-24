"""Refit the selected Large decoder and lock its independent comparison with old D."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import cv2
import torch

import evaluate_static_d_locked_regions as cohorts
import lock_static_d_sam2_large as frozen_large
import train_static_d_region_refiner as regions
import train_static_d_sam2_large_precision as precision
from evaluate_static_d_region_ensemble import SEEDS
from refit_static_d_sam2_regions import ensemble_priors
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, exclusive_training, load_runner, read_json, verify, verify_file
from train_static_d_sam2_decoder import SEED, STEPS, mask_from_logits, prepare_example

STUDY = LOGS / 'static_d_sam2_large_refit_20261124_01'
CHECKPOINT = STUDY / 'decoder.pt'
CANDIDATE = STUDY / 'candidate.json'
ARM = 'dice'
METHOD = 'dice_blend'


def protocol(source, deps, weights):
    control = frozen_large.selected_candidate(source, deps, weights)
    selection = precision.STUDY / 'results.json'
    results = read_json(selection)
    if (results['scope'] != 'development-only' or results['frames'] != 66 or results['patients'] != 5
            or results['steps_per_arm_and_fold'] != STEPS or results['false_positive_weights'] != precision.FP_WEIGHTS):
        raise ValueError('The refit has a different development protocol')
    verify_file(Path(precision.__file__), results['source_sha256'])
    if max(results['overall'], key=lambda key: results['overall'][key]['dice']) != METHOD:
        raise ValueError('Expected the highest-Dice method in the development comparison')
    return {'method': METHOD, 'large_candidate': control, 'development_results_sha256': digest(selection),
            'source_sha256': digest(Path(__file__)), 'cohort_source_sha256': digest(Path(cohorts.__file__)),
            'development_manifest_sha256': digest(BASELINE / 'candidate_manifest.csv'),
            'development_dice': results['overall'][METHOD]['dice'], 'refit_steps': STEPS, 'seed': SEED}


def fit(runner, identity, source, deps, weights):
    for path in (ORIGINAL, BASELINE, STUDY):
        runner.forbid_test_access(path)
    development, _ = verify(BASELINE)
    closed = [r for r in development if r['annotation_kind'] != 'linestrip']
    frames = regions.read_frames(runner, closed)
    rows = [{'sample': r['sample_id'], 'patient': r['patient_source_id']} for r in closed]
    priors, _ = ensemble_priors(runner, rows, frames, identity['large_candidate']['pipeline'])
    torch.manual_seed(SEED)
    predictor = frozen_large.load_large(source, deps, weights)
    model = predictor.model
    decoder = model.sam_mask_decoder.requires_grad_(True)
    with torch.no_grad():
        position = model.sam_prompt_encoder.get_dense_pe()
    samples = defaultdict(list)
    for row in rows:
        sid = row['sample']
        frame = frames[sid]
        prepared = prepare_example(predictor, frame, precision.encode_frame(predictor, frame), priors[sid], True)
        if prepared.prompt_embeddings:
            samples[row['patient']].append(prepared)
    training_count = sum(map(len, samples.values()))
    print(f'Refit on {training_count} complete masks from {len(samples)} patients.', flush=True)
    losses = precision.trains(decoder, position, samples, 'refit', ARM)
    partial = CHECKPOINT.with_suffix('.partial')
    torch.save({'decoder': decoder.state_dict(), 'steps': STEPS, 'arm': ARM}, partial)
    partial.replace(CHECKPOINT)
    return {'scope': 'candidate-only', 'protocol': identity, 'decoder_sha256': digest(CHECKPOINT),
            'training_patient_count': len(samples), 'training_mask_count': training_count, 'train_loss': losses}


@torch.no_grad()
def score(runner, candidate, rows, frames, predictor):
    verify_file(CHECKPOINT, candidate['decoder_sha256'])
    adjusted = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)['decoder']
    model = predictor.model
    decoder = model.sam_mask_decoder
    original = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    position = model.sam_prompt_encoder.get_dense_pe()
    priors, reference = ensemble_priors(runner, rows, frames, candidate['protocol']['large_candidate']['pipeline'],
                                       include_reference_scores=True)
    predictions = defaultdict(list)
    for row in rows:
        sid, patient = row['sample'], row['patient']
        frame, prior = frames[sid], priors[sid]
        sample = prepare_example(predictor, frame, precision.encode_frame(predictor, frame), prior, False)
        mask = sample.prior_mask
        if sample.prompt_embeddings:
            mirror = replace(frame, rgb=cv2.flip(frame.rgb, 1))
            flipped = prepare_example(predictor, mirror, precision.encode_frame(predictor, mirror), cv2.flip(prior, 1), False)
            examples = {sid: (sample, flipped)}
            decoder.load_state_dict(original)
            before = precision.fold_logits(decoder, position, examples)[sid]
            decoder.load_state_dict(adjusted)
            after = precision.fold_logits(decoder, position, examples)[sid]
            mask = mask_from_logits((before + after).cuda() / 2, predictor, frame)
        predictions[patient].append(regions.dice_at_original_size(mask, frame))
    patients = [{'frames': len(values), 'old_d': mean(mean(reference[seed, patient]) for seed in SEEDS),
                 'candidate': mean(values)} for patient, values in predictions.items()]
    return {'frames': len(rows), 'patients': patients, 'old_d': mean(r['old_d'] for r in patients),
            'candidate': mean(r['candidate'] for r in patients)}


def evaluate(runner, candidate, args, source, deps, weights):
    args.evaluation.mkdir(exist_ok=True)
    with exclusive_training(args.evaluation):
        rows = cohorts.read_cohort(runner, args.manifest)
        frames = cohorts.prepare_frames(runner, rows, historical=False)
        locked = args.evaluation / 'locked_regions.private.json'
        proposal = {'scope': 'independent', 'method': METHOD, 'candidate_sha256': digest(CANDIDATE), 'data': rows}
        if locked.exists():
            if read_json(locked) != proposal:
                raise ValueError('Independent patients, mask references or candidate changed after freezing')
        elif args.freeze_only:
            runner.atomic_json(locked, proposal)
        else:
            raise ValueError('Freeze the complete independent references before scoring')
        if args.freeze_only:
            print(f'Frozen {len(rows)} complete independent masks.', flush=True)
            return
        output = args.evaluation / 'results.json'
        if output.exists():
            result = read_json(output)
            if result['locked_manifest_sha256'] != digest(locked):
                raise ValueError('This score was produced for a different frozen cohort')
        else:
            result = score(runner, candidate, rows, frames, frozen_large.load_large(source, deps, weights))
            result.update(scope='independent', method=METHOD, locked_manifest_sha256=digest(locked))
            runner.atomic_json(output, result)
        print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--identity-confirmed', action='store_true')
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    if args.manifest:
        if not args.identity_confirmed or args.evaluation is None:
            parser.error('A dedicated evaluation directory and confirmed patient independence are required')
        args.evaluation = args.evaluation.resolve()
        if args.evaluation.parent != LOGS.resolve() or args.evaluation in (STUDY.resolve(), ORIGINAL.resolve(), BASELINE.resolve()):
            parser.error('Use a separate directory under model/logs for independent evaluation')
    elif args.identity_confirmed or args.evaluation or args.freeze_only:
        parser.error('Evaluation flags require a complete-mask manifest')
    cv2.setNumThreads(2)
    source, deps, weights = (p.resolve() for p in (args.source, args.dependencies, args.weights))
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    identity = protocol(source, deps, weights)
    STUDY.mkdir(exist_ok=True)
    with exclusive_training(STUDY):
        if CANDIDATE.exists():
            candidate = read_json(CANDIDATE)
            if candidate['protocol'] != identity:
                raise ValueError('The existing candidate lock refers to a different pipeline')
            verify_file(CHECKPOINT, candidate['decoder_sha256'])
        elif args.manifest is None:
            candidate = fit(runner, identity, source, deps, weights)
            runner.atomic_json(CANDIDATE, candidate)
        else:
            raise ValueError('Lock the refit candidate before accepting evaluation references')
    if args.manifest is None:
        print(f'Locked Large decoder refit: {CANDIDATE}', flush=True)
    else:
        evaluate(runner, candidate, args, source, deps, weights)


if __name__ == '__main__':
    main()
