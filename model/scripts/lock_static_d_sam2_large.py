"""Lock the selected SAM 2.1 Large pipeline and score complete independent masks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
import sys

import cv2
import torch

import evaluate_static_d_locked_regions as cohorts
import evaluate_static_d_sam2_large as development_study
import train_static_d_region_refiner as region_evaluation
from evaluate_static_d_region_ensemble import SEEDS
from evaluate_static_d_sam2_large import LARGE_CONFIG, LARGE_SHA256
from refit_static_d_sam2_regions import ensemble_priors, pipeline_lock, prepare_image
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, exclusive_training, load_runner, read_json, verify_file
from train_static_d_sam2_decoder import decode, mask_from_logits

STUDY = LOGS / 'static_d_sam2_large_candidate_20261124_01'
METHOD = 'large_flip_blend'


def selected_candidate(source, deps, weights):
    verify_file(weights, LARGE_SHA256)
    study_result = development_study.STUDY / 'results.json'
    result = read_json(study_result)
    if result['scope'] != 'development-only' or result['frames'] != 66 or result['patients'] != 5:
        raise ValueError('The development selection requires the complete patient-fold comparison')
    verify_file(Path(development_study.__file__), result['source_sha256'])
    verify_file(source / 'sam2' / LARGE_CONFIG, result['large_config_sha256'])
    if result['large_weight_sha256'] != LARGE_SHA256 or max(result['overall'], key=result['overall'].get) != METHOD:
        raise ValueError('The pinned Large model and flip recipe must match the development selection')
    pipeline = pipeline_lock(source, deps)
    pipeline['sam2_weights_sha256'] = LARGE_SHA256
    pipeline['sam2_sources_sha256'][LARGE_CONFIG] = result['large_config_sha256']
    return {'scope': 'candidate-only', 'method': METHOD, 'pipeline': pipeline,
            'scorer_sha256': digest(Path(__file__)), 'cohort_code_sha256': digest(Path(cohorts.__file__)),
            'region_score_code_sha256': digest(Path(region_evaluation.__file__)),
            'development_result_sha256': digest(study_result), 'development_dice': result['overall'][METHOD]}


def load_large(source, deps, weights):
    verify_file(weights, LARGE_SHA256)
    if not deps.is_dir():
        raise ValueError('The pinned SAM 2.1 runtime is required')
    sys.path[:0] = [str(source), str(deps)]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(LARGE_CONFIG, str(weights), device='cuda', apply_postprocessing=False)
    return SAM2ImagePredictor(model.requires_grad_(False))


@torch.no_grad()
def exact_mask(predictor, position, frame, prior):
    standard = prepare_image(predictor, frame, prior, False)
    if not standard.prompt_embeddings:
        return standard.prior_mask
    decoder = predictor.model.sam_mask_decoder
    logits = decode(decoder, position, standard, 0)
    mirror = replace(frame, rgb=cv2.flip(frame.rgb, 1))
    flipped = prepare_image(predictor, mirror, cv2.flip(prior, 1), False)
    if flipped.prompt_embeddings:
        mirrored_logits = decode(decoder, position, flipped, 0).flip(dims=(-1,))
        logits = (logits + mirrored_logits) / 2
    return mask_from_logits(logits, predictor, frame)


@torch.no_grad()
def score(runner, candidate, rows, frames, predictor):
    priors, reference = ensemble_priors(runner, rows, frames, candidate['pipeline'], include_reference_scores=True)
    position = predictor.model.sam_prompt_encoder.get_dense_pe()
    predictions = defaultdict(list)
    for row in rows:
        sid, patient = row['sample'], row['patient']
        frame = frames[sid]
        mask = exact_mask(predictor, position, frame, priors[sid])
        predictions[patient].append(region_evaluation.dice_at_original_size(mask, frame))
    patients = [{'frames': len(values), 'old_d': mean(mean(reference[seed, patient]) for seed in SEEDS),
                 'candidate': mean(values)} for patient, values in predictions.items()]
    return {'frames': len(rows), 'patients': patients, 'old_d': mean(r['old_d'] for r in patients),
            'candidate': mean(r['candidate'] for r in patients)}


def evaluate(runner, candidate_path, args, source, deps, weights):
    args.evaluation.mkdir(exist_ok=True)
    with exclusive_training(args.evaluation):
        rows = cohorts.read_cohort(runner, args.manifest)
        frames = cohorts.prepare_frames(runner, rows, historical=False)
        locked = args.evaluation / 'locked_regions.private.json'
        proposal = {'scope': 'independent', 'method': METHOD, 'candidate_sha256': digest(candidate_path), 'data': rows}
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
            result = score(runner, read_json(candidate_path), rows, frames, load_large(source, deps, weights))
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
    source, deps, weights = (path.resolve() for path in (args.source, args.dependencies, args.weights))
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    candidate = selected_candidate(source, deps, weights)
    STUDY.mkdir(exist_ok=True)
    candidate_path = STUDY / 'candidate.json'
    with exclusive_training(STUDY):
        if candidate_path.exists():
            if read_json(candidate_path) != candidate:
                raise ValueError('The candidate pipeline differs from the frozen candidate')
        elif args.manifest is None:
            runner.atomic_json(candidate_path, candidate)
        else:
            raise ValueError('Lock the selected development candidate before accepting evaluation references')
    if args.manifest is None:
        print(f'Frozen candidate: {candidate_path}', flush=True)
    else:
        evaluate(runner, candidate_path, args, source, deps, weights)


if __name__ == '__main__':
    main()
