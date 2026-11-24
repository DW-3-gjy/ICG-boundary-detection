"""Refit the selected SAM 2.1 pipeline and lock an independent-mask evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys

import cv2
import torch

import evaluate_static_d_color_context as color_context
import evaluate_static_d_region_ensemble as region_ensemble
import evaluate_static_d_sam2_regions as sam2_regions
import train_static_d_sam2_decoder as decoder_study
from evaluate_static_d_locked_regions import checkpoints, prepare_frames, read_cohort
from evaluate_static_d_region_ensemble import HIGH_SIZE, SEEDS, predict_frame
from evaluate_static_d_sam2_regions import CONFIG, MODEL_SHA256
from static_d_baseline import (
    DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, exclusive_training, load_runner,
    read_json, verify, verify_file,
)
from train_static_d_region_refiner import dice_at_original_size, read_frames
from train_static_d_sam2_decoder import SEED, STEPS, decode, mask_from_logits, prepare_example, train_decoder

STUDY = LOGS / 'static_d_sam2_refit_20261124_01'
CHECKPOINT = 'decoder.pt'
CANDIDATE = 'candidate.json'


def load_sam(source, deps, weights):
    verify_file(weights, MODEL_SHA256)
    if not deps.is_dir():
        raise ValueError('The isolated SAM 2.1 runtime directory is required')
    sys.path[:0] = [str(source), str(deps)]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(CONFIG, str(weights), device='cuda', apply_postprocessing=False)
    model.requires_grad_(False)
    return SAM2ImagePredictor(model)


def pipeline_lock(source, deps):
    modules = (color_context, region_ensemble, sam2_regions, decoder_study)
    sam = source / 'sam2'
    sam_assets = sorted(p for p in sam.rglob('*.py') if p.is_file()) + [sam / CONFIG]
    dependencies = sorted(path for path in deps.glob('*.dist-info/METADATA') if path.is_file())
    return {
        'source_sha256': digest(Path(__file__)),
        'helper_sha256': {m.__name__: digest(Path(m.__file__)) for m in modules},
        'sam2_weights_sha256': MODEL_SHA256,
        'sam2_sources_sha256': {p.relative_to(sam).as_posix(): digest(p) for p in sam_assets},
        'runtime_metadata_sha256': {p.parent.name: digest(p) for p in dependencies},
        'torch_version': str(torch.__version__), 'opencv_version': cv2.__version__,
        'old_d_checkpoints': checkpoints(('D',)),
    }


def ensemble_priors(runner, rows, frames, lock, include_reference_scores=False):
    priors = dict.fromkeys(frames, 0)
    reference = defaultdict(list)
    large = {sid: cv2.resize(frame.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA) for sid, frame in frames.items()}
    network = runner.MixedBoundaryUNet(True, head_seed=0, encoder_weights=None).cuda().eval()
    for entry in lock['old_d_checkpoints']:
        path = ORIGINAL / 'runs' / entry['run'] / 'best.pt'
        verify_file(path, entry['sha256'])
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['protocol_sha256'] != digest(ORIGINAL / 'protocol.json'):
            raise ValueError('An old D checkpoint does not match its frozen protocol')
        network.load_state_dict(state['model'])
        for row in rows:
            sid = row['sample']
            native, _, multiscale = predict_frame(network, frames[sid], large[sid], runner)
            priors[sid] += multiscale / len(SEEDS)
            if include_reference_scores:
                reference[entry['seed'], row['patient']].append(dice_at_original_size(native >= .5, frames[sid]))
        print(f"Verified and ran the frozen D seed {entry['seed']}.", flush=True)
    return priors, reference


@torch.no_grad()
def prepare_image(predictor, frame, prior, training):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        predictor.set_image(frame.rgb)
    features = (predictor._features['image_embed'], *predictor._features['high_res_feats'])
    return prepare_example(predictor, frame, tuple(t.cpu() for t in features), prior, training)


def refit(runner, lock, source, deps, weights):
    STUDY.mkdir(exist_ok=True)
    with exclusive_training(STUDY):
        for path in (ORIGINAL, BASELINE, STUDY):
            runner.forbid_test_access(path)
        receipt = STUDY / CANDIDATE
        if receipt.exists():
            candidate = read_json(receipt)
            if candidate['pipeline'] != lock:
                raise ValueError('The existing candidate was locked against different inputs')
            verify_file(STUDY / CHECKPOINT, candidate['decoder_sha256'])
            print(f'Candidate stays locked in {receipt}.', flush=True)
            return
        development, _ = verify(BASELINE)
        closed = [r for r in development if r['annotation_kind'] != 'linestrip']
        frames = read_frames(runner, closed)
        rows = [{'sample': r['sample_id'], 'patient': r['patient_source_id']} for r in closed]
        priors, _ = ensemble_priors(runner, rows, frames, lock)
        torch.manual_seed(SEED)
        predictor = load_sam(source, deps, weights)
        model = predictor.model
        decoder = model.sam_mask_decoder.requires_grad_(True)
        with torch.no_grad():
            position = model.sam_prompt_encoder.get_dense_pe()
        samples = defaultdict(list)
        for row in rows:
            sid = row['sample']
            example = prepare_image(predictor, frames[sid], priors[sid], True)
            if example.prompt_embeddings:
                samples[row['patient']].append(example)
        print(f'Refit uses {sum(map(len, samples.values()))} closed masks from {len(samples)} patients.', flush=True)
        losses = train_decoder(decoder, position, samples, 'refit')
        temporary = STUDY / 'decoder.partial'
        torch.save({'decoder': decoder.state_dict(), 'step': STEPS, 'fold': 'refit'}, temporary)
        temporary.replace(STUDY / CHECKPOINT)
        runner.atomic_json(receipt, {
            'scope': 'development-only-refit', 'pipeline': lock,
            'selection_sha256': digest(decoder_study.STUDY / 'results.json'),
            'development_manifest_sha256': digest(BASELINE / 'candidate_manifest.csv'),
            'decoder_sha256': digest(STUDY / CHECKPOINT), 'training_masks': sum(map(len, samples.values())),
            'training_patients': len(samples), 'seed': SEED, 'steps': STEPS, 'train_loss': losses, 'blend': [.5, .5],
        })
        print(f'Candidate locked in {receipt}.', flush=True)


@torch.no_grad()
def score(runner, rows, frames, lock, candidate, predictor):
    path = STUDY / CHECKPOINT
    verify_file(path, candidate['decoder_sha256'])
    adapted = torch.load(path, map_location='cpu', weights_only=True)['decoder']
    decoder = predictor.model.sam_mask_decoder
    original = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
    position = predictor.model.sam_prompt_encoder.get_dense_pe()
    priors, reference = ensemble_priors(runner, rows, frames, lock, include_reference_scores=True)
    predictions = defaultdict(list)
    for row in rows:
        sid, patient = row['sample'], row['patient']
        frame = frames[sid]
        example = prepare_image(predictor, frame, priors[sid], False)
        mask = example.prior_mask
        if example.prompt_embeddings:
            decoder.load_state_dict(original)
            initial = decode(decoder, position, example, 0)
            decoder.load_state_dict(adapted)
            updated = decode(decoder, position, example, 0)
            mask = mask_from_logits((initial + updated) / 2, predictor, frame)
        predictions[patient].append(dice_at_original_size(mask, frame))
    patients = [{'frames': len(values), 'old_d': mean(mean(reference[seed, patient]) for seed in SEEDS),
                 'candidate': mean(values)} for patient, values in predictions.items()]
    return {'frames': len(rows), 'patients': patients, 'old_d': mean(row['old_d'] for row in patients),
            'candidate': mean(row['candidate'] for row in patients)}


def evaluate(runner, lock, args, source, deps, weights):
    path = STUDY / CANDIDATE
    candidate = read_json(path)
    if candidate['pipeline'] != lock or candidate['blend'] != [.5, .5]:
        raise ValueError('Evaluation must use the exact locked candidate')
    verify_file(STUDY / CHECKPOINT, candidate['decoder_sha256'])
    args.evaluation.mkdir(exist_ok=True)
    with exclusive_training(args.evaluation):
        rows = read_cohort(runner, args.manifest)
        frames = prepare_frames(runner, rows, historical=False)
        private = args.evaluation / 'locked_regions.private.json'
        proposal = {'scope': 'independent', 'refit_candidate_sha256': digest(path), 'data': rows}
        if private.exists():
            if read_json(private) != proposal:
                raise ValueError('Candidate, independent patients or references changed after freezing')
        elif args.freeze_only:
            runner.atomic_json(private, proposal)
        else:
            raise ValueError('Freeze the complete independent references first')
        if args.freeze_only:
            print(f'Frozen {len(rows)} independent complete masks.', flush=True)
            return
        output = args.evaluation / 'results.json'
        if output.exists():
            result = read_json(output)
            if result['locked_manifest_sha256'] != digest(private):
                raise ValueError('Stored results belong to a different evaluation freeze')
        else:
            result = score(runner, rows, frames, lock, candidate, load_sam(source, deps, weights))
            result.update(scope='independent', locked_manifest_sha256=digest(private))
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
            parser.error('Confirm patient identity and supply a dedicated evaluation directory for the manifest')
        args.evaluation = args.evaluation.resolve()
        if args.evaluation.parent != LOGS.resolve() or args.evaluation in (STUDY.resolve(), ORIGINAL.resolve(), BASELINE.resolve()):
            parser.error('Use a separate directory under model/logs for independent evaluation')
    elif args.evaluation or args.identity_confirmed or args.freeze_only:
        parser.error('Evaluation options require an independent complete-mask manifest')
    cv2.setNumThreads(2)
    source, deps, weights = (p.resolve() for p in (args.source, args.dependencies, args.weights))
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    verify_file(weights, MODEL_SHA256)
    selected = read_json(decoder_study.STUDY / 'results.json')
    if selected['scope'] != 'development-only' or selected['source_sha256'] != digest(Path(decoder_study.__file__)):
        raise ValueError('The refit must match the completed development decoder procedure')
    lock = pipeline_lock(source, deps)
    if args.manifest:
        evaluate(runner, lock, args, source, deps, weights)
    else:
        refit(runner, lock, source, deps, weights)


if __name__ == '__main__':
    main()
