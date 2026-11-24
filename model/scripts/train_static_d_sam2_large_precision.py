"""Compare Dice and false-positive-weighted adaptation of the Large mask decoder."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import mean

import cv2
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

import evaluate_static_d_sam2_large as control
from evaluate_static_d_color_context import average_prior
from evaluate_static_d_region_ensemble import HIGH_SIZE
from lock_static_d_sam2_large import load_large
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, exclusive_training, load_runner, read_json, verify, verify_file
from train_static_d_region_refiner import FOLDS, read_frames
from train_static_d_sam2_decoder import SEED, STEPS, decode, mask_from_logits, prepare_example
from train_static_d_sam2_prompt_recovery import score_masks

STUDY = LOGS / 'static_d_sam2_large_precision_20261124_01'
METHODS = ('large', 'dice_adapted', 'dice_blend', 'precision_adapted', 'precision_blend')
FP_WEIGHTS = {'dice': .5, 'precision': .65}


def trains(decoder, position, samples, fold, arm):
    random = Random(SEED)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=8e-5, weight_decay=.01)
    patients = sorted(samples)
    decoder.train()
    losses = []
    fp_weight = FP_WEIGHTS[arm]
    for step in range(1, STEPS + 1):
        sample = random.choice(samples[random.choice(patients)])
        n = len(sample.prompt_embeddings)
        index = random.choices(range(n), weights=[2, 1, 1][:n])[0]
        logits = decode(decoder, position, sample, index)
        reference = sample.target.cuda()
        probability = logits.sigmoid()
        tp = (probability * reference).sum()
        fp = (probability * (1 - reference)).sum()
        fn = ((1 - probability) * reference).sum()
        overlap = (tp + .5) / (tp + fp_weight * fp + (1 - fp_weight) * fn + .5)
        loss = .75 * (1 - overlap) + .25 * binary_cross_entropy_with_logits(logits, reference)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 60 == 0:
            print(f'{fold} {arm}: step {step}/{STEPS}, loss={mean(losses[-60:]):.4f}', flush=True)
    decoder.eval()
    return losses


@torch.no_grad()
def encode_frame(predictor, frame):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        predictor.set_image(frame.rgb)
    return tuple(t.cpu() for t in (predictor._features['image_embed'], *predictor._features['high_res_feats']))


@torch.no_grad()
def fold_logits(decoder, position, examples):
    result = {}
    for sid, (ordinary, flip) in examples.items():
        if not ordinary.prompt_embeddings:
            continue
        base = decode(decoder, position, ordinary, 0)
        if flip.prompt_embeddings:
            base = (base + decode(decoder, position, flip, 0).flip(dims=(-1,))) / 2
        result[sid] = base.cpu()
    return result


def save_fold(decoder, arm, fold):
    path = STUDY / f'{fold}_{arm}.pt'
    temporary = path.with_suffix('.partial')
    torch.save({'decoder': decoder.state_dict(), 'step': STEPS, 'fold': fold, 'arm': arm}, temporary)
    temporary.replace(path)
    return digest(path)


def run(runner, source, deps, weights):
    reference_file = control.STUDY / 'results.json'
    previous = read_json(reference_file)
    verify_file(Path(control.__file__), previous['source_sha256'])
    development, _ = verify(BASELINE)
    for path in (ORIGINAL, BASELINE, STUDY):
        runner.forbid_test_access(path)
    closed = [r for r in development if r['annotation_kind'] != 'linestrip']
    frames = read_frames(runner, closed)
    large = {sid: cv2.resize(frame.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA) for sid, frame in frames.items()}
    torch.manual_seed(SEED)
    predictor = load_large(source, deps, weights)
    model = predictor.model
    decoder = model.sam_mask_decoder.requires_grad_(True)
    official = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
    with torch.no_grad():
        position = model.sam_prompt_encoder.get_dense_pe()
    features = {sid: encode_frame(predictor, frame) for sid, frame in frames.items()}
    print(f'Cached Large image features for {len(features)} development images.', flush=True)

    records, evidence = [], {}
    for fold in FOLDS:
        training = [r for r in closed if r['planned_validation_fold'] != fold]
        validation = [r for r in closed if r['planned_validation_fold'] == fold]
        if {r['patient_source_id'] for r in training} & {r['patient_source_id'] for r in validation}:
            raise ValueError(f'Patient overlap in {fold}')
        priors = average_prior(runner, development, closed, frames, large, fold, 'D')
        samples = defaultdict(list)
        examples = {}
        for row in training:
            sid = row['sample_id']
            sample = prepare_example(predictor, frames[sid], features[sid], priors[sid], True)
            if sample.prompt_embeddings:
                samples[row['patient_source_id']].append(sample)
        for row in validation:
            sid = row['sample_id']
            frame, prior = frames[sid], priors[sid]
            mirror = replace(frame, rgb=cv2.flip(frame.rgb, 1))
            examples[sid] = (prepare_example(predictor, frame, features[sid], prior, False),
                             prepare_example(predictor, mirror, encode_frame(predictor, mirror), cv2.flip(prior, 1), False))
        decoder.load_state_dict(official)
        decoder.eval()
        initial = fold_logits(decoder, position, examples)
        adjusted = {}
        evidence[fold] = {'training': sum(map(len, samples.values())), 'validation': len(validation), 'arms': {}}
        print(f'{fold}: {evidence[fold]["training"]} training images and {len(validation)} validation images.', flush=True)
        for arm in FP_WEIGHTS:
            decoder.load_state_dict(official)
            losses = trains(decoder, position, samples, fold, arm)
            adjusted[arm] = fold_logits(decoder, position, examples)
            evidence[fold]['arms'][arm] = {'loss': losses, 'sha256': save_fold(decoder, arm, fold)}
        with torch.no_grad():
            for row in validation:
                sid = row['sample_id']
                frame, (original, _) = frames[sid], examples[sid]
                masks = dict.fromkeys(METHODS, original.prior_mask)
                if sid in initial:
                    baseline = initial[sid].cuda()
                    masks['large'] = mask_from_logits(baseline, predictor, frame)
                    for arm in FP_WEIGHTS:
                        logits = adjusted[arm][sid].cuda()
                        masks[f'{arm}_adapted'] = mask_from_logits(logits, predictor, frame)
                        masks[f'{arm}_blend'] = mask_from_logits((logits + baseline) / 2, predictor, frame)
                records.append({'patient': row['patient_source_id'], 'sample': sid, 'fold': fold,
                                'scores': score_masks(masks, frame)})
        print(f'{fold}: completed scoring and checkpoint receipt.', flush=True)

    patients = defaultdict(list)
    for row in records:
        patients[row['patient']].append(row)
    metrics = ('dice', 'fn_over_reference', 'fp_over_reference')
    paired = [{method: {metric: mean(r['scores'][method][metric] for r in rows) for metric in metrics}
               for method in METHODS} for rows in patients.values()]
    overall = {method: {key: mean(r[method][key] for r in paired) for key in metrics} for method in METHODS}
    if len(records) != 66 or len(patients) != 5 or abs(overall['large']['dice'] - previous['overall']['large_flip_blend']) > .0001:
        raise ValueError('The locked development cohort or Large score did not reproduce')
    runner.atomic_json(STUDY / 'scores.private.json', records)
    runner.atomic_json(STUDY / 'results.json', {
        'scope': 'development-only', 'frames': len(records), 'patients': len(paired), 'overall': overall,
        'paired_patients': sorted(paired, key=lambda p: p['large']['dice']), 'folds': evidence,
        'steps_per_arm_and_fold': STEPS, 'false_positive_weights': FP_WEIGHTS,
        'source_sha256': digest(Path(__file__)), 'control_sha256': digest(reference_file),
    })
    print({m: round(val['dice'], 6) for m, val in overall.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    STUDY.mkdir(exist_ok=True)
    with exclusive_training(STUDY):
        if (STUDY / 'results.json').exists():
            raise FileExistsError('This Large decoder study has already completed')
        run(runner, args.source.resolve(), args.dependencies.resolve(), args.weights.resolve())


if __name__ == '__main__':
    main()
