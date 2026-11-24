"""Test SAM 2.1 training with localized foreground additions and deletions in D prompts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

import train_static_d_sam2_decoder as control
from evaluate_static_d_color_context import average_prior
from evaluate_static_d_region_ensemble import HIGH_SIZE
from evaluate_static_d_sam2_regions import prompts
from refit_static_d_sam2_regions import load_sam
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, exclusive_training, load_runner, read_json, verify, verify_file
from train_static_d_region_refiner import FOLDS, read_frames
from train_static_d_sam2_decoder import SEED, STEPS, decode, mask_from_logits, train_decoder

STUDY = LOGS / 'static_d_sam2_prompt_recovery_20261124_01'
METHODS = ('lab', 'frozen_sam2', 'translation_blend', 'recovery', 'recovery_blend', 'three_decoder_blend')


def local_prompt_errors(probability, initial_mask, random):
    """Add or remove a connected patch along an existing predicted boundary."""
    contour, _ = cv2.findContours(initial_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contour:
        return []
    perimeter = max(contour, key=cv2.contourArea).reshape(-1, 2)
    variants = []
    for foreground in (False, True):
        center = tuple(map(int, perimeter[random.integers(len(perimeter))]))
        radius = int(random.integers(64, 113))
        patch = np.zeros_like(initial_mask, dtype=np.uint8)
        cv2.ellipse(patch, center, (radius, round(radius * .75)), int(random.integers(180)),
                    0, 360, 1, thickness=-1)
        patch = patch.astype(bool)
        changed_mask = initial_mask | patch if foreground else initial_mask & ~patch
        if changed_mask.sum() < .55 * initial_mask.sum() or np.array_equal(changed_mask, initial_mask):
            continue
        changed_probability = np.where(patch, .88 if foreground else .10, probability).astype(np.float32)
        variants.append((changed_probability, changed_mask))
    return variants


@torch.no_grad()
def append_perturbed_prompts(example, predictor, frame, prior, random):
    if not example.prompt_embeddings:
        return
    probability = cv2.resize(prior, HIGH_SIZE, interpolation=cv2.INTER_AREA)
    for changed, changed_mask in local_prompt_errors(probability, example.prior_mask, random):
        mask_logits, coords, kinds, box = prompts(changed, changed_mask, frame.rgb.shape)
        if box is None:
            continue
        mask, coords, kinds, box = predictor._prep_prompts(coords, kinds, box, mask_logits, True)
        points = torch.cat((box.reshape(-1, 2, 2), coords), dim=1)
        labels = torch.cat((kinds.new_tensor([[2, 3]]), kinds), dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            sparse, dense = predictor.model.sam_prompt_encoder(points=(points, labels), boxes=None, masks=mask)
        example.prompt_embeddings.append((sparse.cpu(), dense.cpu()))


def score_masks(masks, frame):
    width, height = frame.original_size
    reference = np.unpackbits(frame.packed, count=width * height).reshape(height, width).astype(bool)
    area = int(reference.sum())
    scores = {}
    for name, mask in masks.items():
        predicted = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlap = int(np.count_nonzero(predicted & reference))
        predicted_area = int(predicted.sum())
        scores[name] = {'dice': 2 * overlap / max(1, predicted_area + area),
                        'fn_over_reference': (area - overlap) / max(1, area),
                        'fp_over_reference': (predicted_area - overlap) / max(1, area)}
    return scores


def evaluate_fold(predictor, position, waiting, recovery_decoder, frozen, translated):
    records = []
    with torch.no_grad():
        for row, sample, frame in waiting:
            sid = row['sample_id']
            masks = dict.fromkeys(METHODS, sample.prior_mask)
            if sid in frozen:
                base = frozen[sid].cuda()
                shift = translated[sid].cuda()
                new = decode(recovery_decoder, position, sample, 0)
                logits = (base, (base + shift) / 2, new, (base + new) / 2, (2 * base + shift + new) / 4)
                for name, value in zip(METHODS[1:], logits, strict=True):
                    masks[name] = mask_from_logits(value, predictor, frame)
            records.append({'patient': row['patient_source_id'], 'fold': row['planned_validation_fold'],
                            'sample': sid, 'scores': score_masks(masks, frame)})
    return records


def run(runner, source, deps, weights):
    previous = read_json(control.STUDY / 'results.json')
    verify_file(Path(control.__file__), previous['source_sha256'])
    development, _ = verify(BASELINE)
    for study in (ORIGINAL, BASELINE, STUDY):
        runner.forbid_test_access(study)
    closed = [r for r in development if r['annotation_kind'] != 'linestrip']
    frames = read_frames(runner, closed)
    large = {sid: cv2.resize(frame.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA) for sid, frame in frames.items()}
    torch.manual_seed(SEED)
    predictor = load_sam(source, deps, weights)
    model = predictor.model
    decoder = model.sam_mask_decoder.requires_grad_(True)
    official = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
    image_features = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        position = model.sam_prompt_encoder.get_dense_pe()
        for sid, frame in frames.items():
            predictor.set_image(frame.rgb)
            image_features[sid] = tuple(t.cpu() for t in (predictor._features['image_embed'], *predictor._features['high_res_feats']))
    print(f'Cached features for {len(frames)} development images.', flush=True)
    records = []
    fold_evidence = {}
    for fold in FOLDS:
        random = np.random.default_rng(SEED)
        training = [r for r in closed if r['planned_validation_fold'] != fold]
        validation = [r for r in closed if r['planned_validation_fold'] == fold]
        if {r['patient_source_id'] for r in training} & {r['patient_source_id'] for r in validation}:
            raise ValueError(f'Patient overlap in {fold}')
        priors = average_prior(runner, development, closed, frames, large, fold, 'D')
        samples = defaultdict(list)
        waiting = []
        for row in training:
            sid = row['sample_id']
            frame = frames[sid]
            # Keep the original, unshifted hint. Its two translated forms are replaced by local errors.
            example = control.prepare_example(predictor, frame, image_features[sid], priors[sid], True)
            if example.prompt_embeddings:
                del example.prompt_embeddings[1:]
                append_perturbed_prompts(example, predictor, frame, priors[sid], random)
                samples[row['patient_source_id']].append(example)
        for row in validation:
            sid = row['sample_id']
            frame = frames[sid]
            waiting.append((row, control.prepare_example(predictor, frame, image_features[sid], priors[sid], False), frame))

        decoder.eval()
        frozen, translated = {}, {}
        old_path = control.STUDY / f'{fold}_decoder.pt'
        verify_file(old_path, previous['folds'][fold]['checkpoint_sha256'])
        with torch.no_grad():
            for state, destination in ((official, frozen), (torch.load(old_path, map_location='cpu', weights_only=True)['decoder'], translated)):
                decoder.load_state_dict(state)
                for row, example, _ in waiting:
                    if example.prompt_embeddings:
                        destination[row['sample_id']] = decode(decoder, position, example, 0).cpu()

        decoder.load_state_dict(official)
        total = sum(map(len, samples.values()))
        generated = sum(len(e.prompt_embeddings)-1 for patient in samples.values() for e in patient)
        print(f'{fold}: {total} training images, {generated} local prompt errors and {len(waiting)} validation images.', flush=True)
        losses = train_decoder(decoder, position, samples, fold)
        records.extend(evaluate_fold(predictor, position, waiting, decoder, frozen, translated))
        checkpoint = STUDY / f'{fold}_decoder.pt'
        temporary = checkpoint.with_suffix('.partial')
        torch.save({'decoder': decoder.state_dict(), 'step': STEPS, 'fold': fold}, temporary)
        temporary.replace(checkpoint)
        fold_evidence[fold] = {'training': total, 'perturbed_prompts': generated, 'validation': len(waiting),
                               'loss': losses, 'checkpoint_sha256': digest(checkpoint)}
        print(f'{fold}: validation complete.', flush=True)

    patients = defaultdict(list)
    for row in records:
        patients[row['patient']].append(row)
    metrics = ('dice', 'fn_over_reference', 'fp_over_reference')
    patient_results = [{method: {metric: mean(r['scores'][method][metric] for r in rows) for metric in metrics}
                        for method in METHODS} for rows in patients.values()]
    overall = {method: {metric: mean(r[method][metric] for r in patient_results) for metric in metrics}
               for method in METHODS}
    if len(records) != len(closed) or len(patients) != 5 or abs(overall['translation_blend']['dice'] - previous['overall']['blended_sam2']) > .0001:
        raise ValueError('Development cohort or selected comparison failed to reproduce')
    runner.atomic_json(STUDY / 'scores.private.json', records)
    runner.atomic_json(STUDY / 'results.json', {
        'scope': 'development-only', 'patients': len(patients), 'frames': len(records), 'steps_per_fold': STEPS,
        'overall': overall, 'paired_patients': sorted(patient_results, key=lambda r: r['translation_blend']['dice']),
        'folds': fold_evidence, 'source_sha256': digest(Path(__file__)), 'control_sha256': digest(control.STUDY / 'results.json'),
    })
    print({method: round(value['dice'], 6) for method, value in overall.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    source, deps, weights = (path.resolve() for path in (args.source, args.dependencies, args.weights))
    cv2.setNumThreads(2)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    STUDY.mkdir(exist_ok=True)
    with exclusive_training(STUDY):
        if (STUDY / 'results.json').exists():
            raise FileExistsError('The prompt recovery experiment already completed')
        run(runner, source, deps, weights)


if __name__ == '__main__':
    main()
