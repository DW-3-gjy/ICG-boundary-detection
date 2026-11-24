"""Fine-tune SAM 2.1's mask decoder on closed development masks by patient fold."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from evaluate_static_d_color_context import average_prior, color_cut
from evaluate_static_d_region_ensemble import HIGH_SIZE
from evaluate_static_d_sam2_regions import CONFIG, MODEL_SHA256, prompts
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, load_runner, verify
from train_static_d_region_refiner import FOLDS, dice_at_original_size, read_frames

STUDY = LOGS / 'static_d_sam2_decoder_20261124_01'
STEPS = 180
SEED = 42


@dataclass
class Example:
    image_features: tuple[torch.Tensor, ...]
    prompt_embeddings: list[tuple[torch.Tensor, torch.Tensor]]
    prior_mask: np.ndarray
    target: torch.Tensor | None


@torch.no_grad()
def prepare_example(predictor, frame, features, deep_prior, training):
    probability = cv2.resize(deep_prior, HIGH_SIZE, interpolation=cv2.INTER_AREA)
    label = color_cut(frame.rgb, deep_prior, HIGH_SIZE, 'lab')
    predictor._orig_hw = [frame.rgb.shape[:2]]
    embeddings = []
    offsets = [(0, 0), (-21, 10), (19, -12)] if training else [(0, 0)]
    for dx, dy in offsets:
        shift = np.array([[1., 0., dx], [0., 1., dy]], dtype=np.float32)
        moved_probability = cv2.warpAffine(probability, shift, HIGH_SIZE) if dx or dy else probability
        moved_mask = cv2.warpAffine(label.astype(np.uint8), shift, HIGH_SIZE).astype(bool) if dx or dy else label
        prior, coords, points, box = prompts(moved_probability, moved_mask, frame.rgb.shape)
        if box is None:
            continue
        mask, xy, kinds, box = predictor._prep_prompts(coords, points, box, prior, True)
        kinds = torch.cat((kinds.new_tensor([[2, 3]]), kinds), dim=1)
        xy = torch.cat((box.reshape(-1, 2, 2), xy), dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            sparse, dense = predictor.model.sam_prompt_encoder(points=(xy, kinds), boxes=None, masks=mask)
        embeddings.append((sparse.cpu(), dense.cpu()))
    target = None
    if training:
        reduced = cv2.resize(frame.region, (256, 256), interpolation=cv2.INTER_AREA)
        target = torch.from_numpy(reduced)[None, None]
    return Example(features, embeddings, label, target)


def decode(decoder, position, example, prompt_index):
    image, *pyramid = (feature.cuda() for feature in example.image_features)
    sparse, dense = (embedding.cuda() for embedding in example.prompt_embeddings[prompt_index])
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        logits, _, _, _ = decoder(
            image_embeddings=image, image_pe=position, sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense, multimask_output=False, repeat_image=False,
            high_res_features=pyramid,
        )
    return logits.float()


def train_decoder(decoder, position, samples, fold):
    random = Random(SEED)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=8e-5, weight_decay=.01)
    patients = sorted(samples)
    decoder.train()
    losses = []
    for step in range(1, STEPS + 1):
        sample = random.choice(samples[random.choice(patients)])
        index = random.choices(range(len(sample.prompt_embeddings)), weights=[2, 1, 1][:len(sample.prompt_embeddings)])[0]
        logits = decode(decoder, position, sample, index)
        reference = sample.target.cuda()
        probability = logits.sigmoid()
        dice = (2 * (probability * reference).sum() + 1) / (probability.sum() + reference.sum() + 1)
        loss = .75 * (1 - dice) + .25 * F.binary_cross_entropy_with_logits(logits, reference)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 30 == 0:
            print(f'{fold}: step {step}/{STEPS}, loss={mean(losses[-30:]):.4f}, grad={float(norm):.4f}', flush=True)
    decoder.eval()
    return losses


def mask_from_logits(logits, predictor, frame):
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite SAM decoder logits')
    mask = predictor._transforms.postprocess_masks(logits, frame.rgb.shape[:2])[0, 0] > predictor.mask_threshold
    return cv2.resize(mask.cpu().numpy().astype(np.uint8), HIGH_SIZE, interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    source, deps, weights = (p.resolve() for p in (args.source, args.dependencies, args.weights))
    if digest(weights) != MODEL_SHA256 or not (source / 'sam2' / CONFIG).is_file() or not deps.is_dir():
        raise ValueError('Expected the pinned SAM 2.1 source, weights and dependencies')
    sys.path[:0] = [str(source), str(deps)]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    development, _ = verify(BASELINE)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    STUDY.mkdir(exist_ok=True)
    runner.forbid_test_access(STUDY)
    if (STUDY / 'results.json').exists():
        raise FileExistsError('This decoder study has already completed')
    cv2.setNumThreads(2)
    torch.manual_seed(SEED)
    model = build_sam2(CONFIG, str(weights), device='cuda', apply_postprocessing=False)
    model.requires_grad_(False)
    predictor = SAM2ImagePredictor(model)
    decoder = model.sam_mask_decoder.requires_grad_(True)
    original = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
    closed = [r for r in development if r['annotation_kind'] != 'linestrip']
    frames = read_frames(runner, closed)
    large = {sid: cv2.resize(frame.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA) for sid, frame in frames.items()}
    features = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        position = model.sam_prompt_encoder.get_dense_pe()
        for sid, frame in frames.items():
            predictor.set_image(frame.rgb)
            features[sid] = tuple(t.cpu() for t in (predictor._features['image_embed'], *predictor._features['high_res_feats']))
    print(f'Cached image encoder features for {len(features)} development images.', flush=True)

    records, evidence = [], {}
    for fold in FOLDS:
        decoder.load_state_dict(original)
        decoder.eval()
        training = [r for r in closed if r['planned_validation_fold'] != fold]
        validation = [r for r in closed if r['planned_validation_fold'] == fold]
        if {r['patient_source_id'] for r in training} & {r['patient_source_id'] for r in validation}:
            raise ValueError(f'Patients overlap across {fold} partitions')
        priors = average_prior(runner, development, closed, frames, large, fold, 'D')
        train_samples = defaultdict(list)
        for row in training:
            sid = row['sample_id']
            sample = prepare_example(predictor, frames[sid], features[sid], priors[sid], True)
            if sample.prompt_embeddings:
                train_samples[row['patient_source_id']].append(sample)
        evaluation_examples, frozen_logits = {}, {}
        with torch.no_grad():
            for row in validation:
                sid = row['sample_id']
                sample = prepare_example(predictor, frames[sid], features[sid], priors[sid], False)
                evaluation_examples[sid] = sample
                if sample.prompt_embeddings:
                    frozen_logits[sid] = decode(decoder, position, sample, 0).cpu()
        print(f'{fold}: {sum(map(len, train_samples.values()))} training images from {len(train_samples)} patients and {len(validation)} validation images.', flush=True)
        losses = train_decoder(decoder, position, train_samples, fold)
        with torch.no_grad():
            for row in validation:
                sid = row['sample_id']
                frame, sample = frames[sid], evaluation_examples[sid]
                masks = dict.fromkeys(('lab', 'frozen_sam2', 'adapted_sam2', 'blended_sam2'), sample.prior_mask)
                if sid in frozen_logits:
                    frozen = frozen_logits[sid].cuda()
                    adapted = decode(decoder, position, sample, 0)
                    for method, logits in (('frozen_sam2', frozen), ('adapted_sam2', adapted),
                                           ('blended_sam2', (frozen + adapted) / 2)):
                        masks[method] = mask_from_logits(logits, predictor, frame)
                records.append({'fold': fold, 'patient': row['patient_source_id'], 'sample': sid,
                                **{method: dice_at_original_size(mask, frame) for method, mask in masks.items()}})
        checkpoint = STUDY / f'{fold}_decoder.pt'
        temporary = checkpoint.with_suffix('.partial')
        torch.save({'decoder': decoder.state_dict(), 'step': STEPS, 'fold': fold}, temporary)
        temporary.replace(checkpoint)
        evidence[fold] = {'validation': len(validation), 'training': sum(map(len, train_samples.values())),
                          'loss': losses, 'checkpoint_sha256': digest(checkpoint)}
        print(f'{fold}: validation and checkpoint complete.', flush=True)

    grouped = defaultdict(list)
    for row in records:
        grouped[row['patient']].append(row)
    methods = ('lab', 'frozen_sam2', 'adapted_sam2', 'blended_sam2')
    paired = [{key: mean(row[key] for row in rows) for key in methods} for rows in grouped.values()]
    overall = {key: mean(patient[key] for patient in paired) for key in methods}
    if len(records) != len(closed) or len(grouped) != 5 or abs(overall['frozen_sam2'] - .8829902733) > .0001:
        raise ValueError(f'The SAM control or the cohort did not reproduce: {overall}')
    runner.atomic_json(STUDY / 'scores.private.json', records)
    runner.atomic_json(STUDY / 'results.json', {
        'scope': 'development-only', 'frames': len(records), 'patients': len(grouped), 'overall': overall,
        'paired_patients': sorted(paired, key=lambda row: row['lab']), 'folds': evidence,
        'steps_per_fold': STEPS, 'seed': SEED, 'source_sha256': digest(Path(__file__)),
        'sam2_archive_version': source.name, 'sam2_weights_sha256': MODEL_SHA256,
    })
    print(overall, flush=True)


if __name__ == '__main__':
    main()
