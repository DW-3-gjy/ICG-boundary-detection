"""Compare SAM 2.1 Large with the locked Tiny development control by patient fold."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

import cv2
import torch

import train_static_d_sam2_decoder as control
from evaluate_static_d_color_context import average_prior
from evaluate_static_d_region_ensemble import HIGH_SIZE
from refit_static_d_sam2_regions import load_sam, prepare_image
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, exclusive_training, load_runner, read_json, verify, verify_file
from train_static_d_region_refiner import FOLDS, dice_at_original_size, read_frames
from train_static_d_sam2_decoder import decode, mask_from_logits

STUDY = LOGS / 'static_d_sam2_large_20261124_01'
LARGE_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
LARGE_SHA256 = '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
METHODS = ('lab', 'tiny_blend', 'large', 'large_flip_blend', 'tiny_large_blend', 'tiny_large_flip_blend')


@torch.no_grad()
def compare(frame, prior, tiny, tiny_position, tiny_official, tiny_adapted, large, large_position):
    prepared = prepare_image(tiny, frame, prior, False)
    masks = dict.fromkeys(METHODS, prepared.prior_mask)
    if not prepared.prompt_embeddings:
        return masks

    tiny_decoder = tiny.model.sam_mask_decoder
    tiny_decoder.load_state_dict(tiny_official)
    tiny_frozen = decode(tiny_decoder, tiny_position, prepared, 0)
    tiny_decoder.load_state_dict(tiny_adapted)
    tiny_mix = (tiny_frozen + decode(tiny_decoder, tiny_position, prepared, 0)) / 2

    large_decoder = large.model.sam_mask_decoder
    standard = prepare_image(large, frame, prior, False)
    large_logits = decode(large_decoder, large_position, standard, 0)
    mirror_frame = replace(frame, rgb=cv2.flip(frame.rgb, 1))
    mirrored = prepare_image(large, mirror_frame, cv2.flip(prior, 1), False)
    if mirrored.prompt_embeddings:
        recovered = decode(large_decoder, large_position, mirrored, 0).flip(dims=(-1,))
        large_mix = (large_logits + recovered) / 2
    else:
        large_mix = large_logits
    outputs = (tiny_mix, large_logits, large_mix, (tiny_mix + large_logits) / 2, (tiny_mix + large_mix) / 2)
    for key, value in zip(METHODS[1:], outputs, strict=True):
        masks[key] = mask_from_logits(value, tiny, frame)
    return masks


def run(runner, source, deps, weights, large_weights):
    previous = read_json(control.STUDY / 'results.json')
    verify_file(Path(control.__file__), previous['source_sha256'])
    verify_file(large_weights, LARGE_SHA256)
    for study in (ORIGINAL, BASELINE, STUDY):
        runner.forbid_test_access(study)
    development, _ = verify(BASELINE)
    closed = [r for r in development if r['annotation_kind'] != 'linestrip']
    frames = read_frames(runner, closed)
    large_frames = {sid: cv2.resize(frame.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA) for sid, frame in frames.items()}
    tiny = load_sam(source, deps, weights)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    large = SAM2ImagePredictor(build_sam2(LARGE_CONFIG, str(large_weights), device='cuda', apply_postprocessing=False))
    large.model.requires_grad_(False)
    tiny_official = {k: v.detach().cpu().clone() for k, v in tiny.model.sam_mask_decoder.state_dict().items()}
    with torch.no_grad():
        tiny_position = tiny.model.sam_prompt_encoder.get_dense_pe()
        large_position = large.model.sam_prompt_encoder.get_dense_pe()
    records = []
    for fold in FOLDS:
        rows = [r for r in closed if r['planned_validation_fold'] == fold]
        path = control.STUDY / f'{fold}_decoder.pt'
        verify_file(path, previous['folds'][fold]['checkpoint_sha256'])
        tiny_adapted = torch.load(path, map_location='cpu', weights_only=True)['decoder']
        priors = average_prior(runner, development, rows, frames, large_frames, fold, 'D')
        for row in rows:
            sid = row['sample_id']
            frame = frames[sid]
            masks = compare(frame, priors[sid], tiny, tiny_position, tiny_official, tiny_adapted, large, large_position)
            records.append({'patient': row['patient_source_id'], 'sample': sid, 'fold': fold,
                            **{name: dice_at_original_size(mask, frame) for name, mask in masks.items()}})
        print(f'{fold}: compared {len(rows)} complete masks.', flush=True)

    patients = defaultdict(list)
    for row in records:
        patients[row['patient']].append(row)
    paired = [{metric: mean(r[metric] for r in rows) for metric in METHODS} for rows in patients.values()]
    overall = {metric: mean(r[metric] for r in paired) for metric in METHODS}
    if len(records) != len(closed) or len(paired) != 5 or abs(overall['tiny_blend'] - previous['overall']['blended_sam2']) > .0001:
        raise ValueError('Development cohort or Tiny control did not reproduce')
    runner.atomic_json(STUDY / 'scores.private.json', records)
    runner.atomic_json(STUDY / 'results.json', {
        'scope': 'development-only', 'frames': len(closed), 'patients': len(paired),
        'overall': overall, 'paired_patients': sorted(paired, key=lambda row: row['tiny_blend']),
        'source_sha256': digest(Path(__file__)), 'control_sha256': digest(control.STUDY / 'results.json'),
        'large_weight_sha256': LARGE_SHA256, 'large_config_sha256': digest(source / 'sam2' / LARGE_CONFIG),
    })
    print(overall, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--large-weights', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    STUDY.mkdir(exist_ok=True)
    with exclusive_training(STUDY):
        if (STUDY / 'results.json').exists():
            raise FileExistsError('The SAM 2.1 Large evaluation already completed')
        run(runner, args.source.resolve(), args.dependencies.resolve(), args.weights.resolve(), args.large_weights.resolve())


if __name__ == '__main__':
    main()
