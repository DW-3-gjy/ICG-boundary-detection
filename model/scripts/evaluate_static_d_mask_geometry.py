"""Check foreground island cleanup and seed-guided mask recovery on development folds."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

from evaluate_static_d_region_ensemble import HIGH_SIZE, SEEDS, predict_frame
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, load_runner, verify
from train_static_d_region_dice import coarse_checkpoint
from train_static_d_region_refiner import FOLDS, dice_at_original_size, read_frames

OUTPUT = LOGS / 'static_d_auxiliary_dice_20261124_01' / 'diagnostics' / 'mask_geometry.json'
REGION_SIZE = (768, 432)


def clean_islands(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_component = max(16, .006 * int(areas.max()))
    kept = np.r_[False, areas >= min_component][labels]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~kept).astype(np.uint8), connectivity=8)
    if count <= 1:
        return kept
    h, w = mask.shape
    left, top, width, height, area = stats[1:].T
    inside = (left > 0) & (top > 0) & (left + width < w) & (top + height < h)
    fill = inside & (area <= max(24, .012 * np.count_nonzero(kept)))
    return kept | np.r_[False, fill][labels]


def seed_guided_growth(probability):
    count, labels = cv2.connectedComponents((probability >= .28).astype(np.uint8), connectivity=8)
    if count <= 1:
        return probability >= .5
    seeds = np.bincount(labels[probability >= .74], minlength=count)
    keep = seeds >= 24
    keep[0] = False
    return clean_islands(keep[labels] | (probability >= .5))


def color_guided_cut(frame, probability):
    prob = cv2.resize(probability, REGION_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(frame.rgb, REGION_SIZE, interpolation=cv2.INTER_AREA)
    labels = np.where(prob >= .50, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    labels[prob >= .94] = cv2.GC_FGD
    labels[prob <= .025] = cv2.GC_BGD
    if not (labels == cv2.GC_FGD).any() or not (labels == cv2.GC_BGD).any():
        return prob >= .5
    cv2.grabCut(rgb, labels, None, np.zeros((1,65), np.float64), np.zeros((1,65), np.float64),
                2, cv2.GC_INIT_WITH_MASK)
    return (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)


def compare():
    development, _ = verify(BASELINE)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    detection = [row for row in development if row['annotation_kind'] != 'linestrip']
    frames = read_frames(runner, detection)
    large = {sample: cv2.resize(frame.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA)
             for sample, frame in frames.items()}
    network = runner.MixedBoundaryUNet(True, head_seed=0, encoder_weights=None).cuda().eval()
    scores = defaultdict(lambda: defaultdict(list))
    for fold in FOLDS:
        rows = [row for row in detection if row['planned_validation_fold'] == fold]
        average = dict.fromkeys((row['sample_id'] for row in rows), 0)
        for seed in SEEDS:
            path = coarse_checkpoint(development, fold, seed)
            network.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
            for row in rows:
                sid = row['sample_id']
                _, _, probability = predict_frame(network, frames[sid], large[sid], runner)
                average[sid] += probability / len(SEEDS)
        for row in rows:
            sid = row['sample_id']
            frame = frames[sid]
            probability = average[sid]
            base = probability >= .5
            arms = {'base': base, 'clean_islands': clean_islands(base),
                    'seed_guided_growth': seed_guided_growth(probability),
                    'color_guided_cut': color_guided_cut(frame, probability)}
            for key, mask in arms.items():
                scores[key][row['patient_source_id']].append(dice_at_original_size(mask,frame))
        print(f'{fold}: scored {len(rows)} complete masks.', flush=True)
    all_results = {key:mean(mean(frames) for frames in values.values()) for key,values in scores.items()}
    if abs(all_results['base'] - .8566076624633716) > .0001:
        raise ValueError('The previous region ensemble did not reproduce')
    order = sorted(scores['base'], key=lambda patient:mean(scores['base'][patient]))
    patients = [{key:mean(values[patient]) for key,values in scores.items()} for patient in order]
    runner.atomic_json(OUTPUT, {'scope':'development-only', 'frames':len(detection),'patients':len(patients),
                     'evaluator_sha256':digest(Path(__file__)),'overall':all_results,
                     'patient_pairs_sorted_by_base_dice':patients})
    print(all_results, flush=True)


if __name__ == '__main__':
    compare()
