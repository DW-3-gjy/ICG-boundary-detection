"""Development comparison of local SAM 2.1 prompted solely from frozen D probabilities."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys

import cv2
import numpy as np
import torch

from evaluate_static_d_color_context import average_prior, color_cut
from evaluate_static_d_mask_geometry import clean_islands
from evaluate_static_d_region_ensemble import HIGH_SIZE
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, load_runner, verify
from train_static_d_region_refiner import FOLDS, dice_at_original_size, read_frames

STUDY = LOGS / 'static_d_sam2_regions_20261124_01'
CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'
MODEL_SHA256 = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'


def far_apart(mask, count, separation):
    if not mask.any():
        return np.empty((0, 2), dtype=np.float32)
    rank = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    results = []
    for _ in range(count):
        y, x = np.unravel_index(rank.argmax(), rank.shape)
        if rank[y, x] < 1:
            break
        results.append((x, y))
        cv2.circle(rank, (x, y), separation, 0, thickness=-1)
    return np.asarray(results, dtype=np.float32).reshape(-1, 2)


def prompts(probability, mask, image_shape):
    h, w = mask.shape
    image_h, image_w = image_shape[:2]
    ratio = np.array([image_w / w, image_h / h], dtype=np.float32)
    clipped = probability.clip(.005, .995)
    mask_logit = .55 * np.log(clipped / (1 - clipped)) + np.where(mask, 2., -2.).astype(np.float32)
    prior = cv2.resize(mask_logit, (256, 256), interpolation=cv2.INTER_AREA)[None]
    shape = clean_islands(mask)
    x, y, box_w, box_h = cv2.boundingRect(shape.astype(np.uint8))
    if not box_w or not box_h:
        return prior, None, None, None
    pad = round(.04 * max(box_h, box_w))
    box = np.asarray([max(0, x - pad), max(0, y - pad), min(w - 1, x + box_w + pad),
                     min(h - 1, y + box_h + pad)], dtype=np.float32) * np.tile(ratio, 2)
    strong = shape & (probability >= .78)
    positives = far_apart(strong if strong.any() else shape, 4, round(min(h, w) * .13))
    spatial = cv2.distanceTransform((~shape).astype(np.uint8), cv2.DIST_L2, 5)
    negatives = far_apart((spatial > 6) & (spatial <= 70) & (probability <= .07), 6,
                          round(min(h, w) * .10))
    coords = np.concatenate((positives, negatives), axis=0) * ratio
    labels = np.r_[np.ones(len(positives), dtype=np.int32), np.zeros(len(negatives), dtype=np.int32)]
    return prior, coords, labels, box


def score_frame(predictor, frame, deep_prior):
    probability = cv2.resize(deep_prior, HIGH_SIZE, interpolation=cv2.INTER_AREA)
    label = color_cut(frame.rgb, deep_prior, HIGH_SIZE, 'lab')
    prior, coords, points, box = prompts(probability, label, frame.rgb.shape)
    if box is None:
        return dict(lab=label, sam2_mask=label, sam2_prompted=label, filtered_sam2=label)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                               enabled=torch.cuda.is_bf16_supported()):
        predictor.set_image(frame.rgb)
        mask_output, _, _ = predictor.predict(mask_input=prior, multimask_output=False)
        result, predicted_iou, _ = predictor.predict(point_coords=coords, point_labels=points, box=box,
                                                     mask_input=prior, multimask_output=False)

    def to_high(output):
        return cv2.resize(output[0].astype(np.uint8), HIGH_SIZE, interpolation=cv2.INTER_NEAREST).astype(bool)

    segment = to_high(result)
    agreement = 2 * np.count_nonzero(segment & label) / max(1, np.count_nonzero(segment) + np.count_nonzero(label))
    uncertain = (probability >= .12) & (probability <= .90)
    blended = np.where(uncertain, segment, label) if agreement >= .65 and float(predicted_iou[0]) >= .60 else label
    return {'lab': label, 'sam2_mask': to_high(mask_output), 'sam2_prompted': segment,
            'filtered_sam2': blended}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    source, deps, weights = (path.resolve() for path in (args.source, args.dependencies, args.weights))
    if digest(weights) != MODEL_SHA256:
        raise ValueError('The pinned public SAM 2.1 weights do not match')
    source_config = source / 'sam2' / CONFIG
    if not source_config.is_file() or not deps.is_dir():
        raise ValueError('The pinned local SAM 2.1 source and runtime dependencies are required')
    sys.path[:0] = [str(source), str(deps)]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    development, _ = verify(BASELINE)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    STUDY.mkdir(exist_ok=True)
    runner.forbid_test_access(STUDY)
    predictor = SAM2ImagePredictor(build_sam2(CONFIG, str(weights), device='cuda', apply_postprocessing=False))
    closed = [row for row in development if row['annotation_kind'] != 'linestrip']
    frames = read_frames(runner, closed)
    large = {sid: cv2.resize(frame.rgb, HIGH_SIZE, interpolation=cv2.INTER_AREA) for sid, frame in frames.items()}
    scores = defaultdict(lambda: defaultdict(list))
    for fold in FOLDS:
        rows = [r for r in closed if r['planned_validation_fold'] == fold]
        probabilities = average_prior(runner, development, rows, frames, large, fold, 'D')
        for row in rows:
            sid, patient = row['sample_id'], row['patient_source_id']
            frame = frames[sid]
            for method, mask in score_frame(predictor, frame, probabilities[sid]).items():
                scores[method][patient].append(dice_at_original_size(mask, frame))
        print(f'{fold}: scored {len(rows)} SAM 2.1 comparisons.', flush=True)
    overall = {key: mean(mean(v) for v in patients.values()) for key, patients in scores.items()}
    if abs(overall['lab'] - .8771110420407898) > .0001:
        raise ValueError('The frozen Lab correction did not reproduce')
    order = sorted(scores['lab'], key=lambda p: mean(scores['lab'][p]))
    runner.atomic_json(STUDY / 'results.json', {
        'scope': 'development-only', 'frames': len(closed), 'patients': len(order), 'overall': overall,
        'paired_patients': [{key: mean(v[p]) for key, v in scores.items()} for p in order],
        'source_sha256': digest(Path(__file__)), 'sam2_archive_version': source.name,
        'sam2_weights_sha256': MODEL_SHA256, 'sam2_config_sha256': digest(source_config),
    })
    print(overall, flush=True)


if __name__ == '__main__':
    main()
