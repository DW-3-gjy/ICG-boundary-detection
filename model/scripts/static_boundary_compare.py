"""Train and compare static ICG refiners on patient-disjoint image sets."""
import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from icg_pipeline.boundary_study import (
    BoundaryReference, MixedBoundaryLoss, MixedBoundaryUNet, atomic_json, file_sha256,
    mask_boundary, normalized_image, patient_macro, rasterize_shapes, reduced_boundary,
    reference_points, seed_all,
)
from icg_pipeline.data import read_image, write_image
from icg_pipeline.refinement import (
    BoundaryStripRefiner, RefinerConfig, curve_geometry, load_refiner, make_targets,
    sample_strip_source, save_refiner, strip_source,
)
from icg_pipeline.static_boundary import (
    base_region_mask, coarse_probabilities, corrected_region_mask, curve_points,
    load_coarse, refine_open_curves, sampled_curve, trace_candidates,
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    patient: str
    split: str
    image: Path
    annotation: Path
    kind: str
    valid_mask: Path | None
    complete_boundary: bool


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids, patients, image_splits = set(), {}, {}
    groups = {split: [] for split in ('coarse', 'train', 'val', 'test')}
    for row in rows:
        required = ('sample_id', 'patient', 'split', 'image', 'annotation', 'kind')
        if not all(row.get(key) for key in required):
            raise ValueError('Required CSV columns: ' + ', '.join(required))
        sample_id, patient, split = (row[key] for key in required[:3])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sample_id) or sample_id in ids:
            raise ValueError(f'Invalid or duplicate sample ID: {sample_id}')
        ids.add(sample_id)
        if split not in groups or patients.setdefault(patient, split) != split:
            raise ValueError(f'Unknown split or patient crosses splits: {sample_id}')
        kind = row['kind']
        if kind not in ('labelme', 'region_mask', 'boundary_mask', 'tristate'):
            raise ValueError(f'Unsupported annotation kind: {kind}')
        image = path.parent / row['image']
        annotation = path.parent / row['annotation']
        valid = path.parent / row['valid_mask'] if row.get('valid_mask') else None
        for source in (image, annotation, valid):
            if source is not None and not source.is_file():
                raise FileNotFoundError(source)
        digest = file_sha256(image)
        if image_splits.setdefault(digest, split) != split:
            raise ValueError(f'Identical image bytes in different splits: {sample_id}')
        complete = (row.get('complete_boundary') or '').strip().lower()
        if complete not in ('', '0', '1', 'false', 'true'):
            raise ValueError(f'Use true or false for complete_boundary: {sample_id}')
        groups[split].append(Sample(sample_id, patient, split, image, annotation, kind,
                                    valid, complete in ('1', 'true')))
    if not all(groups[split] for split in ('train', 'val', 'test')):
        raise ValueError('Patient-disjoint train, val and test are required')
    return groups


def load_checkpoints(path, groups):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding='utf-8-sig'))
    by_patient = {}
    assignment = {s.patient: s.split for key in ('train', 'val', 'test') for s in groups[key]}
    shared = {}
    for entry in plan:
        checkpoint = (path.parent / entry['checkpoint']).resolve()
        fitted = set(entry['fitted_patients'])
        predicted = set(entry['predict_patients'])
        if not fitted or not predicted or not checkpoint.is_file():
            raise ValueError('Each coarse checkpoint requires training provenance and assigned patients')
        identity = file_sha256(checkpoint)
        if identity in shared and shared[identity] != fitted:
            raise ValueError('One checkpoint cannot have inconsistent training provenance')
        shared[identity] = fitted
        if fitted & (predicted | {p for p, split in assignment.items() if split in ('val', 'test')}):
            raise ValueError('A coarse checkpoint used its prediction, validation or test patient for fitting')
        for patient in predicted:
            if patient in by_patient or patient not in assignment:
                raise ValueError(f'Duplicate or unexpected coarse routing for {patient}')
            by_patient[patient] = checkpoint
    if set(by_patient) != set(assignment):
        raise ValueError('Coarse routing must cover every train, val and test patient')
    if len({file_sha256(by_patient[p]) for p, split in assignment.items() if split in ('val', 'test')}) != 1:
        raise ValueError('Validation and test must use the same coarse checkpoint')
    return by_patient


def read_sample(sample, image=None):
    if image is None:
        image = read_image(sample.image)
    h, w = image.shape[:2]
    if sample.kind == 'labelme':
        doc = json.loads(sample.annotation.read_text(encoding='utf-8-sig'))
        region, line, has_region = rasterize_shapes(doc, w, h)
    else:
        raw = read_image(sample.annotation, grayscale=True)
        if raw.shape != (h, w):
            raise ValueError(f'Annotation size mismatch: {sample.sample_id}')
        has_region = sample.kind == 'region_mask'
        region = (raw > 0).astype(np.uint8) if has_region else np.zeros((h, w), np.uint8)
        line = mask_boundary(region) if has_region else (raw == 255 if sample.kind == 'tristate' else raw > 0)
    line = line.astype(bool)
    if not line.any():
        raise ValueError(f'No reference line: {sample.sample_id}')
    known = np.full((h, w), sample.complete_boundary, dtype=bool)
    if sample.valid_mask:
        known = read_image(sample.valid_mask, grayscale=True) > 0
        if known.shape != (h, w) or not known[line].all():
            raise ValueError(f'Validity mask must cover the full reference line: {sample.sample_id}')
    if sample.kind == 'tristate':
        if not np.isin(raw, (0, 128, 255)).all():
            raise ValueError(f'Tri-state annotation must contain only 0, 128, 255: {sample.sample_id}')
        known = (raw != 128) & known if sample.valid_mask else raw != 128
    labels = np.where(line, 255, np.where(known, 0, 128)).astype(np.uint8)
    # A reviewed scoring mask is required for incomplete annotations.
    if sample.split in ('val', 'test') and not (sample.complete_boundary or sample.valid_mask or sample.kind == 'tristate'):
        raise ValueError(f'Incomplete evaluation reference needs a validity mask: {sample.sample_id}')
    dice_reference = region if has_region and sample.valid_mask is None else None
    return image, labels, dice_reference


def patient_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.patient, []).append(row)
    return list(grouped.values())


def draw_sample(groups, rng):
    rows = groups[rng.integers(len(groups))]
    return rows[rng.integers(len(rows))]


def fit_coarse(rows, args, output):
    if not rows or any(not row.complete_boundary or row.valid_mask or row.kind == 'tristate' for row in rows):
        raise ValueError('Coarse fitting requires a separate coarse split with complete visible-boundary labels')
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    grouped = patient_rows(rows)
    model = MixedBoundaryUNet(True, args.seed).to(args.device)
    criterion = MixedBoundaryLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    @lru_cache(maxsize=None)
    def tensors(row):
        image, labels, region = read_sample(row)
        rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), args.input_size, interpolation=cv2.INTER_AREA)
        coarse_line = reduced_boundary(labels == 255, args.input_size)
        coarse_region = np.zeros_like(coarse_line) if region is None else cv2.resize(
            region, args.input_size, interpolation=cv2.INTER_NEAREST)
        return (normalized_image(rgb)[None], torch.from_numpy(coarse_region[None].astype(np.int64)),
                torch.from_numpy(coarse_line[None].astype(np.float32)), torch.tensor([region is not None]))

    model.train()
    for update in range(1, args.coarse_updates + 1):
        image, target, line, has_region = (x.to(args.device) for x in tensors(draw_sample(grouped, rng)))
        optimizer.zero_grad(set_to_none=True)
        loss, _ = criterion(model(image), target, line, has_region)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if update % 100 == 0 or update == args.coarse_updates:
            print(f'coarse {update}/{args.coarse_updates} loss={loss.item():.4f}', flush=True)
    torch.save({'model': model.state_dict(), 'threshold': args.coarse_threshold,
                'input_size': args.input_size}, output)


def prepare_case(sample, image, region, boundary, threshold, args):
    image, labels, target_region = read_sample(sample, image)
    curves = trace_candidates(boundary, threshold, image.shape)
    candidate = trace_candidates(boundary, args.candidate_threshold, image.shape) if args.candidate_threshold else curves
    return image, labels, target_region, region, boundary, curves, candidate


def score_case(sample, case, fragments, mask, step, elapsed_ms, changed_pixels=0):
    _, labels, target_region, *_ = case
    reference = BoundaryReference(reference_points(labels == 255), (labels.shape[1], labels.shape[0]))
    points = curve_points(fragments, step)
    if len(points):
        xx, yy = np.rint(points).astype(int).T
        points = points[labels[yy, xx] != 128]
    at10 = reference.score(points, tolerance=10, distances=True)
    at20 = reference.score(points, tolerance=20)
    dice = None
    if target_region is not None:
        denominator = np.count_nonzero(mask) + np.count_nonzero(target_region)
        dice = float(2 * np.count_nonzero(mask & target_region) / denominator) if denominator else 1.
    return {'sample_id': sample.sample_id, 'patient': sample.patient, 'f1_10': at10['f1'],
            'f1_20': at20['f1'], 'precision_20': at20['precision'], 'recall_20': at20['recall'],
            'assd': at10['assd'], 'hd95': at10['hd95'], 'empty': at10['empty_prediction'],
            'region_dice': dice, 'region_changed_pixels': changed_pixels, 'refine_ms': elapsed_ms}


def make_runner(routing, args):
    @lru_cache(maxsize=1)
    def model_at(path):
        return load_coarse(path, args.device, args.input_size)

    @lru_cache(maxsize=2)
    def case_for(sample):
        model, threshold = model_at(routing[sample.patient])
        img = read_image(sample.image)
        region, boundary = coarse_probabilities(model, img, args.input_size, args.device)
        return prepare_case(sample, img, region, boundary, threshold, args)

    return case_for


def fit_refiners(train_rows, val_rows, case_for, args, output):
    # Store small probability grids for repeatable experiments and load raw images lazily.
    prepared = {}
    for sample in sorted([*train_rows, *val_rows], key=lambda s: s.patient):
        _, _, _, region, boundary, curves, candidates = case_for(sample)
        prepared[sample] = region, boundary, curves, candidates
    training = {}
    target_classes = 2 * args.radius + 1
    for sample in train_rows:
        _, _, _, candidates = prepared[sample]
        _, labels, _ = read_sample(sample)
        cfg = RefinerConfig(radius=args.radius, tangent_radius=args.tangent_radius)
        strokes = [sampled_curve(curve, args.step) for curve in candidates if len(curve) >= 2]
        centers, normals, _ = curve_geometry(strokes)
        targets = make_targets(labels, centers, normals, cfg)
        usable = targets != -100
        if usable.any():
            training[sample] = centers[usable], normals[usable], targets[usable]
    positives = sum(np.count_nonzero(data[2] != target_classes) for data in training.values())
    negatives = sum(np.count_nonzero(data[2] == target_classes) for data in training.values())
    if not positives or not negatives:
        raise ValueError('Refiner requires on-window positive offsets and reviewed negatives. Check annotations and candidates.')
    print(f'Refiner strips: {positives} positive, {negatives} reviewed negative', flush=True)
    grouped = patient_rows(training)

    @lru_cache(maxsize=1)
    def image_source(row):
        image = read_image(row.image)
        probability = cv2.resize(prepared[row][1], image.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        return strip_source(image, probability, args.device)

    checkpoints = {}
    for variant in ('local', 'bilateral'):
        seed_all(args.seed)
        rng = np.random.default_rng(args.seed)
        config = RefinerConfig(radius=args.radius, tangent_radius=args.tangent_radius,
                               batch_size=args.batch_size, variant=variant,
                               max_link_distance=max(12., 2.5 * args.step),
                               min_position_mass=args.position_mass if variant == 'bilateral' else 0.)
        model = BoundaryStripRefiner(config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        path, best = output / f'{variant}.pt', -1.
        for epoch in range(1, args.epochs + 1):
            model.train()
            for _ in range(args.steps_per_epoch):
                sample = draw_sample(grouped, rng)
                centers, normals, targets = training[sample]
                index = rng.choice(len(targets), min(len(targets), args.batch_size), replace=False)
                strips = sample_strip_source(image_source(sample), centers[index], normals[index], config)
                labels = torch.as_tensor(targets[index], device=args.device)
                if rng.random() < .5:
                    strips = strips.flip((2, 3))
                    labels = torch.where(labels == target_classes, labels, target_classes - 1 - labels)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(strips), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
            results = []
            for sample in val_rows:
                image, labels, target = read_sample(sample)
                region, boundary, curves, candidates = prepared[sample]
                _, ref = refine_open_curves(model, image, boundary, candidates, args.step)
                case = (image, labels, target, region, boundary, curves, candidates)
                results.append(score_case(sample, case, ref['fragments'], None if target is None else
                                          base_region_mask(region, image.shape), args.step, None))
            score = patient_macro(results, 'f1_10')['mean']
            print(f'{variant} {epoch}/{args.epochs} val patient F1@10={score:.4f}', flush=True)
            if score > best:
                save_refiner(path, model, {'seed': args.seed, 'epoch': epoch, 'validation_f1_10': score})
                best = score
        checkpoints[variant] = path
    return checkpoints


def evaluate(case_for, rows, checkpoints, args, output):
    models = {name: load_refiner(path, args.device) for name, path in checkpoints.items()}
    records = {name: [] for name in ('coarse_d', 'local', 'bilateral', 'bilateral_region')}
    for sample in rows:
        case = case_for(sample)
        image, _, _, region, boundary, curves, candidates = case
        mask = base_region_mask(region, image.shape)
        records['coarse_d'].append(score_case(sample, case, curves, mask, args.step, 0.))
        for name, model in models.items():
            started = time.perf_counter()
            ordered, ref = refine_open_curves(model, image, boundary, candidates, args.step)
            refined_ms = (time.perf_counter() - started) * 1000
            records[name].append(score_case(sample, case, ref['fragments'], mask, args.step, refined_ms))
            if name == 'bilateral':
                started = time.perf_counter()
                corrected = corrected_region_mask(region, mask, ordered, ref, model.config)
                elapsed = refined_ms + (time.perf_counter() - started) * 1000
                changed = int(np.count_nonzero(corrected != mask))
                records['bilateral_region'].append(score_case(sample, case, ref['fragments'], corrected,
                                                              args.step, elapsed, changed))
    summary = {}
    for name, results in records.items():
        with (output / f'{name}_per_image.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        summary[name] = {key: patient_macro(results, key)['mean'] for key in (
            'f1_10', 'f1_20', 'precision_20', 'recall_20', 'assd', 'hd95', 'region_dice')}
        summary[name].update(empty_predictions=sum(r['empty'] for r in results),
                             evaluated_frames=len(results), evaluated_patients=len({r['patient'] for r in results}),
                             dice_frames=sum(r['region_dice'] is not None for r in results),
                             dice_patients=len({r['patient'] for r in results if r['region_dice'] is not None}),
                             region_changed_frames=sum(r['region_changed_pixels'] > 0 for r in results),
                             refine_ms_median=float(np.median([r['refine_ms'] for r in results])))
    atomic_json(output / 'results.json', summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


def compare(args):
    groups = load_manifest(args.manifest)
    if args.coarse_plan:
        routing = load_checkpoints(args.coarse_plan, groups)
    elif not groups['coarse']:
        raise ValueError('Supply --coarse-plan or add coarse patients to the manifest')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / '.gitignore').write_text('*\n', encoding='utf-8')
    if not args.coarse_plan:
        coarse = output / 'coarse.pt'
        fit_coarse(groups['coarse'], args, coarse)
        routing = {s.patient: coarse for key in ('train', 'val', 'test') for s in groups[key]}
    config = {key: value for key, value in vars(args).items() if key != 'run'}
    config.update(manifest_sha256=file_sha256(args.manifest),
                  coarse_sha256={p: file_sha256(path) for p, path in routing.items()})
    atomic_json(output / 'config.json', config)
    case_for = make_runner(routing, args)
    checkpoints = fit_refiners(groups['train'], groups['val'], case_for, args, output)
    atomic_json(output / 'locked_refiners.json', {name: file_sha256(path) for name, path in checkpoints.items()})
    evaluate(case_for, groups['test'], checkpoints, args, output)


def predict(args):
    image = read_image(args.image)
    coarse, threshold = load_coarse(args.coarse, args.device, args.input_size)
    region, boundary = coarse_probabilities(coarse, image, args.input_size, args.device)
    candidates = trace_candidates(boundary, args.candidate_threshold or threshold, image.shape)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / '.gitignore').write_text('*\n', encoding='utf-8')
    mask = base_region_mask(region, image.shape)
    if args.refiner:
        model = load_refiner(args.refiner, args.device)
        ordered, result = refine_open_curves(model, image, boundary, candidates, args.step)
        if args.correct_region:
            mask = corrected_region_mask(region, mask, ordered, result, model.config)
        fragments = result['fragments']
    else:
        fragments = [curve.tolist() for curve in candidates]
    atomic_json(output / 'boundary.json', {'fragments': fragments, 'coordinate_system': 'original_xy',
                                          'width': image.shape[1], 'height': image.shape[0]})
    write_image(output / 'region_mask.png', mask * 255)
    for fragment in fragments:
        points = np.rint(fragment).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points], False, (0, 220, 255), 2, cv2.LINE_AA)
        elif len(points):
            cv2.circle(image, tuple(points[0]), 1, (0, 220, 255), -1)
    write_image(output / 'overlay.png', image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='command', required=True)
    train = modes.add_parser('compare')
    train.add_argument('--manifest', required=True)
    train.add_argument('--coarse-plan')
    train.add_argument('--coarse-updates', type=int, default=2400)
    train.add_argument('--coarse-threshold', type=float, default=.5)
    train.add_argument('--radius', type=int, default=32)
    train.add_argument('--tangent-radius', type=int, default=16)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--epochs', type=int, default=12)
    train.add_argument('--steps-per-epoch', type=int, default=100)
    train.add_argument('--position-mass', type=float, default=.20)
    train.add_argument('--seed', type=int, default=42)
    train.set_defaults(run=compare)
    infer = modes.add_parser('predict')
    infer.add_argument('--image', required=True)
    infer.add_argument('--coarse', required=True)
    infer.add_argument('--refiner')
    infer.add_argument('--correct-region', action='store_true')
    infer.set_defaults(run=predict)
    for mode in (train, infer):
        mode.add_argument('--input-size', type=int, metavar=('WIDTH', 'HEIGHT'), nargs=2, default=(768, 432))
        mode.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        mode.add_argument('--step', type=float, default=5.)
        mode.add_argument('--candidate-threshold', type=float)
        mode.add_argument('--output', required=True)
    args = parser.parse_args()
    if (min(args.input_size) < 32 or not np.isfinite(args.step) or args.step <= 0
            or (args.candidate_threshold is not None and not 0 < args.candidate_threshold < 1)):
        parser.error('Require input dimensions >=32, positive step and threshold in (0, 1)')
    if args.command == 'compare' and (min(args.epochs, args.steps_per_epoch, args.coarse_updates,
                                         args.radius, args.tangent_radius, args.batch_size) < 1
                                      or not 0 < args.coarse_threshold < 1 or not 0 <= args.position_mass <= 1):
        parser.error('Invalid training budget, strip dimensions or confidence thresholds')
    if args.command == 'predict' and args.correct_region and not args.refiner:
        parser.error('--correct-region requires --refiner')
    torch.set_num_threads(2)
    args.run(args)


if __name__ == '__main__':
    main()
