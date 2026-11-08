"""Execute the pre-specified 2026-11-05 patient-disjoint boundary study.

Preparation is separate from model development. Development cannot open sealed
test caches. All final models are locked before a single final test stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT / 'src'))

import cv2
import numpy as np
import torch
from icg_pipeline.boundary_study import (
    BoundaryReference, CachedTrainingData, MixedBoundaryLoss, MixedBoundaryUNet,
    atomic_json, file_sha256, mask_boundary, normalized_image, patient_macro,
    prediction_points, rasterize_shapes, reduced_boundary, reference_points,
    seed_all, train_records,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty result table')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare(args):
    study = Path(args.study).resolve()
    if study.exists():
        raise ValueError('Study directory already exists; do not overwrite evidence')
    root = MODEL_ROOT.parents[0]
    plan = Path(args.plan).resolve()
    protocol = read_json(plan / 'protocol.json')
    intake_dir = MODEL_ROOT / 'data/intake_20261103_02'
    intake = read_json(intake_dir / 'intake_report.private.json')
    source = Path(intake['source_root'])
    original = MODEL_ROOT / 'data/cross_patient_20261022_01'
    old = {r['filename'].removesuffix('.png'): r for r in read_json(original / 'manifest.private.json')['records']}
    new = {r['sample_id']: r for r in intake['images']}
    candidates = read_csv(plan / 'candidate_manifest.csv')
    if not args.identities_confirmed or not args.visible_boundaries_complete:
        raise ValueError('Explicit source confirmations required')
    protocol.update({
        'study_id': study.name, 'design_status': 'frozen_ready_for_training',
        'data_freeze_status': 'frozen_with_user_confirmed_identity_and_visible_boundary_completeness',
        'globally_distinct_patients_not_yet_confirmed': False,
        'user_confirmations_20261105': {'old_and_new_patients_all_distinct': True,
            'all_requested_visible_boundaries_annotated': True,
            'missing_region_is_due_to_occlusion_or_incomplete_field_of_view': True,
            'independent_clinical_double_annotation_performed': False,
            'source': 'explicit user answers in this conversation, 2026-11-05'},
        'protocol_amendment_before_training': {
            'version': '1.1',
            'reference_scope': 'delivered visible-line references and polygon-derived region contours; includes instrument/contact/crop segments in polygons; not isolated anatomical-interface truth',
            'boundary_negative_scope': 'outside uncertainty bands of all delivered visible boundaries, based on user completeness confirmation; no amodal target is invented',
            'boundary_geometry': 'native-grid skeleton point centers mapped to original 4K coordinates, matched against full-resolution reference points; 5 original pixels per network grid cell',
            'region_loss_normalization': 'mean over complete-mask images in each microbatch; absent region references never evaluated',
        },
        'data_root': str(study / 'data'), 'training_started': False, 'final_test_started': False,
    })
    protocol['primary_metric']['scope'] = protocol['protocol_amendment_before_training']['reference_scope']
    study.mkdir(parents=True)
    shutil.copy2(plan / 'candidate_manifest.csv', study / 'candidate_manifest.csv')
    shutil.copy2(plan / 'protocol.json', study / 'original_design_protocol.json')
    manifests, private_paths = {}, []
    size = tuple(protocol['input_width_height'])
    for i, row in enumerate(candidates):
        sample = row['sample_id']
        if row['batch'] == '1102':
            src = new[sample]
            image_path = source / src['relative_path']
            annotation_path = source / src['annotation_relative']
            if file_sha256(image_path) != row['image_sha256'] or file_sha256(annotation_path) != row['annotation_or_mask_sha256']:
                raise ValueError(f'Source changed: {sample}')
            image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Unreadable: {sample}')
            h, w = image.shape[:2]
            region, boundary, has_region = rasterize_shapes(read_json(annotation_path), w, h)
        else:
            src = old[sample]
            image_path = original / src['files']['fullres_images']['path']
            annotation_path = original / src['files']['fullres_masks']['path']
            if file_sha256(image_path) != row['image_sha256'] or file_sha256(annotation_path) != row['annotation_or_mask_sha256']:
                raise ValueError(f'Historical source changed: {sample}')
            image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
            region = (cv2.imdecode(np.fromfile(str(annotation_path), np.uint8), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            h, w = image.shape[:2]
            boundary, has_region = mask_boundary(region), True
        if (w, h) != (3840, 2160):
            raise ValueError('Unexpected source resolution')
        directory = study / 'data' / ('sealed_test' if row['planned_partition'] == 'test' else 'development')
        directory.mkdir(parents=True, exist_ok=True)
        cache = directory / (sample + '.npz')
        np.savez_compressed(cache,
            image=cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA),
            region=cv2.resize(region, size, interpolation=cv2.INTER_NEAREST),
            boundary=reduced_boundary(boundary, size), has_region=np.array(has_region),
            reference_points=reference_points(boundary),
            packed_region=np.packbits(region.reshape(-1)) if has_region else np.array([], np.uint8),
            original_size=np.array([w, h], np.int32))
        manifests[cache.relative_to(study).as_posix()] = file_sha256(cache)
        private_paths.append({'sample_id': sample, 'image': str(image_path), 'annotation_or_mask': str(annotation_path)})
        if (i + 1) % 10 == 0:
            print(f'Prepared {i + 1}/{len(candidates)} source pairs (preparation only, no model predictions)', flush=True)
    snapshot = study / 'code'
    shutil.copytree(MODEL_ROOT / 'src', snapshot / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (snapshot / 'scripts').mkdir()
    shutil.copy2(__file__, snapshot / 'scripts' / Path(__file__).name)
    code_hashes = {p.relative_to(snapshot).as_posix(): file_sha256(p) for p in snapshot.rglob('*.py')}
    cache_weight = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    protocol['official_encoder_weights_sha256'] = file_sha256(cache_weight)
    protocol['prepared_cache_manifest_sha256'] = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
    atomic_json(study / 'protocol.json', protocol)
    atomic_json(study / 'cache_manifest.json', manifests)
    atomic_json(study / 'source_snapshot_manifest.json', code_hashes)
    atomic_json(study / 'source_paths.private.json', private_paths)
    atomic_json(study / 'environment.json', {'torch': torch.__version__, 'numpy': np.__version__,
        'opencv': cv2.__version__, 'gpu': torch.cuda.get_device_name(0),
        'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
        'python': sys.version, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    atomic_json(study / 'freeze.json', {'protocol_sha256': file_sha256(study / 'protocol.json'),
        'candidate_manifest_sha256': file_sha256(study / 'candidate_manifest.csv'),
        'cache_manifest_sha256': file_sha256(study / 'cache_manifest.json'),
        'source_snapshot_manifest_sha256': file_sha256(study / 'source_snapshot_manifest.json'),
        'source_pairs': len(candidates), 'preparation_reads_test_sources': True,
        'test_model_predictions_made': False, 'test_labels_used_for_model_selection': False})
    print(json.dumps({'prepared': str(study), 'pairs': len(candidates), 'frozen': True}), flush=True)


def verify_frozen(study):
    freeze = read_json(study / 'freeze.json')
    for name in ['protocol', 'candidate_manifest', 'cache_manifest', 'source_snapshot_manifest']:
        suffix = '.csv' if name == 'candidate_manifest' else '.json'
        if file_sha256(study / (name + suffix)) != freeze[name + '_sha256']:
            raise ValueError(f'Frozen metadata changed: {name}')
    for relative, expected in read_json(study / 'source_snapshot_manifest.json').items():
        if file_sha256(study / 'code' / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')


def verify_caches(study, include_test=False):
    for relative, expected in read_json(study / 'cache_manifest.json').items():
        if '/sealed_test/' in relative and not include_test:
            continue
        if file_sha256(study / relative) != expected:
            raise ValueError(f'Prepared data changed: {relative}')


def forbid_test_access(study):
    forbidden = os.path.normcase(str(study / 'data/sealed_test')).replace('\\', '/').rstrip('/') + '/'
    def hook(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.normcase(os.path.abspath(os.fsdecode(args[0]))).replace('\\', '/')
            if path.startswith(forbidden):
                raise PermissionError('Sealed test cache cannot be opened during development/refit')
    sys.addaudithook(hook)


def model_optimizer(seed, group, protocol):
    seed_all(seed)
    cache_weight = Path(torch.hub.get_dir()) / 'checkpoints/resnet18-f37072fd.pth'
    if file_sha256(cache_weight) != protocol['official_encoder_weights_sha256']:
        raise ValueError('ImageNet cache identity changed')
    model = MixedBoundaryUNet(group['boundary_head'], head_seed=seed + 100000).cuda()
    encoder_ids = {id(x) for x in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 3e-5},
        {'params': [x for x in model.parameters() if id(x) not in encoder_ids], 'lr': 3e-4}], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol['development_updates'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    # Shared stochastic layers use the same seed after isolated head creation.
    seed_all(seed + 200000)
    return model, optimizer, scheduler, scaler


def evaluate(model, study, rows, thresholds, distances=False, region_metrics=False):
    model.eval()
    by_threshold = {float(t): [] for t in thresholds}
    with torch.inference_mode():
        for row in rows:
            part = 'sealed_test' if row['planned_partition'] == 'test' else 'development'
            with np.load(study / 'data' / part / (row['sample_id'] + '.npz'), allow_pickle=False) as data:
                image = normalized_image(data['image'])[None].cuda()
                ref = BoundaryReference(data['reference_points'], data['original_size'])
                has_region = bool(data['has_region'])
                original = np.array(data['packed_region']) if region_metrics and has_region else None
                with torch.autocast('cuda', dtype=torch.float16):
                    output = model(image)
                p_region = output['logits'].float().softmax(1)[0, 1].cpu().numpy()
                source = 'boundary' if output['boundary_logits'] is not None else 'region'
                probability = output['boundary_logits'].float().sigmoid()[0].cpu().numpy() if source == 'boundary' else p_region
                for threshold in thresholds:
                    points = prediction_points(probability, threshold, source, ref.original_size)
                    metrics = ref.score(points, distances=distances)
                    result = {'sample_id': row['sample_id'], 'patient': row['patient_source_id'],
                        'stratum': row['task_stratum'], 'reference_kind': row['annotation_kind'],
                        'has_region_reference': has_region, **metrics}
                    if distances:
                        for tolerance in [10, 40]:
                            result[f'f1_{tolerance}px'] = ref.score(points, tolerance=tolerance)['f1']
                    if region_metrics:
                        if has_region:
                            w, h = ref.original_size
                            target = np.unpackbits(original, count=w * h).reshape(h, w).astype(bool)
                            pred = cv2.resize((p_region >= .5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            intersection = int(np.count_nonzero(pred & target))
                            denominator = int(pred.sum()) + int(target.sum())
                            union = int(np.count_nonzero(pred | target))
                            result.update(dice=2 * intersection / denominator if denominator else 1.,
                                          iou=intersection / union if union else 1.)
                        else:
                            result.update(dice=None, iou=None)
                    by_threshold[float(threshold)].append(result)
    return {threshold: {'patient_macro_f1': patient_macro(rows, 'f1')['mean'],
                         'per_patient': patient_macro(rows, 'f1')['by_patient'], 'per_image': rows}
            for threshold, rows in by_threshold.items()}


def atomic_checkpoint(path, state):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def rng_state(data):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'sampler': data.rng.bit_generator.state}


def restore_rng(value, data):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    torch.cuda.set_rng_state_all(value['cuda'])
    data.rng.bit_generator.state = value['sampler']


def train_one(study, protocol, records, group, seed, fold, max_updates, threshold=None, microbatch=1):
    tag = f"{group['id']}_{fold}_s{seed}"
    output_dir = study / 'runs' / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'complete.json').exists():
        return read_json(output_dir / 'complete.json')
    if (study / 'final_test_started.json').exists():
        raise ValueError('Training after final test start is forbidden')
    train_rows = train_records(records, group, None if fold == 'refit' else fold)
    val_rows = [r for r in records if r['planned_partition'] == 'development' and r['planned_validation_fold'] == fold]
    if fold != 'refit' and not val_rows:
        raise ValueError('No validation subjects')
    if {r['patient_source_id'] for r in train_rows} & {r['patient_source_id'] for r in val_rows}:
        raise ValueError('Patient leakage')
    model, optimizer, scheduler, scaler = model_optimizer(seed, group, protocol)
    data = CachedTrainingData(study / 'data', train_rows, seed)
    criterion = MixedBoundaryLoss().cuda()
    start, best_score, best_update, best_threshold = 0, -1., None, None
    history = []
    resume_path = output_dir / 'resume.pt'
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        if resume['protocol_sha256'] != file_sha256(study / 'protocol.json'):
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        scaler.load_state_dict(resume['scaler'])
        start, history = resume['update'], resume['history']
        best_score, best_update, best_threshold = resume['best_score'], resume['best_update'], resume['best_threshold']
        restore_rng(resume['rng'], data)
    else:
        atomic_json(output_dir / 'initialization.json', {'seed': seed,
            'encoder_weights_sha256': protocol['official_encoder_weights_sha256'],
            'shared_parameter_sha256': hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for n,p in model.named_parameters() if not n.startswith('boundary_head'))).hexdigest(),
            'warm_start_icg_checkpoint': False})
    atomic_json(output_dir / 'config.json', {'group': group, 'seed': seed, 'fold': fold,
        'max_updates': max_updates, 'microbatch': microbatch, 'effective_batch': 4,
        'training_ids': [r['sample_id'] for r in train_rows], 'validation_ids': [r['sample_id'] for r in val_rows],
        'protocol_sha256': file_sha256(study / 'protocol.json')})
    if 4 % microbatch:
        raise ValueError('Microbatch must divide 4')
    started = time.perf_counter()
    for update in range(start + 1, max_updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.
        components = None
        for _ in range(4 // microbatch):
            images, region, boundary, has_region = data.batch(microbatch, 'cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                loss, components = criterion(model(images), region, boundary, has_region)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at {tag}/{update}')
            running += float(loss.detach()) / (4 // microbatch)
            scaler.scale(loss / (4 // microbatch)).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=False)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise FloatingPointError(f'AMP overflow at {tag}/{update}; do not count a skipped update')
        scheduler.step()
        if update % 40 == 0 or update == max_updates:
            item = {'update': update, 'training_loss': running, 'last_microbatch_components': components,
                'gradient_norm': float(norm.detach()), 'elapsed_seconds_this_session': time.perf_counter() - started}
            if fold != 'refit':
                validation = evaluate(model, study, val_rows, protocol['boundary_probability_threshold_grid'])
                chosen_threshold, chosen = max(validation.items(), key=lambda p: (p[1]['patient_macro_f1'], -abs(p[0]-.5), -p[0]))
                score = chosen['patient_macro_f1']
                item.update(validation_patient_macro_f1=score, threshold=chosen_threshold,
                    validation_grid={str(t): v['patient_macro_f1'] for t,v in validation.items()})
                if score > best_score:
                    best_score, best_update, best_threshold = score, update, chosen_threshold
                    atomic_checkpoint(output_dir / 'best.pt', {'model': model.state_dict(), 'update': update,
                        'threshold': best_threshold, 'validation_score': best_score,
                        'protocol_sha256': file_sha256(study / 'protocol.json')})
                    atomic_json(output_dir / 'best_validation.json', chosen)
            history.append(item)
            atomic_json(output_dir / 'history.json', history)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(data), 'update': update, 'history': history,
                'best_score': best_score, 'best_update': best_update, 'best_threshold': best_threshold,
                'protocol_sha256': file_sha256(study / 'protocol.json')}
            atomic_checkpoint(resume_path, state)
            progress = {'run': tag, 'update': update, 'max_updates': max_updates,
                'training_loss': running, 'best_validation_f1': best_score if fold != 'refit' else None,
                'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            atomic_json(study / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    if fold == 'refit':
        best_update, best_threshold = max_updates, threshold
        atomic_checkpoint(output_dir / 'best.pt', {'model': model.state_dict(), 'update': max_updates,
            'threshold': threshold, 'protocol_sha256': file_sha256(study / 'protocol.json')})
    complete = {'run': tag, 'group': group['id'], 'fold': fold, 'seed': seed,
        'updates': max_updates, 'best_update': best_update, 'best_threshold': best_threshold,
        'best_validation_f1': best_score if fold != 'refit' else None,
        'checkpoint_sha256': file_sha256(output_dir / 'best.pt'),
        'training_candidates': len(train_rows), 'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    atomic_json(output_dir / 'complete.json', complete)
    del data, model, optimizer, scheduler, scaler, criterion
    torch.cuda.empty_cache()
    return complete


def run(args):
    study = Path(args.study).resolve()
    verify_frozen(study)
    configure()
    forbid_test_access(study)
    verify_caches(study)
    protocol = read_json(study / 'protocol.json')
    records = read_csv(study / 'candidate_manifest.csv')
    if (study / 'final_test_started.json').exists():
        raise ValueError('Final test has already started; no training rerun')
    if not (study / 'training_started.json').exists():
        atomic_json(study / 'training_started.json', {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'protocol_sha256': file_sha256(study / 'protocol.json')})
    results = []
    for group in protocol['groups']:
        for fold in protocol['planned_development_validation_folds']:
            for seed in protocol['seeds']:
                results.append(train_one(study, protocol, records, group, seed, fold,
                    protocol['development_updates'], microbatch=args.microbatch))
    atomic_json(study / 'development_summary.json', results)
    refits = []
    for group in protocol['groups']:
        subset = [r for r in results if r['group'] == group['id']]
        assert len(subset) == 9
        updates = int(max(40, min(2400, 40 * round(np.median([r['best_update'] for r in subset]) / 40))))
        threshold = float(np.median([r['best_threshold'] for r in subset]))
        for seed in protocol['seeds']:
            refits.append(train_one(study, protocol, records, group, seed, 'refit', updates,
                threshold=threshold, microbatch=args.microbatch))
    atomic_json(study / 'checkpoint_lock.json', {'checkpoints': refits,
        'protocol_sha256': file_sha256(study / 'protocol.json'), 'locked_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    print('All 45 development runs and 15 final refits completed and locked; final test is still sealed.', flush=True)


def final_test(args):
    study = Path(args.study).resolve()
    verify_frozen(study)
    verify_caches(study, include_test=True)
    configure()
    lock = read_json(study / 'checkpoint_lock.json')
    if len(lock['checkpoints']) != 15:
        raise ValueError('Incomplete lock')
    for item in lock['checkpoints']:
        if file_sha256(study / 'runs' / item['run'] / 'best.pt') != item['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
    marker = study / 'final_test_started.json'
    with marker.open('x', encoding='utf-8') as f:
        json.dump({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'checkpoint_lock_sha256': file_sha256(study / 'checkpoint_lock.json')}, f)
    protocol = read_json(study / 'protocol.json')
    rows = [r for r in read_csv(study / 'candidate_manifest.csv') if r['planned_partition'] == 'test']
    groups = {g['id']: g for g in protocol['groups']}
    results = []
    for item in lock['checkpoints']:
        model, optimizer, scheduler, scaler = model_optimizer(item['seed'], groups[item['group']], protocol)
        checkpoint = torch.load(study / 'runs' / item['run'] / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        score = evaluate(model, study, rows, [item['best_threshold']], distances=True, region_metrics=True)[item['best_threshold']]
        directory = study / 'runs' / item['run'] / 'test'
        directory.mkdir()
        atomic_json(directory / 'evaluation.json', score)
        write_csv(directory / 'per_image.csv', score['per_image'])
        results.append({'run': item['run'], 'group': item['group'], 'seed': item['seed'],
            'boundary_f1': score['patient_macro_f1'], 'per_patient': score['per_patient'],
            'region_dice': patient_macro(score['per_image'], 'dice'),
            'empty_predictions': sum(r['empty_prediction'] for r in score['per_image'])})
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        print(json.dumps(results[-1]), flush=True)
    summary = {}
    for group in groups:
        subset = [r for r in results if r['group'] == group]
        scores = [r['boundary_f1'] for r in subset]
        summary[group] = {'boundary_f1_mean': float(np.mean(scores)), 'boundary_f1_seed_sd': float(np.std(scores, ddof=1)), 'runs': subset}
    atomic_json(study / 'final_summary.json', {'groups': summary, 'test_patients': 3, 'candidate_test_frames': len(rows),
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})


def benchmark(args):
    study = Path(args.study).resolve()
    verify_frozen(study)
    configure()
    forbid_test_access(study)
    protocol = read_json(study / 'protocol.json')
    group = next(g for g in protocol['groups'] if g['id'] == 'D')
    rows = train_records(read_csv(study / 'candidate_manifest.csv'), group, 'fold_1')
    model, optimizer, scheduler, scaler = model_optimizer(910001, group, protocol)
    data = CachedTrainingData(study / 'data', rows, 910001)
    loss_fn = MixedBoundaryLoss().cuda()
    times = []
    for step in range(8):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        value = 0.
        for micro in range(4 // args.microbatch):
            image, region, boundary, has_region = data.batch(args.microbatch, 'cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                loss, components = loss_fn(model(image), region, boundary, has_region)
            value += float(loss.detach()) / (4 // args.microbatch)
            scaler.scale(loss / (4 // args.microbatch)).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    result = {'status': 'nonformal_development_data_pipeline_check', 'updates': 8,
        'effective_batch': 4, 'microbatch': args.microbatch, 'finite_last_loss': value,
        'median_seconds_per_update_after_warmup': float(np.median(times[2:])),
        'peak_vram_GB': torch.cuda.max_memory_allocated() / 1024**3,
        'test_access_forbidden': True, 'weights_discarded': True}
    atomic_json(study / f'pipeline_benchmark_micro{args.microbatch}.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'benchmark', 'run', 'test'])
    parser.add_argument('--study', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--identities-confirmed', action='store_true')
    parser.add_argument('--visible-boundaries-complete', action='store_true')
    parser.add_argument('--microbatch', type=int, default=1)
    args = parser.parse_args()
    {'prepare': prepare, 'benchmark': benchmark, 'run': run, 'test': final_test}[args.command](args)


if __name__ == '__main__':
    main()
