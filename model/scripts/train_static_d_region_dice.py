"""Measure the development Dice effect of whole-image foreground adaptation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import time

import cv2
import numpy as np
import torch

from static_d_baseline import (
    DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, exclusive_training,
    load_runner, read_json, verify, verify_file,
)
from train_static_d_region_refiner import FOLDS, dice_at_original_size, read_frames, save_checkpoint

DEFAULT_STUDY = LOGS / 'static_d_region_dice_20261124_01'
SIZE = (1024, 576)
UPDATES = 240
INTERVAL = 40
BATCH = 2


def coarse_checkpoint(development, fold, seed):
    folder = ORIGINAL / 'runs' / f'D_{fold}_s{seed}'
    receipt = read_json(folder / 'complete.json')
    config = read_json(folder / 'config.json')
    train = {r['sample_id'] for r in development if r['planned_validation_fold'] != fold}
    validation = {r['sample_id'] for r in development} - train
    if ((receipt['group'], receipt['fold'], receipt['seed']) != ('D', fold, seed) or
            set(config['training_ids']) != train or set(config['validation_ids']) != validation):
        raise ValueError(f'Unexpected D checkpoint split for {fold}')
    path = folder / 'best.pt'
    verify_file(path, receipt['checkpoint_sha256'])
    return path


def training_batch(rng, patients, frames, runner):
    images, masks = [], []
    w, h = SIZE
    for _ in range(BATCH):
        options = patients[int(rng.integers(len(patients)))]
        image, mask = frames[options[int(rng.integers(len(options)))]]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(.93, 1.07))
        matrix[:, 2] += rng.uniform(-.025, .025, 2) * (w, h)
        image = cv2.warpAffine(image, matrix, SIZE, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, matrix, SIZE, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if rng.random() < .5:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        image = ((image.astype(np.float32) - 127.5) * rng.uniform(.84, 1.16)
                 + 127.5 + rng.uniform(-20, 20)).clip(0, 255)
        images.append(runner.normalized_image(image))
        masks.append(torch.from_numpy(mask.astype(np.int64)))
    return torch.stack(images).cuda(), torch.stack(masks).cuda()


def get_region_probability(model, image, runner):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        out = model(runner.normalized_image(image)[None].cuda())['logits']
        return out.float().softmax(1)[0, 1].cpu().numpy()


def macro(rows, keys):
    patients = defaultdict(list)
    for row in rows:
        patients[row['patient']].append(row)
    return {key: mean(mean(r[key] for r in group) for group in patients.values()) for key in keys}


def train_fold(runner, study, rows, frames, resized, coarse, fold, seed, protocol_hash):
    from icg_pipeline.training import DiceCELoss

    folder = study / 'runs' / f'G_{fold}_s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    fit_rows = [r for r in rows if r['planned_validation_fold'] != fold]
    validation_rows = [r for r in rows if r['planned_validation_fold'] == fold]
    patients = defaultdict(list)
    for row in fit_rows:
        patients[row['patient_source_id']].append(row['sample_id'])
    if set(patients) & {r['patient_source_id'] for r in validation_rows}:
        raise ValueError('Patient overlap in foreground adaptation')
    checkpoint = folder / 'foreground.pt'
    if (folder / 'complete.json').is_file():
        receipt = read_json(folder / 'complete.json')
        verify_file(checkpoint, receipt['checkpoint_sha256'])
        return read_json(folder / 'per_image.json')

    rng = np.random.default_rng(seed)
    runner.seed_all(seed)
    net = runner.MixedBoundaryUNet(True, head_seed=0, encoder_weights=None).cuda()
    state = torch.load(coarse, map_location='cpu', weights_only=True)
    net.load_state_dict(state['model'])
    del state
    net.eval()
    predictions = {}
    for row in validation_rows:
        sid = row['sample_id']
        predictions[sid] = (get_region_probability(net, frames[sid].old_rgb, runner),
                            get_region_probability(net, resized[sid][0], runner))
    net.requires_grad_(True)
    net.encoder.requires_grad_(False)
    net.boundary_head.requires_grad_(False)
    net.train()
    for layer in net.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
    loss_function = DiceCELoss(dice_weight=.75, ce_weight=.25, include_background=False)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=4e-5, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, UPDATES, eta_min=4e-6)
    start = 0
    history = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['protocol_sha256'] != protocol_hash:
            raise ValueError('Foreground adaptation resume protocol mismatch')
        net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['scheduler'])
        start, history = saved['update'], saved['history']
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    started = time.perf_counter()
    losses = []
    for update in range(start + 1, UPDATES + 1):
        image, mask = training_batch(rng, list(patients.values()), resized, runner)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            logits = net(image)['logits']
        loss = loss_function(logits.float(), mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite foreground adaptation loss {fold}/{update}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        losses.append(float(loss.detach()))
        if update % INTERVAL == 0 or update == UPDATES:
            entry = {'update': update, 'training_loss': mean(losses),
                     'elapsed_seconds_this_session': time.perf_counter() - started}
            history.append(entry)
            losses = []
            save_checkpoint(checkpoint, {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': schedule.state_dict(), 'rng': rng.bit_generator.state,
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                            'update': update, 'history': history, 'protocol_sha256': protocol_hash})
            runner.atomic_json(folder / 'history.json', history)
            runner.atomic_json(study / 'progress.json', {'run': folder.name, 'update': update, 'max_updates': UPDATES,
                               'training_loss': entry['training_loss'], 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
            print(folder.name, update, round(entry['training_loss'], 5), flush=True)
    net.eval()
    results = []
    for row in validation_rows:
        sid = row['sample_id']
        base, large = predictions[sid]
        adapted = get_region_probability(net, resized[sid][0], runner)
        scores = {'old_d': base, 'larger_input': large, 'adapted': adapted, 'blend': (large + adapted) * .5}
        results.append({'patient': row['patient_source_id'], 'sample': sid, 'fold': fold,
                        **{key: dice_at_original_size(prob >= .5, frames[sid]) for key, prob in scores.items()}})
    runner.atomic_json(folder / 'per_image.json', results)
    runner.atomic_json(folder / 'complete.json', {
        'run': folder.name, 'group': 'G', 'fold': fold, 'seed': seed, 'updates': UPDATES,
        'best_update': UPDATES, 'training_region_patients': len(patients),
        'training_region_frames': len(fit_rows), 'validation_region_frames': len(validation_rows),
        'checkpoint_sha256': digest(checkpoint),
    })
    print(folder.name, macro(results, scores), flush=True)
    return results


def report(runner, study, rows, seed):
    report_dir = study / 'report'
    report_dir.mkdir(exist_ok=True)
    arms = {'old_d': '旧 D', 'larger_input': '旧权重直接用 1024×576',
            'adapted': '完整掩膜微调', 'blend': '大输入旧权重与微调输出等权融合'}
    all_scores = macro(rows, arms)
    patients = defaultdict(list)
    for row in rows:
        patients[row['patient']].append(row)
    if len(rows) != 66 or len(patients) != 5:
        raise ValueError('Foreground adaptation evaluation coverage changed')
    if seed == 42 and abs(all_scores['old_d']-.833334) > .0001:
        raise ValueError('Original seed 42 development Dice did not reproduce')
    by_patient = sorted([{'frames': len(entries), 'fold': entries[0]['fold'], **macro(entries, arms)}
                         for entries in patients.values()], key=lambda r: r['old_d'])
    runner.atomic_json(report_dir / 'metrics.json', {'overall': all_scores, 'patients': by_patient,
                     'seed': seed, 'validation_frames': len(rows), 'validation_patients': len(patients)})
    lines = ['# 旧 D 整幅前景微调', '',
             f'固定种子 {seed} 的患者三折。Dice 用 5 位开发患者的 66 张完整掩膜计算。',
             '各折从同一归档旧 D 权重开始，使用 1024×576、AdamW 和 240 步固定训练预算。',
             '冻结编码器、区域解码器微调。损失为 0.75 Dice 与 0.25 交叉熵，只使用折内训练患者的完整掩膜。',
             '', '| 方案 | 患者平均 Dice |', '| --- | ---: |']
    lines.extend(f'| {label} | {all_scores[key]:.2%} |' for key, label in arms.items())
    lines.extend(['', '| 验证折 | 完整掩膜 | 旧 D | 大输入旧 D | 全图微调 | 等权融合 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for row in by_patient:
        values = ' | '.join(f'{row[key]:.2%}' for key in arms)
        lines.append(f"| 第 {row['fold'][-1]} 折 | {row['frames']} | {values} |")
    output = report_dir / '整幅前景微调报告.md'
    temporary = output.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    args = parser.parse_args()
    study = args.study.resolve()
    if study.parent != LOGS.resolve() or study in (ORIGINAL.resolve(), BASELINE.resolve()):
        parser.error('Use a new study folder directly under model/logs')
    study.mkdir(exist_ok=True)
    with exclusive_training(study):
        development, _ = verify(BASELINE)
        runner = load_runner(ORIGINAL)
        runner.forbid_test_access(study)
        runner.verify_frozen(ORIGINAL)
        sources = {fold: coarse_checkpoint(development, fold, args.seed) for fold in FOLDS}
        protocol = {'development_only': True, 'groups': [{'id': 'G', 'display_name': '整幅前景微调'}],
                    'experiment_label': '旧 D 整幅前景微调 · 患者三折', 'seeds': [args.seed],
                    'planned_development_validation_folds': list(FOLDS), 'development_updates': UPDATES,
                    'validation_interval_updates': INTERVAL, 'checkpoint_selection': 'fixed_last_update',
                    'size': list(SIZE), 'batch': BATCH, 'trainer_sha256': digest(Path(__file__)),
                    'base_protocol_sha256': digest(ORIGINAL / 'protocol.json'),
                    'base_checkpoints': {fold: digest(path) for fold, path in sources.items()}}
        path = study / 'protocol.json'
        if path.exists() and read_json(path) != protocol:
            raise ValueError('Foreground adaptation protocol mismatch')
        runner.atomic_json(path, protocol)
        rows = [row for row in development if row['annotation_kind'] != 'linestrip']
        frames = read_frames(runner, rows)
        resized = {sid: (cv2.resize(fr.rgb, SIZE, interpolation=cv2.INTER_AREA),
                    (cv2.resize(fr.region, SIZE, interpolation=cv2.INTER_AREA) >= .5).astype(np.uint8))
                   for sid, fr in frames.items()}
        result = []
        protocol_hash = digest(path)
        for fold in FOLDS:
            result.extend(train_fold(runner, study, rows, frames, resized, sources[fold], fold, args.seed, protocol_hash))
        state = study / 'report_pipeline_state.json'
        runner.atomic_json(state, {'stage': 'comparing_development'})
        output = report(runner, study, result, args.seed)
        runner.atomic_json(state, {'stage': 'complete', 'final_report': str(output)})
        print(macro(result, ('old_d', 'larger_input', 'adapted', 'blend')), flush=True)


if __name__ == '__main__':
    main()
