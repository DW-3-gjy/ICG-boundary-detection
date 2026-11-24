"""Train and score a local region-mask refiner with archived D development folds."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from statistics import mean
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from static_d_baseline import (
    DEFAULT_STUDY as BASELINE, LOGS, MODEL, ORIGINAL, digest, exclusive_training,
    load_runner, read_json, verify, verify_file,
)

DEFAULT_STUDY = LOGS / 'static_d_region_refiner_20261124_01'
REFINER = MODEL / 'src' / 'icg_pipeline' / 'mask_refiner.py'
SEED = 42
SIZE = (1536, 864)
PATCH = 192
BATCH = 8
RADIUS = 32
UPDATES = 640
INTERVAL = 40
FOLDS = ('fold_1', 'fold_2', 'fold_3')


@dataclass
class Frame:
    rgb: np.ndarray
    region: np.ndarray
    packed: np.ndarray
    old_rgb: np.ndarray
    original_size: tuple[int, int]


@dataclass
class Proposal:
    logits: np.ndarray
    distance: np.ndarray
    edge_positions: np.ndarray


def read_frames(runner, rows):
    paths = {r['sample_id']: Path(r['image']) for r in read_json(ORIGINAL / 'source_paths.private.json')}
    frames = {}
    for row in rows:
        sid = row['sample_id']
        if row['planned_partition'] != 'development':
            raise ValueError('Unexpected partition in region training')
        path = paths[sid]
        verify_file(path, row['image_sha256'])
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        with np.load(BASELINE / 'data' / 'development' / f'{sid}.npz', allow_pickle=False) as data:
            w, h = (int(v) for v in data['original_size'])
            if bgr is None or bgr.shape[:2] != (h,w) or not bool(data['has_region']):
                raise ValueError(f'Invalid image or incomplete region for {sid}')
            rgb = cv2.cvtColor(cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            packed = data['packed_region']
            region = np.unpackbits(packed, count=w*h).reshape(h,w).astype(np.float32)
            reduced = cv2.resize(region, SIZE, interpolation=cv2.INTER_AREA).clip(0., 1.)
            frames[sid] = Frame(rgb, reduced, packed, data['image'], (w,h))
    print(f'Verified and prepared {len(frames)} complete development masks.', flush=True)
    return frames


def frozen_checkpoints(development):
    records = {}
    for fold in FOLDS:
        source = ORIGINAL / 'runs' / f'D_{fold}_s{SEED}'
        receipt = read_json(source / 'complete.json')
        config = read_json(source / 'config.json')
        if (receipt['group'],receipt['fold'],receipt['seed']) != ('D',fold,SEED):
            raise ValueError('Frozen D checkpoint scope changed')
        train = {row['sample_id'] for row in development if row['planned_validation_fold'] != fold}
        validation = {row['sample_id'] for row in development} - train
        if set(config['training_ids']) != train or set(config['validation_ids']) != validation:
            raise ValueError(f'Frozen D patient split changed for {fold}')
        records[fold] = {'run': source.name, 'sha256': verify_file(source/'best.pt', receipt['checkpoint_sha256'])}
    return records


def make_proposals(runner, frames, checkpoint, subset):
    net = runner.MixedBoundaryUNet(True, head_seed=0, encoder_weights=None).cuda().eval()
    path = ORIGINAL / 'runs' / checkpoint['run'] / 'best.pt'
    verify_file(path, checkpoint['sha256'])
    net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
    proposals = {}
    with torch.inference_mode():
        for row in subset:
            sid = row['sample_id']
            with torch.autocast('cuda', dtype=torch.float16):
                result = net(runner.normalized_image(frames[sid].old_rgb)[None].cuda())['logits'].float()[0]
            logit = (result[1] - result[0]).cpu().numpy()
            prior = cv2.resize(logit, SIZE, interpolation=cv2.INTER_NEAREST).clip(-6.,6.)
            foreground = (prior >= 0).astype(np.uint8)
            distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
            distance -= cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
            edge = np.flatnonzero(np.abs(distance) <= 1.5)
            proposals[sid] = Proposal(prior, distance, edge)
    return proposals


def sample_batch(rng, groups, frames, proposals):
    rgb, prior, distance, targets = [], [], [], []
    patients = list(groups)
    w,h = SIZE
    for _ in range(BATCH):
        patient = patients[int(rng.integers(len(patients)))]
        options = groups[patient]
        sid = options[int(rng.integers(len(options)))]
        frame, coarse = frames[sid], proposals[sid]
        y,x = divmod(int(rng.choice(coarse.edge_positions)), w)
        # Perturb overconfident training priors independently of the image.
        dy,dx = rng.integers(-20,21,2) if rng.random() < .75 else (0,0)
        y0,x0 = int(np.clip(y + dy - PATCH//2,0,h-PATCH)), int(np.clip(x + dx - PATCH//2,0,w-PATCH))
        yy,xx = int(np.clip(y0 - dy,0,h-PATCH)), int(np.clip(x0 - dx,0,w-PATCH))
        image_crop = np.s_[y0:y0+PATCH, x0:x0+PATCH]
        prior_crop = np.s_[yy:yy+PATCH, xx:xx+PATCH]
        image = frame.rgb[image_crop].astype(np.float32) / 127.5 - 1.
        image = (image * rng.uniform(.85,1.15) + rng.uniform(-.12,.12)).clip(-1.,1.)
        arrays = [image, coarse.logits[prior_crop], coarse.distance[prior_crop], frame.region[image_crop]]
        if rng.random() < .5:
            arrays = [a[:,::-1] for a in arrays]
        rgb.append(arrays[0].transpose(2,0,1))
        prior.append(arrays[1][None])
        distance.append(arrays[2][None])
        targets.append(arrays[3][None])
    return tuple(torch.from_numpy(np.stack(x)).cuda() for x in (rgb,prior,distance,targets))


def save_checkpoint(path, state):
    tmp = path.with_suffix('.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def fit(runner, model_class, study, fold, rows, frames, proposals, protocol_hash):
    name = f'R_{fold}_s{SEED}'
    folder = study / 'runs' / name
    folder.mkdir(parents=True,exist_ok=True)
    training = [(r['patient_source_id'],r['sample_id']) for r in rows if r['planned_validation_fold'] != fold]
    validation = [(r['patient_source_id'],r['sample_id']) for r in rows if r['planned_validation_fold'] == fold]
    if {p for p,_ in training} & {p for p,_ in validation}:
        raise ValueError('Patient overlap in region-refiner fold')
    groups = {}
    for patient,sid in training:
        if len(proposals[sid].edge_positions):
            groups.setdefault(patient,[]).append(sid)
    if set(groups) != {patient for patient,_ in training}:
        raise ValueError('A training patient has no usable coarse foreground edge')
    rng = np.random.default_rng(SEED)
    runner.seed_all(SEED)
    net = model_class(RADIUS).cuda()
    opt = torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(opt,UPDATES,eta_min=3e-5)
    ck = folder/'final.pt'
    history = []
    start = 0
    if ck.exists():
        old = torch.load(ck,map_location='cpu',weights_only=True)
        if old['protocol_sha256'] != protocol_hash:
            raise ValueError('Resume protocol mismatch')
        net.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        decay.load_state_dict(old['scheduler'])
        rng.bit_generator.state = old['rng']
        start,history = old['update'],old['history']
    start_time = time.perf_counter()
    losses = []
    for update in range(start+1,UPDATES+1):
        rgb, prior, dist, target = sample_batch(rng,groups,frames,proposals)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            logits = net(rgb,prior,dist)
        loss = net.loss(logits,target,dist)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss {name} / {update}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
        opt.step()
        decay.step()
        losses.append(float(loss.detach()))
        if update % INTERVAL == 0 or update == UPDATES:
            entry = {'update': update, 'training_loss': mean(losses),
                     'elapsed_seconds_this_session': time.perf_counter()-start_time}
            history.append(entry)
            losses = []
            save_checkpoint(ck,{'model': net.state_dict(), 'optimizer': opt.state_dict(),
                                'scheduler': decay.state_dict(), 'rng': rng.bit_generator.state,
                                'update': update, 'history': history, 'protocol_sha256': protocol_hash})
            runner.atomic_json(folder/'history.json',history)
            runner.atomic_json(study/'progress.json',{'run': name, 'update': update,
                'max_updates': UPDATES, 'training_loss': entry['training_loss'],
                'time': time.strftime('%Y-%m-%d %H:%M:%S')})
            print(name, update, round(entry['training_loss'],5), flush=True)
    report = score_fold(net.eval(),validation,frames,proposals,fold)
    count = sum(map(len,groups.values()))
    runner.atomic_json(folder/'per_image.json',report)
    runner.atomic_json(folder/'complete.json',{'run':name,'group':'R','seed':SEED,'fold':fold,
        'updates':UPDATES,'best_update':UPDATES,'training_region_frames':count,
        'validation_region_frames':len(validation),'training_region_patients':len(groups),
        'validation_region_patients': len({p for p,_ in validation}),
        'checkpoint_sha256': digest(ck)})
    print(f'{name}: {count} training masks, {len(validation)} validation masks.',flush=True)
    return report


def dice_at_original_size(mask, frame):
    w,h=frame.original_size
    reference=np.unpackbits(frame.packed,count=w*h).reshape(h,w).astype(bool)
    predicted=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    overlap=int(np.count_nonzero(predicted & reference))
    return 2.*overlap/max(1,int(predicted.sum()+reference.sum()))


def score_fold(net, rows, frames, proposals, fold):
    report=[]
    with torch.inference_mode():
        for patient,sid in rows:
            frame,prior=frames[sid],proposals[sid]
            rgb = torch.from_numpy(frame.rgb.transpose(2,0,1).copy()).cuda()[None].float()/127.5 - 1.
            base = torch.from_numpy(prior.logits).cuda()[None,None]
            dist = torch.from_numpy(prior.distance).cuda()[None,None]
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                corrected = net(rgb,base,dist)
            if not torch.isfinite(corrected).all():
                raise FloatingPointError(f'Nonfinite validation prediction {fold}')
            logits = corrected[0,0].cpu().numpy()
            outside=np.abs(prior.distance) >= RADIUS
            if not np.array_equal(logits[outside],prior.logits[outside]):
                raise ValueError('Refinement changed a pixel outside its band')
            scores={'old_d':dice_at_original_size(prior.logits >= 0,frame),
                    'refined':dice_at_original_size(logits >= 0,frame)}
            report.append({'patient':patient,'sample':sid,'fold':fold,**scores})
    return report


def write_report(runner,study,rows):
    patients={}
    for r in rows:
        patients.setdefault(r['patient'],[]).append(r)
    result = [{'fold':entries[0]['fold'],'frames':len(entries),
               **{key:mean(r[key] for r in entries) for key in ('old_d','refined')}}
              for entries in patients.values()]
    averaged = {key:mean(r[key] for r in result) for key in ('old_d','refined')}
    if abs(averaged['old_d']-.833334) > .0001 or len(rows) != 66 or len(patients) != 5:
        raise ValueError('Frozen old D baseline Dice or evaluation coverage failed to reproduce')
    report = study/'report'
    report.mkdir(exist_ok=True)
    runner.atomic_json(report/'metrics.json',{'overall':averaged, 'patients': result,
        'seed':SEED,'validation_frames':len(rows),'validation_patients':len(patients)})
    lines = ['# 旧 D 区域掩膜精修试跑','',
             '开发集患者三折，固定种子 42。区域 Dice 覆盖 5 位患者的 66 张完整掩膜。',
             '每个精修器只用该折训练患者的完整掩膜，以固定的 640 步训练预算拟合。',
             '两组共享旧 D 的归档检查点。区域精修器以 1536×864 图像工作，在原预测边缘约 80 个原图像素内修改掩膜。',
             '粗模型预测逐帧与人工掩膜在原图分辨率上评分。开发折成绩用于对照。','',
             '| 指标 | 旧 D | 旧 D＋区域精修 | 变化 |',
             '| --- | ---: | ---: | ---: |',
             f"| 患者等权 Dice | {averaged['old_d']:.2%} | {averaged['refined']:.2%} | {(averaged['refined']-averaged['old_d'])*100:+.2f} 个百分点 |",
             '', '| 验证折 | 完整掩膜 | 旧 D Dice | 精修 Dice |', '| --- | ---: | ---: | ---: |']
    for r in sorted(result,key=lambda item:item['old_d']):
        lines.append(f"| 第 {r['fold'][-1]} 折 | {r['frames']} | {r['old_d']:.2%} | {r['refined']:.2%} |")
    path=report/'区域精修训练报告.md'
    temp=path.with_suffix('.tmp')
    temp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    temp.replace(path)
    return path,averaged


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    args=parser.parse_args()
    study=args.study.resolve()
    if study.parent != LOGS.resolve() or study in (ORIGINAL.resolve(), BASELINE.resolve()):
        parser.error('Use a dedicated directory directly under model/logs')
    study.mkdir(exist_ok=True)
    with exclusive_training(study):
        development,_=verify(BASELINE)
        runner=load_runner(ORIGINAL)
        runner.forbid_test_access(study)
        runner.verify_frozen(ORIGINAL)
        spec=importlib.util.spec_from_file_location('icg_pipeline.mask_refiner',REFINER)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        closed=[r for r in development if r['annotation_kind'] != 'linestrip']
        checkpoints=frozen_checkpoints(development)
        protocol={'development_only': True, 'groups': [{'id':'R','display_name':'旧 D 区域掩膜精修'}],
                  'experiment_label':'旧 D 区域掩膜精修 · 患者三折', 'seeds':[SEED],
                  'planned_development_validation_folds':list(FOLDS), 'development_updates':UPDATES,
                  'validation_interval_updates':INTERVAL,'checkpoint_selection':'fixed_last_update',
                  'base':checkpoints,'refiner_sha256':digest(REFINER), 'trainer_sha256':digest(Path(__file__)),
                  'base_protocol_sha256':digest(ORIGINAL/'protocol.json'), 'refinement_size':list(SIZE),
                  'batch':BATCH,'patch':PATCH,'radius':RADIUS,'region_frames':len(closed)}
        pp=study/'protocol.json'
        if pp.exists() and protocol != read_json(pp):
            raise ValueError('The region-refiner study protocol has changed')
        runner.atomic_json(pp,protocol)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this run')
        frames=read_frames(runner,closed)
        all_rows=[]
        protocol_hash=digest(pp)
        for fold in FOLDS:
            proposal=make_proposals(runner,frames,checkpoints[fold],closed)
            all_rows.extend(fit(runner,module.RegionMaskRefiner,study,fold,closed,frames,proposal,protocol_hash))
        state=study/'report_pipeline_state.json'
        runner.atomic_json(state,{'stage':'comparing_development'})
        report,score=write_report(runner,study,all_rows)
        runner.atomic_json(state,{'stage':'complete','final_report':str(report)})
        print(score,flush=True)


if __name__=='__main__':
    main()
