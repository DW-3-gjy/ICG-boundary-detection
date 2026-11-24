"""Compare color-space region recovery and an alternate frozen region prior."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

from evaluate_static_d_region_ensemble import HIGH_SIZE, SEEDS, predict_frame
from static_d_baseline import DEFAULT_STUDY as BASELINE, LOGS, ORIGINAL, digest, load_runner, read_json, verify, verify_file
from train_static_d_region_dice import coarse_checkpoint
from train_static_d_region_refiner import FOLDS, dice_at_original_size, read_frames

OUTPUT = LOGS / 'static_d_auxiliary_dice_20261124_01' / 'diagnostics' / 'color_context.json'
NATIVE = (768,432)


def region_prior_checkpoint(development, fold, seed):
    folder = ORIGINAL / 'runs' / f'C_{fold}_s{seed}'
    config = read_json(folder / 'config.json')
    receipt = read_json(folder / 'complete.json')
    expected = {row['sample_id'] for row in development if row['planned_validation_fold'] != fold}
    if (receipt['group'],receipt['fold'],receipt['seed']) != ('C',fold,seed) or set(config['training_ids']) != expected:
        raise ValueError('C region prior leaked outside the frozen training patients')
    path = folder / 'best.pt'
    verify_file(path,receipt['checkpoint_sha256'])
    return path


def seed_mask(probability, foreground, background):
    mask = np.where(probability >= .50,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
    mask[probability >= foreground] = cv2.GC_FGD
    mask[probability <= background] = cv2.GC_BGD
    return mask


def color_cut(rgb, probability, size, color, *, foreground=.94, background=.025):
    image = cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    if color == 'lab':
        image = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    probability = cv2.resize(probability,size,interpolation=cv2.INTER_AREA)
    mask = seed_mask(probability, foreground, background)
    if not (mask == cv2.GC_FGD).any() or not (mask == cv2.GC_BGD).any():
        return probability >= .50
    cv2.setRNGSeed(42)
    cv2.grabCut(image,mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),
                2,cv2.GC_INIT_WITH_MASK)
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def average_prior(runner,development,rows,frames,large,fold,group):
    network = runner.MixedBoundaryUNet(group=='D',head_seed=0,encoder_weights=None).cuda().eval()
    output = dict.fromkeys((r['sample_id'] for r in rows), 0)
    getter = coarse_checkpoint if group == 'D' else region_prior_checkpoint
    for seed in SEEDS:
        source = getter(development,fold,seed)
        network.load_state_dict(torch.load(source,map_location='cpu',weights_only=True)['model'])
        for row in rows:
            sample = row['sample_id']
            _,_,map_ = predict_frame(network,frames[sample],large[sample],runner)
            output[sample] += map_ / len(SEEDS)
    return output


def main():
    development,_ = verify(BASELINE)
    runner = load_runner(ORIGINAL)
    runner.verify_frozen(ORIGINAL)
    target = [r for r in development if r['annotation_kind']!='linestrip']
    frames = read_frames(runner,target)
    large = {sid:cv2.resize(fr.rgb,HIGH_SIZE,interpolation=cv2.INTER_AREA) for sid,fr in frames.items()}
    values = defaultdict(lambda:defaultdict(list))
    for fold in FOLDS:
        rows=[r for r in target if r['planned_validation_fold']==fold]
        prior_d = average_prior(runner,development,rows,frames,large,fold,'D')
        prior_c = average_prior(runner,development,rows,frames,large,fold,'C')
        for r in rows:
            sid,patient = r['sample_id'],r['patient_source_id']
            fr = frames[sid]
            d = prior_d[sid]
            combined = d * .75 + prior_c[sid] * .25
            masks = {'d_ensemble':d>=.5,
                     'd_rgb_cut':color_cut(fr.rgb,d,NATIVE,'rgb'),
                     'd_lab_cut':color_cut(fr.rgb,d,NATIVE,'lab'),
                     'd_lab_high_resolution':color_cut(fr.rgb,d,HIGH_SIZE,'lab'),
                     'd_lab_weaker_background':color_cut(fr.rgb,d,NATIVE,'lab',background=.01),
                     'c_d_rgb_cut':color_cut(fr.rgb,combined,NATIVE,'rgb')}
            for key,mask in masks.items():
                values[key][patient].append(dice_at_original_size(mask,fr))
        print(f'{fold}: regional recovery scored on {len(rows)} images',flush=True)
    scores = {key:mean(mean(v) for v in by_patient.values()) for key,by_patient in values.items()}
    if abs(scores['d_ensemble'] - .8566076624633716) > .0001:
        raise ValueError('Frozen D region ensemble did not reproduce')
    order = sorted(values['d_ensemble'],key=lambda p:mean(values['d_ensemble'][p]))
    patients = [{key:mean(v[p]) for key,v in values.items()} for p in order]
    runner.atomic_json(OUTPUT,{'scope':'development-only exploratory scoring',
                     'frames':len(target),'patients':len(patients),'overall':scores,
                     'evaluator_sha256':digest(Path(__file__)),'paired_patient_dice':patients})
    print(scores,flush=True)


if __name__=='__main__':
    main()
