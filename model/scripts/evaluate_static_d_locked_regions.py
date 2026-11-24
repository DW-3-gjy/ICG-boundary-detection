"""Freeze and score one region candidate on a complete-mask evaluation cohort."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

import evaluate_static_d_color_context as color_context
import evaluate_static_d_region_ensemble as region_ensemble
from evaluate_static_d_color_context import NATIVE, color_cut
from evaluate_static_d_region_ensemble import HIGH_SIZE, SEEDS, predict_frame
from static_d_baseline import LOGS, MODEL, ORIGINAL, digest, exclusive_training, load_runner, read_json, verify_file
from train_static_d_region_refiner import Frame, SIZE, dice_at_original_size

METHODS = {
    'd_rgb_cut': ('rgb', NATIVE, .025, False),
    'd_lab_cut': ('lab', NATIVE, .025, False),
    'd_lab_high_resolution': ('lab', HIGH_SIZE, .025, False),
    'd_lab_weaker_background': ('lab', NATIVE, .01, False),
    'c_d_rgb_cut': ('rgb', NATIVE, .025, True),
}


def read_cohort(runner, manifest):
    if manifest is None:
        source = {r['sample_id']:r for r in read_json(ORIGINAL/'source_paths.private.json')}
        return [{'sample':r['sample_id'], 'patient':r['patient_source_id'],
                 'image':source[r['sample_id']]['image'], 'reference':source[r['sample_id']]['annotation_or_mask'],
                 'image_sha256':r['image_sha256'],'reference_sha256':r['annotation_or_mask_sha256'],
                 'kind':r['annotation_kind']}
                for r in runner.read_csv(ORIGINAL/'candidate_manifest.csv')
                if r['planned_partition']=='test' and r['annotation_kind']!='linestrip']
    with manifest.open(encoding='utf-8-sig',newline='') as stream:
        entries = list(csv.DictReader(stream))
    rows = []
    for row in entries:
        image = (manifest.parent/row['image']).resolve()
        label = (manifest.parent/row['reference']).resolve()
        rows.append({'sample':row['sample_id'],'patient':row['patient_id'],'kind':row['reference_kind'],
                     'image':str(image),'reference':str(label),
                     'image_sha256':digest(image),'reference_sha256':digest(label)})
    return rows


def prepare_frames(runner, rows, historical):
    if not rows or len({r['sample'] for r in rows}) != len(rows) or any(not r['patient'] for r in rows):
        raise ValueError('Manifest needs unique sample IDs and a patient ID for every complete mask')
    hashes = {r['image_sha256'] for r in rows}
    if len(hashes) != len(rows):
        raise ValueError('The evaluation cohort contains duplicate files')
    if not historical:
        known = runner.read_csv(ORIGINAL/'candidate_manifest.csv')
        known_hashes = {row['image_sha256'] for row in known}
        known_patients = {row['patient_source_id'] for row in known}
        auxiliary = LOGS/'static_d_auxiliary_dice_20261124_01'/'auxiliary_sources.private.json'
        if auxiliary.is_file():
            known_hashes.update(row['image_sha256'] for row in read_json(auxiliary))
        if hashes & known_hashes or known_patients & {row['patient'] for row in rows}:
            raise ValueError('Previously used images or patient IDs are present in the independent cohort')
    frames = {}
    for row in rows:
        image, reference = Path(row['image']), Path(row['reference'])
        verify_file(image, row['image_sha256'])
        verify_file(reference, row['reference_sha256'])
        bgr = cv2.imdecode(np.fromfile(image,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Cannot read an evaluation image')
        h, w = bgr.shape[:2]
        if row['kind'] in ('polygon','labelme_polygon'):
            label = read_json(reference)
            if (int(label.get('imageWidth',w)),int(label.get('imageHeight',h))) != (w,h):
                raise ValueError('LabelMe image dimensions do not match')
            region, _, whole = runner.rasterize_shapes(label,w,h)
            if not whole:
                raise ValueError('Open lines do not define a complete evaluation mask')
        elif row['kind'] in ('region_mask','polygon_derived_region_mask'):
            region = cv2.imdecode(np.fromfile(reference,np.uint8),cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError('Evaluation requires a closed polygon or an explicit binary foreground mask')
        if region is None or region.shape != (h,w) or not region.any():
            raise ValueError('Evaluation region is empty or has incorrect dimensions')
        if not set(np.unique(region)) <= {0,1,255}:
            raise ValueError('Evaluation mask is not binary')
        packed = np.packbits((region>0).reshape(-1))
        small = cv2.cvtColor(cv2.resize(bgr,NATIVE,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        large = cv2.cvtColor(cv2.resize(bgr,SIZE,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        frames[row['sample']] = Frame(large,None,packed,small,(w,h))
    return frames


def checkpoints(groups):
    frozen = read_json(ORIGINAL/'checkpoint_lock.json')
    if frozen['protocol_sha256'] != digest(ORIGINAL/'protocol.json'):
        raise ValueError('Frozen final checkpoint protocol mismatch')
    weights = []
    for row in frozen['checkpoints']:
        if row['group'] in groups and row['seed'] in SEEDS:
            path = ORIGINAL/'runs'/row['run']/'best.pt'
            if row['fold']!='refit' or row['training_candidates']!=106:
                raise ValueError('The evaluator requires full-development frozen checkpoints')
            weights.append({'group':row['group'],'seed':row['seed'],'run':row['run'],
                            'sha256':verify_file(path,row['checkpoint_sha256'])})
    if len(weights)!=len(groups)*len(SEEDS):
        raise ValueError('The frozen checkpoint ensemble is incomplete')
    return weights


def score(runner, frames, rows, weights, method):
    color,size,background,combine_c = METHODS[method]
    data = {sid:dict.fromkeys(('D','C'),0) for sid in frames}
    seeds = defaultdict(list)
    for group in ('D','C') if combine_c else ('D',):
        model = runner.MixedBoundaryUNet(group=='D',head_seed=0,encoder_weights=None).cuda().eval()
        for entry in (r for r in weights if r['group']==group):
            path = ORIGINAL/'runs'/entry['run']/'best.pt'
            verify_file(path,entry['sha256'])
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state['protocol_sha256'] != digest(ORIGINAL/'protocol.json'):
                raise ValueError('The loaded weights refer to a different protocol')
            model.load_state_dict(state['model'])
            for row in rows:
                sid = row['sample']
                frame=frames[sid]
                large=cv2.resize(frame.rgb,HIGH_SIZE,interpolation=cv2.INTER_AREA)
                native, _, probability=predict_frame(model,frame,large,runner)
                data[sid][group] += probability/len(SEEDS)
                if group=='D':
                    seeds[(entry['seed'],row['patient'])].append(dice_at_original_size(native>=.5,frame))
    paired=defaultdict(list)
    for row in rows:
        sid, patient=row['sample'],row['patient']
        frame=frames[sid]
        d=data[sid]['D']
        prior=.75*d+.25*data[sid]['C'] if combine_c else d
        fixed=color_cut(frame.rgb,prior,size,color,background=background)
        paired[patient].append({'ensemble':dice_at_original_size(d>=.5,frame),
                               'candidate':dice_at_original_size(fixed,frame)})
    patients = [{'frames':len(result), **{metric:mean(r[metric] for r in result)
                                        for metric in ('ensemble','candidate')}}
                for result in paired.values()]
    return {'old_d':mean(mean(v) for v in seeds.values()),
            'ensemble':mean(r['ensemble'] for r in patients),
            'candidate':mean(r['candidate'] for r in patients),'patients':patients,'frames':len(rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--historical',action='store_true')
    source.add_argument('--manifest',type=Path,help='CSV with sample_id, patient_id, image, reference, reference_kind')
    parser.add_argument('--identity-confirmed',action='store_true')
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    study=args.study.resolve()
    if study.parent!=LOGS.resolve() or study==ORIGINAL.resolve():
        parser.error('Use a dedicated directory under model/logs')
    if args.manifest and not args.identity_confirmed:
        parser.error('Confirm that the listed patients are independent of development and previously reviewed cohorts')
    study.mkdir(exist_ok=True)
    with exclusive_training(study):
        runner=load_runner(ORIGINAL)
        runner.verify_frozen(ORIGINAL)
        rows=read_cohort(runner,args.manifest)
        frames=prepare_frames(runner,rows,args.historical)
        weights=checkpoints(('C','D') if METHODS[args.method][-1] else ('D',))
        path=study/'locked_regions.private.json'
        proposal={'scope':'historical' if args.historical else 'independent','method':args.method,
                  'data':rows,'weights':weights,'source_sha256':digest(Path(__file__)),
                  'candidate_source_sha256': {
                      'prior':digest(Path(region_ensemble.__file__)),
                      'region_correction':digest(Path(color_context.__file__)),
                  }}
        if path.exists() and read_json(path)!=proposal:
            raise ValueError('Cohort, method or checkpoint changed after the evaluation freeze')
        if not path.exists():
            if not args.historical and not args.freeze_only:
                raise ValueError('Freeze the independent cohort and candidate before scoring')
            runner.atomic_json(path,proposal)
        if args.freeze_only:
            print(f'Frozen {len(rows)} full masks, {len({r["patient"] for r in rows})} patients.',flush=True)
            return
        result_path=study/'results.json'
        if result_path.exists():
            previous=read_json(result_path)
            if previous['locked_manifest_sha256']!=digest(path):
                raise ValueError('Stored results are for a different evaluation freeze')
            print(previous,flush=True)
            return
        result=score(runner,frames,rows,weights,args.method)
        if args.historical and abs(result['old_d']-.8409612167275982)>.0002:
            raise ValueError('The previously scored original D region baseline did not reproduce')
        result.update(scope=proposal['scope'],method=args.method,locked_manifest_sha256=digest(path))
        runner.atomic_json(result_path,result)
        print(result,flush=True)


if __name__=='__main__':
    main()
