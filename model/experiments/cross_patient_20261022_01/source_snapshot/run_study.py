"""Frozen 24-run study; patient2 access is denied until development is complete."""
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(protocol):
    records=protocol['records']
    if not records or len({r['filename'] for r in records})!=len(records):
        raise ValueError('Missing or duplicate sample records')
    patients={s:{r['patient'] for r in records if r['split']==s} for s in ('train','val','test')}
    if patients['train']!={'patient_001'} or patients['val']!={'patient_001'} or patients['test']!={'patient_002'}:
        raise ValueError('Patient-level split violation')
    groups={s:{r['source_group'] for r in records if r['split']==s} for s in ('train','val')}
    if groups['train']&groups['val']:
        raise ValueError('Development source group crosses train/validation')
    hashes={}
    for r in records:
        if r['decoded_image_sha256'] in hashes:
            raise ValueError('Duplicate image pixels in study')
        hashes[r['decoded_image_sha256']]=r['split']
        if r['split']!='test' and r['phase']!='Z':
            raise ValueError('Non-Z development sample')
    tests={r['filename'] for r in records if r['split']=='test'}
    subgroups=protocol['test_subgroups']
    if set(subgroups['all_69_auxiliary'])!=tests:
        raise ValueError('Test subgroup coverage mismatch')
    a,b=set(subgroups['Z_original_label']),set(subgroups['Z_instrument_occluded'])
    if a&b or a|b!=set(subgroups['primary_Z_all']):
        raise ValueError('Z strata overlap or do not cover the primary endpoint')
    identities=[r['run_id'] for r in protocol['runs']]
    if len(set(identities))!=len(identities):
        raise ValueError('Duplicate run identity')


def test_access_guard(test_roots, state):
    prefixes=[os.path.normcase(os.path.abspath(os.fspath(p))) for p in test_roots]
    def guard(event,args):
        if state['test_allowed'] or event not in {'open','os.listdir','os.scandir'} or not args:
            return
        value=args[0]
        if not isinstance(value,(str,bytes,os.PathLike)):
            return
        candidate=os.path.normcase(os.path.abspath(os.fsdecode(value)))
        if any(candidate==p or candidate.startswith(p+os.sep) for p in prefixes):
            state['denied_test_access_attempts']+=1
            raise PermissionError('Patient2 image access denied during development stage')
    return guard


def verify_files(root,fingerprints,include_test):
    for relative,expected in fingerprints.items():
        if ('/test/' in relative) != include_test:
            continue
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=expected:
            raise ValueError('Frozen data changed: '+relative)


def tensor_state_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def csv_write(path,rows):
    if not rows:
        return
    columns=sorted({k for r in rows for k in r})
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def run(study):
    study=study.resolve()
    protocol_path=study/'protocol.json'
    protocol_sha=(study/'protocol.sha256').read_text().strip()
    if sha(protocol_path)!=protocol_sha:
        raise ValueError('Frozen protocol changed')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    validate_protocol(protocol)
    snapshot=study/'code_snapshot'
    for relative,expected in protocol['code_fingerprints'].items():
        if sha(snapshot/relative)!=expected:
            raise ValueError('Frozen source code changed')
    sys.path.insert(0,str(snapshot/'model/src'))
    from icg_pipeline.training import Trainer
    from icg_pipeline.data import ICGDataset,read_image,write_image
    from icg_pipeline.inference import ICGPredictor
    from icg_pipeline.eval import evaluate_segmentation,aggregate_metrics
    from icg_pipeline.utils import write_json

    data=Path(protocol['data_root'])
    state={'test_allowed':False,'denied_test_access_attempts':0}
    sys.addaudithook(test_access_guard([data/'fullres/test'],state))
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError('This frozen study requires the recorded CUDA device')
    environment={'python':sys.version,'torch':str(torch.__version__),'device':torch.cuda.get_device_name(0),
                 'cuda_version':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                 'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                 'deterministic_algorithms':True,'deterministic_warn_only':True,'TF32':False,
                 'protocol_sha256':protocol_sha}
    write_json(study/'environment.json',environment)
    messages=[]
    original_showwarning=warnings.showwarning
    def record_warning(message,category,filename,lineno,file=None,line=None):
        text=f'{category.__name__}: {message}'
        if text not in messages:
            messages.append(text);write_json(study/'runtime_warnings.json',messages)
        original_showwarning(message,category,filename,lineno,file,line)
    warnings.showwarning=record_warning
    started=time.perf_counter()

    def code_and_protocol_check():
        if sha(protocol_path)!=protocol_sha:
            raise ValueError('Frozen protocol changed during experiment')
        for relative,expected in protocol['code_fingerprints'].items():
            if sha(snapshot/relative)!=expected:
                raise ValueError('Frozen code changed during experiment')

    def evaluate(config,split,output):
        output.mkdir(parents=True,exist_ok=True)
        predictor=ICGPredictor(config)
        pairs=ICGDataset(config['data'][f'{split}_data_dir'],is_train=False).pairs
        rows=[]
        for image_path,mask_path in pairs:
            image,target=read_image(image_path),read_image(mask_path,True)
            if image.shape[:2]!=target.shape:
                raise ValueError('Evaluation image/mask dimension mismatch')
            torch.cuda.synchronize();begin=time.perf_counter()
            prediction=predictor.predict_image(image)
            torch.cuda.synchronize();latency=(time.perf_counter()-begin)*1000
            metrics=evaluate_segmentation(prediction['mask'],target,config['evaluation']['metrics'])
            metric_keys=list(metrics)
            row=metrics|{'filename':image_path.name,'image_size_hw':list(target.shape),
                         'empty_prediction':not bool(prediction['mask'].any()),
                         'empty_reference':not bool(target.any()),'inference_ms':latency,
                         'nonfinite_metrics':[k for k,v in metrics.items() if not math.isfinite(v)]}
            rows.append(row)
            write_image(output/'masks'/image_path.name,prediction['mask'])
            overlay=image.copy()
            for mask,color in ((target,(255,255,0)),(prediction['mask'],(0,140,255))):
                contours,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,contours,-1,color,6)
            dest=output/'overlays'/(image_path.stem+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True)
            ok,encoded=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,91])
            if not ok:
                raise RuntimeError('Overlay encoding failed')
            encoded.tofile(dest)
        result={'per_sample':rows,'metric_keys':metric_keys,'aggregated':aggregate_metrics([
            {k:r[k] for k in metric_keys} for r in rows]),'sample_count':len(rows),
            'split':split,'distance_units':'original_image_pixels',
            'scope':'patient1_internal_validation' if split=='val' else 'one_independent_patient_retrospective_test'}
        if split=='test':
            result['subgroups']={}
            for name,filenames in protocol['test_subgroups'].items():
                selected=[r for r in rows if r['filename'] in filenames]
                if len(selected)!=len(filenames):
                    raise ValueError('Subgroup sample lost during evaluation')
                result['subgroups'][name]={'n':len(selected),'empty_predictions':sum(r['empty_prediction'] for r in selected),
                    'aggregated':aggregate_metrics([{k:r[k] for k in metric_keys} for r in selected])}
        write_json(output/'evaluation.json',result);csv_write(output/'per_image.csv',rows)
        del predictor
        gc.collect();torch.cuda.empty_cache()
        return result

    completed=[]
    if (study/'validation_summary.json').exists():
        previous=json.loads((study/'validation_summary.json').read_text(encoding='utf-8'))
        if previous['protocol_sha256']!=protocol_sha:
            raise ValueError('Existing development records belong to another protocol')
        completed=previous['runs']
    for number,definition in enumerate(protocol['runs'],1):
        run_id=definition['run_id'];run_dir=study/'runs'/run_id
        existing=next((r for r in completed if r['run_id']==run_id),None)
        if existing:
            if sha(run_dir/'checkpoints/best_model.pt')!=existing['checkpoint_sha256']:
                raise ValueError('Completed run checkpoint changed')
            continue
        if run_dir.exists():
            raise FileExistsError('Incomplete prior run preserved; audit it before continuing: '+run_id)
        code_and_protocol_check();verify_files(data,protocol['file_fingerprints'],False)
        config=copy.deepcopy(definition['config'])
        run_dir.mkdir(parents=True)
        write_json(run_dir/'config.json',config)
        write_json(study/'progress.json',{'stage':'training','run':run_id,'run_number':number,
            'total_runs':len(protocol['runs']),'completed_runs':len(completed),'protocol_sha256':protocol_sha})
        begin=time.perf_counter()
        trainer=Trainer(config)
        initialized={'seed':definition['seed'],'initial_epoch':trainer.current_epoch,
            'optimizer_initial_entries':len(trainer.optimizer.state),'loaded_prior_ICG_checkpoint':False,
            'encoder_weights':config['model'].get('encoder_weights'),
            'model_state_sha256':tensor_state_hash(trainer.model.state_dict()),
            'nonencoder_state_sha256':tensor_state_hash({k:v for k,v in trainer.model.state_dict().items() if not k.startswith('encoder.')}),
            'parameters':trainer.model.count_parameters(),'start_time':datetime.now(timezone(timedelta(hours=8))).isoformat()}
        if hasattr(trainer.model,'encoder'):
            initialized['encoder_state_sha256']=tensor_state_hash(trainer.model.encoder.state_dict())
            initialized['encoder_bn_frozen']=config['model']['freeze_encoder_batchnorm']
        if initialized['initial_epoch']!=0 or initialized['optimizer_initial_entries']!=0:
            raise ValueError('Training did not start fresh')
        write_json(run_dir/'initialization.json',initialized)
        trainer.train()
        for epoch in trainer.history:
            for part in ('train','val'):
                if not all(math.isfinite(v) for v in epoch[part].values()):
                    raise ValueError('Nonfinite development metric; retain failed run')
        if len(trainer.history)!=120 or trainer.optimizer_updates<=0:
            raise ValueError('Frozen training budget was not completed')
        chosen=torch.load(config['inference']['model_path'],map_location='cpu',weights_only=True)
        record={k:definition[k] for k in ('run_id','condition_id','family','seed')}
        record.update(best_epoch=chosen['epoch']+1,best_validation_dice=chosen['metrics']['dice'],
                      optimizer_updates=trainer.optimizer_updates,expected_updates=120*4,
                      training_seconds=time.perf_counter()-begin,
                      checkpoint_sha256=sha(config['inference']['model_path']))
        write_json(run_dir/'provenance.json',trainer.metadata|record|{'protocol_sha256':protocol_sha})
        del trainer,chosen;gc.collect();torch.cuda.empty_cache()
        verify_files(data,protocol['file_fingerprints'],False)
        validation_config=copy.deepcopy(config)
        validation_config['data']['val_data_dir']=str(data/'fullres/val')
        val=evaluate(validation_config,'val',run_dir/'val')
        record['fullres_validation']=val['aggregated']
        completed.append(record)
        write_json(study/'validation_summary.json',{'runs':completed,'complete':len(completed)==len(protocol['runs']),
                    'protocol_sha256':protocol_sha,'denied_test_access_attempts':state['denied_test_access_attempts']})
        print(f'RUN_COMPLETE {len(completed)}/{len(protocol["runs"])} {run_id} epoch={record["best_epoch"]} valDice={record["best_validation_dice"]:.4f}',flush=True)
    if len(completed)!=len(protocol['runs']):
        raise ValueError('Not all development runs completed')
    code_and_protocol_check();verify_files(data,protocol['file_fingerprints'],False)
    checkpoint_lock={r['run_id']:r['checkpoint_sha256'] for r in completed}
    write_json(study/'checkpoint_lock.json',{'protocol_sha256':protocol_sha,'checkpoints':checkpoint_lock,
        'selected_on':'patient1 resized validation Dice','test_data_used_for_selection':False})
    with (study/'test_evaluation_started.json').open('x',encoding='utf-8') as stream:
        json.dump({'protocol_sha256':protocol_sha,'checkpoint_lock_sha256':sha(study/'checkpoint_lock.json'),
                   'test_patient':'patient_002','started_at':datetime.now(timezone(timedelta(hours=8))).isoformat()},stream)
    state['test_allowed']=True
    verify_files(data,protocol['file_fingerprints'],True)
    test_runs=[]
    for number,definition in enumerate(protocol['runs'],1):
        run_id=definition['run_id'];run_dir=study/'runs'/run_id
        if sha(run_dir/'checkpoints/best_model.pt')!=checkpoint_lock[run_id]:
            raise ValueError('Checkpoint changed after test lock')
        write_json(study/'progress.json',{'stage':'test','run':run_id,'run_number':number,
            'total_runs':len(protocol['runs']),'completed_test_runs':len(test_runs),'protocol_sha256':protocol_sha})
        result=evaluate(definition['config'],'test',run_dir/'test')
        verify_files(data,protocol['file_fingerprints'],True)
        record={k:definition[k] for k in ('run_id','condition_id','family','seed')}
        record.update(aggregated=result['aggregated'],subgroups=result['subgroups'])
        test_runs.append(record)
        write_json(study/'test_summary.json',{'runs':test_runs,'complete':len(test_runs)==len(protocol['runs']),
                   'protocol_sha256':protocol_sha,'test_patient_count':1})
        print(f'TEST_COMPLETE {len(test_runs)}/{len(protocol["runs"])} {run_id}',flush=True)
    code_and_protocol_check();verify_files(data,protocol['file_fingerprints'],False)
    write_json(study/'progress.json',{'stage':'complete','training_runs':len(completed),'test_runs':len(test_runs),
               'elapsed_seconds':time.perf_counter()-started,'protocol_sha256':protocol_sha,
               'denied_test_access_attempts':state['denied_test_access_attempts']})
    print('STUDY_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study-root',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.study_root)
    except Exception as error:
        args.study_root.mkdir(parents=True,exist_ok=True)
        (args.study_root/'execution_error.json').write_text(json.dumps({'type':type(error).__name__,
            'message':str(error),'time':datetime.now(timezone(timedelta(hours=8))).isoformat()},indent=2),encoding='utf-8')
        raise
