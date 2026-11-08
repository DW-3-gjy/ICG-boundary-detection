"""Independently aggregate locked results and replay fixed-seed case figures."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mean_sd(values):
    values = [float(v) for v in values if v is not None]
    return {'mean': float(np.mean(values)) if values else None,
            'seed_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def macro(rows, metric):
    patients = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value not in (None, '', 'None'):
            patients[row['patient']].append(float(value))
    means = {k: float(np.mean(v)) for k,v in patients.items()}
    return float(np.mean(list(means.values()))) if means else None, means


def aggregate(study):
    summary = load(study / 'final_summary.json')
    lock = load(study / 'checkpoint_lock.json')
    assert len(lock['checkpoints']) == 15
    protocol=load(study/'protocol.json')
    initialization_hashes=defaultdict(set)
    complete_files=list((study/'runs').glob('*/complete.json'))
    assert len(complete_files)==60
    for complete_path in complete_files:
        directory=complete_path.parent
        complete=load(complete_path);initial=load(directory/'initialization.json')
        config=load(directory/'config.json')
        assert initial['warm_start_icg_checkpoint'] is False
        assert initial['encoder_weights_sha256']==protocol['official_encoder_weights_sha256']
        initialization_hashes[complete['seed']].add(initial['shared_parameter_sha256'])
        assert config['microbatch']==4 and config['effective_batch']==4
        if complete['fold']!='refit':
            history=load(directory/'history.json')
            assert len(history)==60 and history[-1]['update']==2400
            assert [r['update'] for r in history]==list(range(40,2401,40))
            assert all(np.isfinite(r['training_loss']) for r in history)
            best=max(r['validation_patient_macro_f1'] for r in history)
            first=next(r for r in history if r['validation_patient_macro_f1']==best)
            assert complete['best_update']==first['update']
            assert complete['best_threshold']==first['threshold']
    assert len(initialization_hashes)==3 and all(len(v)==1 for v in initialization_hashes.values())
    runs = []
    for item in lock['checkpoints']:
        directory = study / 'runs' / item['run']
        assert digest(directory / 'best.pt') == item['checkpoint_sha256']
        rows = table(directory / 'test/per_image.csv')
        assert len(rows) == 23 and len({r['patient'] for r in rows}) == 3
        f1, patients = macro(rows, 'f1')
        dice, region_patients = macro(rows, 'dice')
        assert len(region_patients) == 2
        runs.append({'run': item['run'], 'group': item['group'], 'seed': item['seed'],
            'boundary_f1': f1, 'per_patient': patients,
            'dice': dice, 'region_patients': region_patients,
            'f1_10px': macro(rows, 'f1_10px')[0], 'f1_40px': macro(rows, 'f1_40px')[0],
            'empty_predictions': sum(r['empty_prediction'] == 'True' for r in rows),
            'region_reference_frames': sum(r['has_region_reference'] == 'True' for r in rows),
            'undefined_hd95': sum(r['hd95'] in ('', 'None') for r in rows),
            'strata': {
                'Z_label': macro([r for r in rows if r['stratum']=='Z_label'],'f1')[0],
                'L_visible_fluorescence': macro([r for r in rows if r['stratum']=='L_visible_fluorescence'],'f1')[0],
                'linestrip': macro([r for r in rows if r['reference_kind']=='linestrip'],'f1')[0],
                'region_reference': macro([r for r in rows if r['has_region_reference']=='True'],'f1')[0],
            }})
    groups = {}
    for group in 'ABCDE':
        items = [r for r in runs if r['group'] == group]
        assert len(items) == 3 and {r['seed'] for r in items} == {42,43,44}
        metrics = {m: mean_sd([r[m] for r in items]) for m in ['boundary_f1','dice','f1_10px','f1_40px']}
        assert abs(metrics['boundary_f1']['mean'] - summary['groups'][group]['boundary_f1_mean']) < 1e-12
        assert abs(metrics['boundary_f1']['seed_sd'] - summary['groups'][group]['boundary_f1_seed_sd']) < 1e-12
        groups[group] = {'metrics': metrics, 'runs': items,
            'strata':{k:mean_sd([r['strata'][k] for r in items]) for k in items[0]['strata']},
            'per_patient_seed_mean': {p:float(np.mean([r['per_patient'][p] for r in items])) for p in items[0]['per_patient']}}
    contrasts = []
    for left,right,name in [('B','A','边界模块：仅Z'),('D','C','边界模块：Z及L'),
                            ('C','A','加入L：区域模型'),('D','B','加入L：边界模型'),('D','E','加入旧训练样本')]:
        patient_delta = {p:groups[left]['per_patient_seed_mean'][p]-groups[right]['per_patient_seed_mean'][p]
                         for p in groups[left]['per_patient_seed_mean']}
        contrasts.append({'name':name,'contrast':left+'-'+right,
            'mean_delta':groups[left]['metrics']['boundary_f1']['mean']-groups[right]['metrics']['boundary_f1']['mean'],
            'per_patient_delta':patient_delta, 'patients_improved':sum(v>0 for v in patient_delta.values())})
    return {'groups':groups,'contrasts':contrasts,'aggregation_verified_from_all_15_csv_files':True,
            'training_audit':{'runs':60,'development_histories_checked':45,'updates_per_development_run':2400,
                'all_shared_initializations_matched_within_seed':True,'all_initializations_official_imagenet_no_icg_warmstart':True,
                'selection_first_validation_maximum_verified':True},
            'independent_test_patients':3,'boundary_test_frames':23,'region_test_patients':2,'region_test_frames':11}


def figures(study, analysis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'font.size':11,'savefig.facecolor':'white'})
    out = study/'analysis/figures'; out.mkdir(parents=True,exist_ok=True)
    colors = ['#446B91','#54A19A','#B58957','#287D70','#9280A4']
    groups = analysis['groups']; x=np.arange(5)
    fig,ax=plt.subplots(figsize=(9,5.3),layout='constrained')
    means=[groups[g]['metrics']['boundary_f1']['mean'] for g in 'ABCDE']
    sd=[groups[g]['metrics']['boundary_f1']['seed_sd'] for g in 'ABCDE']
    ax.bar(x,means,color=colors,width=.6,alpha=.9)
    ax.errorbar(x,means,yerr=sd,fmt='none',ecolor='#263443',capsize=5,linewidth=1.4)
    for i,g in enumerate('ABCDE'):
        for j,r in enumerate(groups[g]['runs']):
            ax.plot(i+(j-1)*.09,r['boundary_f1'],'o',color='#1D2834',markersize=4)
        ax.annotate(f'{means[i]:.3f}',(i,means[i]+sd[i]+.025),ha='center',fontsize=11)
    ax.set(xticks=x,xticklabels=list('ABCDE'),ylim=(0,max(1.02,max(m+s for m,s in zip(means,sd))+.08)),ylabel='患者宏平均 Boundary F1',
           title='独立测试：3位患者 / 23帧 / 每组3个种子')
    ax.grid(axis='y',alpha=.14); ax.set_axisbelow(True)
    fig.savefig(out/'primary_boundary_f1.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.3),layout='constrained')
    for patient,color in zip(sorted(groups['A']['per_patient_seed_mean']),['#32658F','#BD8151','#748D54']):
        vals=[groups[g]['per_patient_seed_mean'][patient] for g in 'ABCDE']
        ax.plot(x,vals,'o-',label=patient,color=color,linewidth=1.7)
    ax.set(xticks=x,xticklabels=list('ABCDE'),ylim=(0,1),ylabel='Boundary F1（3种子均值）',title='逐患者结果：保留异质性')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.14)
    fig.savefig(out/'per_patient_boundary_f1.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained')
    for group,color in zip('ABCDE',colors):
        traces=[]
        for fold in ['fold_1','fold_2','fold_3']:
            h=load(study/f'runs/{group}_{fold}_s42/history.json')
            xs=[r['update'] for r in h];ys=[r['validation_patient_macro_f1'] for r in h]
            traces.append(ys);ax.plot(xs,ys,color=color,alpha=.14,linewidth=.8)
        ax.plot(xs,np.mean(traces,axis=0),color=color,label=group,linewidth=1.7)
    ax.set(ylim=(0,1),xlabel='优化器更新数',ylabel='验证 Boundary F1',title='开发曲线：固定种子42，粗线为3折均值')
    ax.legend(ncol=5,frameon=False);ax.grid(alpha=.12)
    fig.savefig(out/'development_curves.png',dpi=220);plt.close(fig)


def replay(study, analysis):
    # Import the frozen scientific implementation, not a later working checkout.
    sys.path.insert(0,str(study/'code/src'))
    from icg_pipeline.boundary_study import MixedBoundaryUNet, normalized_image, prediction_points
    from PIL import Image,ImageDraw,ImageFont
    torch.set_num_threads(3);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    candidates=[r for r in table(study/'candidate_manifest.csv') if r['planned_partition']=='test']
    caches={}
    for row in candidates:
        with np.load(study/'data/sealed_test'/f"{row['sample_id']}.npz",allow_pickle=False) as z:
            caches[row['sample_id']]={k:z[k] for k in ['image','reference_points','original_size']}
    panels={};max_error=0.;latencies={};pipeline_times={}
    lock=load(study/'checkpoint_lock.json')
    for group in 'ABCDE':
        item=next(r for r in lock['checkpoints'] if r['group']==group and r['seed']==42)
        model=MixedBoundaryUNet(group in 'BDE',100042).cuda().eval()
        ck=torch.load(study/'runs'/item['run']/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);threshold=item['best_threshold']
        original={r['sample_id']:r for r in table(study/'runs'/item['run']/'test/per_image.csv')}
        latencies[group]=[];pipeline_times[group]=[]
        with torch.inference_mode():
            for row in candidates:
                sid=row['sample_id'];data=caches[sid]
                pipeline_start=time.perf_counter()
                tensor=normalized_image(data['image'])[None].cuda()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.autocast('cuda',dtype=torch.float16): out=model(tensor)
                end.record();end.synchronize();latencies[group].append(start.elapsed_time(end))
                prob=(out['boundary_logits'].float().sigmoid()[0].cpu().numpy() if group in 'BDE'
                      else out['logits'].float().softmax(1)[0,1].cpu().numpy())
                pts=prediction_points(prob,threshold,'boundary' if group in 'BDE' else 'region',data['original_size'])
                pipeline_times[group].append((time.perf_counter()-pipeline_start)*1000)
                truth=data['reference_points']
                if len(pts):
                    pp=float(np.mean(cKDTree(truth).query(pts)[0]<=20));rr=float(np.mean(cKDTree(pts).query(truth)[0]<=20))
                    score=2*pp*rr/(pp+rr) if pp+rr else 0.
                else: score=0.
                max_error=max(max_error,abs(score-float(original[sid]['f1'])))
                panels[(sid,group)]={'points':pts,'f1':float(original[sid]['f1'])}
        del model;torch.cuda.empty_cache()
    if max_error>1e-10:
        raise ValueError(f'Visualization replay does not reproduce recorded boundary F1: {max_error}')
    outdir=study/'analysis/cases';outdir.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    case_files=[]
    def panel(image,points,color):
        canvas=Image.fromarray(image).resize((640,360))
        pixels=np.zeros((360,640),np.uint8)
        if len(points):
            pts=np.rint((points+.5)*np.array([640/3840,360/2160])-.5).astype(int)
            pts[:,0]=np.clip(pts[:,0],0,639);pts[:,1]=np.clip(pts[:,1],0,359)
            pixels[pts[:,1],pts[:,0]]=1
            pixels=cv2.dilate(pixels,np.ones((2,2),np.uint8))
        ar=np.array(canvas);ar[pixels>0]=color
        return Image.fromarray(ar)
    for row in candidates:
        sid=row['sample_id'];data=caches[sid]
        canvas=Image.new('RGB',(1952,1250),'white');draw=ImageDraw.Draw(canvas)
        draw.text((20,12),f"{sid} | {row['patient_source_id']} | {row['task_stratum']} | seed 42",font=font,fill='#17334D')
        images=[('原图',Image.fromarray(data['image']).resize((640,360))),
                ('现有参考边界',panel(data['image'],data['reference_points'],(0,228,221)))]
        for g in 'ABCDE':
            images.append((f"{g} | Boundary F1={panels[(sid,g)]['f1']:.3f}",panel(data['image'],panels[(sid,g)]['points'],(255,193,72))))
        for i,(label,image) in enumerate(images):
            x=12+(i%3)*648;y=56+(i//3)*396
            draw.text((x+5,y),label,font=small,fill='#283C4E');canvas.paste(image,(x,y+28))
        draw.text((668,900),'青色：参考；金色：预测\n固定种子42用于展示，主表使用3种子统计。\n折线不填充区域，遮挡下无参考的界面不补造。',font=small,fill='#465A6B',spacing=9)
        path=outdir/(sid+'.jpg');canvas.save(path,quality=94);case_files.append(str(path))
    analysis['replay_audit']={'seed42_models':5,'frames_per_model':23,'metric_checks':115,
        'maximum_boundary_f1_absolute_error':max_error,'replay_is_visualization_not_new_model_selection':True}
    analysis['gpu_forward_ms_replay']={g:{'median':float(np.median(v[2:])),'p95':float(np.percentile(v[2:],95)),
        'excludes_preprocessing_postprocessing_transfer':True,'first_two_frames_excluded':True} for g,v in latencies.items()}
    analysis['cached_frame_pipeline_ms_replay']={g:{'median':float(np.median(v[2:])),'p95':float(np.percentile(v[2:],95)),
        'includes_normalization_transfer_forward_boundary_postprocessing':True,
        'excludes_source_frame_capture_decode_resize_and_metric_computation':True,
        'is_not_live_video_end_to_end_latency':True} for g,v in pipeline_times.items()}
    analysis['case_files']=case_files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True)
    args=parser.parse_args();study=Path(args.study).resolve()
    if not (study/'final_summary.json').exists(): raise ValueError('Final results not completed')
    analysis=aggregate(study)
    figures(study,analysis);replay(study,analysis)
    out=study/'analysis';out.mkdir(exist_ok=True)
    (out/'report_data.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'analysis':str(out),'verified_csv_files':15,'case_panels':len(analysis['case_files']),
                      'replay_audit':analysis['replay_audit']},ensure_ascii=False))


if __name__=='__main__': main()
