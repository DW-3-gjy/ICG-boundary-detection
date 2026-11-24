"""Predeclared descriptive analysis for one held-out patient and three seeds."""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'ICG-boundary-detection/model/logs/cross_patient_20261022_01'
ORDER=['unet_control','unet_tversky','unet_combined','unet_resnet18_control',
       'unet_resnet18_tversky','unet_resnet18_combined','resnet18_scratch_bn_train','resnet18_imagenet_bn_train']
SHORT=['UD','UT','UB','RD','RT','RB','RS','RP']
NAMES=['U-Net Dice CE','U-Net Tversky CE','U-Net 组合损失','ResNet ImageNet Dice CE',
       'ResNet ImageNet Tversky CE','ResNet ImageNet 组合损失','ResNet 随机 BN可训练','ResNet ImageNet BN可训练']
LABELS=dict(zip(ORDER,NAMES))
METRICS=['dice','iou','precision','recall','hausdorff_95','mean_surface_distance','area_ratio']
CONTRASTS=[
    ('完整方案对基础方案','unet_resnet18_combined','unet_control'),
    ('U-Net Tversky组合','unet_tversky','unet_control'),
    ('U-Net 边界组合调整','unet_combined','unet_tversky'),
    ('ResNet Tversky组合','unet_resnet18_tversky','unet_resnet18_control'),
    ('ResNet 边界组合调整','unet_resnet18_combined','unet_resnet18_tversky'),
    ('配对预训练消融','resnet18_imagenet_bn_train','resnet18_scratch_bn_train'),
    ('编码器BN冻结敏感性','unet_resnet18_control','resnet18_imagenet_bn_train'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def stats(values):
    values=np.asarray([np.inf if v is None else v for v in values],float)
    finite=values[np.isfinite(values)]
    return {'mean':float(values.mean()) if len(finite)==len(values) else None,
            'sd':float(values.std(ddof=1)) if len(values)>1 and len(finite)==len(values) else None,
            'values':safe(values.tolist()),'nonfinite_count':int(len(values)-len(finite)),
            'finite_mean':float(finite.mean()) if len(finite) else None}


def freeze():
    path=STUDY/'analysis_plan.json'
    if path.exists():raise FileExistsError('Analysis plan already exists')
    if (STUDY/'test_evaluation_started.json').exists():raise RuntimeError('Must predeclare before test')
    plan={'frozen_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'training_protocol_sha256':(STUDY/'protocol.sha256').read_text(),
          'condition_order':ORDER,'contrasts':[list(v) for v in CONTRASTS],
          'primary_population':'patient2 primary_Z_all 25 images',
          'primary_contrast':'unet_resnet18_combined minus unet_control',
          'primary_metric':'mean foreground Dice across images, then mean across seeds',
          'uncertainty':'sample SD across three seeds is algorithm variability conditional on one patient; no patient CI or p-value',
          'precision_convention':'Report precision with zero_division=0 for empty predictions; preserve legacy smoothed precision in raw files.',
          'nonfinite_distances':'nonfinite retained; mean undefined if any failing frame, finite-only values separately labelled',
          'image_case_selection':{'seed':42,'models':['unet_control','unet_resnet18_combined'],
              'rules':['largest Dice gain','largest Dice loss','median Dice change','lowest proposed Dice in Z+L subgroup'],
              'tie_break':'filename','deduplicate':'choose next eligible rank for a distinct case'},
          'all_test_metrics_are_descriptive':True,'no_model_selection_on_test':True,
          'script_sha256':digest(__file__)}
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    (STUDY/'analysis_plan.sha256').write_text(digest(path),encoding='ascii')
    print('Analysis plan frozen before patient2 model evaluation.')


def main():
    if digest(STUDY/'analysis_plan.json')!=(STUDY/'analysis_plan.sha256').read_text():raise ValueError('Analysis plan changed')
    analysis_plan=json.loads((STUDY/'analysis_plan.json').read_text(encoding='utf-8'))
    if digest(__file__)!=analysis_plan['script_sha256']:raise ValueError('Analysis script changed after predeclaration')
    protocol=json.loads((STUDY/'protocol.json').read_text(encoding='utf-8'))
    test=json.loads((STUDY/'test_summary.json').read_text(encoding='utf-8'))
    validation=json.loads((STUDY/'validation_summary.json').read_text(encoding='utf-8'))
    if not test['complete'] or len(test['runs'])!=24 or not validation['complete']:raise RuntimeError('Incomplete study')
    if test['protocol_sha256']!=(STUDY/'protocol.sha256').read_text():raise ValueError('Protocol mismatch')
    os.environ['MPLCONFIGDIR']=str(STUDY/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image,ImageDraw,ImageFont
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':240,'figure.facecolor':'white'})
    dest=STUDY/'analysis';figs=dest/'figures'
    figs.mkdir(parents=True,exist_ok=True)
    runs={r['run_id']:r for r in test['runs']}
    raw={key:json.loads((STUDY/'runs'/key/'test/evaluation.json').read_text(encoding='utf-8')) for key in runs}
    grouped={}
    for condition in ORDER:
        definitions=sorted([r for r in test['runs'] if r['condition_id']==condition],key=lambda r:r['seed'])
        assert [r['seed'] for r in definitions]==[42,43,44]
        groups={}
        for group,filenames in protocol['test_subgroups'].items():
            summary={}
            for metric in METRICS:
                seed_means=[]
                for run in definitions:
                    samples=[row for row in raw[run['run_id']]['per_sample'] if row['filename'] in filenames]
                    values=[(0.0 if row['empty_prediction'] else row['precision']) if metric=='precision' else row[metric] for row in samples]
                    assert len(values)==len(filenames)
                    seed_means.append(float(np.mean([np.inf if v is None else v for v in values])))
                summary[metric]=stats(seed_means)
            summary['n_images']=len(filenames)
            summary['empty_predictions_by_seed']=[sum(row['empty_prediction'] for row in raw[r['run_id']]['per_sample']
                if row['filename'] in filenames) for r in definitions]
            groups[group]=summary
        val_runs=sorted([r for r in validation['runs'] if r['condition_id']==condition],key=lambda r:r['seed'])
        grouped[condition]={'name':LABELS[condition],'groups':groups,
            'validation_dice':stats([r['fullres_validation']['mean_dice'] for r in val_runs]),
            'best_epochs':[r['best_epoch'] for r in val_runs],
            'training_seconds':stats([r['training_seconds'] for r in val_runs]),
            'optimizer_updates':[r['optimizer_updates'] for r in val_runs]}
    contrasts=[]
    for name,a,b in CONTRASTS:
        result={'name':name,'condition_a':a,'condition_b':b,'groups':{}}
        for group in ('primary_Z_all','Z_original_label','Z_instrument_occluded'):
            result['groups'][group]={}
            for metric in ('dice','hausdorff_95','recall'):
                av=grouped[a]['groups'][group][metric]['values'];bv=grouped[b]['groups'][group][metric]['values']
                values=[None if x is None or y is None else x-y for x,y in zip(av,bv)]
                result['groups'][group][metric]=stats(values)
        contrasts.append(result)
    initial_checks=[]
    for seed in (42,43,44):
        a=json.loads((STUDY/'runs'/f'resnet18_scratch_bn_train_seed_{seed}'/'initialization.json').read_text())
        b=json.loads((STUDY/'runs'/f'resnet18_imagenet_bn_train_seed_{seed}'/'initialization.json').read_text())
        initial_checks.append({'seed':seed,'same_initial_nonencoder':a['nonencoder_state_sha256']==b['nonencoder_state_sha256'],
                               'different_initial_encoder':a['encoder_state_sha256']!=b['encoder_state_sha256']})
    primary_ids=protocol['test_subgroups']['primary_Z_all']
    a={r['filename']:r for r in raw['unet_control_seed_42']['per_sample'] if r['filename'] in primary_ids}
    b={r['filename']:r for r in raw['unet_resnet18_combined_seed_42']['per_sample'] if r['filename'] in primary_ids}
    differences=sorted([{'filename':f,'delta':b[f]['dice']-a[f]['dice'],'baseline_dice':a[f]['dice'],
                         'proposed_dice':b[f]['dice']} for f in a],key=lambda r:(r['delta'],r['filename']))
    selected=[]
    def choose(name,candidates):
        item=next(r for r in candidates if r['filename'] not in {x['filename'] for x in selected})
        selected.append({'selection':name,**item})
    choose('最大改善',list(reversed(differences)))
    choose('最大退步',differences)
    choose('变化中位附近',sorted(differences,key=lambda r:(abs(r['delta']-np.median([x['delta'] for x in differences])),r['filename'])))
    choose('遮挡子集较差案例',sorted([r for r in differences if r['filename'] in protocol['test_subgroups']['Z_instrument_occluded']],
                                 key=lambda r:(r['proposed_dice'],r['filename'])))
    summary={'protocol_sha256':test['protocol_sha256'],'analysis_plan_sha256':digest(STUDY/'analysis_plan.json'),
             'conditions':grouped,'contrasts':contrasts,'pretraining_initialization_checks':initial_checks,
             'selected_examples':selected,'primary_seed42_frame_deltas':differences,
             'runs_completed':24,'test_patient_count':1,'patient_count_total':2,
             'precision_report_zero_division':0,'inference_scope':'static image segmentation only'}
    (dest/'summary.json').write_text(json.dumps(safe(summary),ensure_ascii=False,indent=2),encoding='utf-8')
    import csv
    with (dest/'condition_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['condition','group','n_images','metric','mean','sd','seed42','seed43','seed44','nonfinite_count']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for condition,item in grouped.items():
            for group,values in item['groups'].items():
                for metric in METRICS:
                    s=values[metric]
                    writer.writerow({'condition':condition,'group':group,'n_images':values['n_images'],'metric':metric,
                                     'mean':s['mean'],'sd':s['sd'],'seed42':s['values'][0],'seed43':s['values'][1],
                                     'seed44':s['values'][2],'nonfinite_count':s['nonfinite_count']})
    palette=['#24639A']*3+['#B86B20']*3+['#68727C','#8062A5']
    def save(fig,name):
        fig.savefig(figs/(name+'.png'),bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for ax,metric,title in [(axes[0],'dice','确认Z阶段 25张'),(axes[1],'hausdorff_95','标注轮廓HD95 原图像素')]:
        for i,condition in enumerate(ORDER):
            st=grouped[condition]['groups']['primary_Z_all'][metric]
            if st['mean'] is None:
                ax.text(i,0,'非有限',rotation=90,ha='center',va='bottom');continue
            ax.bar(i,st['mean'],color=palette[i],width=.65,alpha=.85,yerr=st['sd'],capsize=3)
            ax.scatter([i-.12,i,i+.12],st['values'],s=13,color='black',zorder=3)
        ax.set_xticks(range(8),SHORT);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if metric=='dice':ax.set_ylim(0,1);ax.set_ylabel('前景Dice')
        else:ax.set_ylabel('像素')
    fig.suptitle('跨患者测试 每点一个固定种子 误差线为种子间样本标准差',fontsize=11)
    fig.tight_layout();save(fig,'primary_results')
    fig,ax=plt.subplots(figsize=(10.8,4))
    for i,group in enumerate(['Z_original_label','Z_instrument_occluded','primary_Z_all']):
        means=[grouped[c]['groups'][group]['dice']['mean'] for c in ORDER]
        deviations=[grouped[c]['groups'][group]['dice']['sd'] for c in ORDER]
        ax.bar(np.arange(8)+(i-1)*.24,means,width=.23,yerr=deviations,capsize=2,
               label=['原Z组 12张','Z加L遮挡组 13张','全部Z 25张'][i])
    ax.set_xticks(range(8),SHORT);ax.set_ylim(0,1);ax.set_ylabel('前景Dice');ax.legend(ncol=3,fontsize=9)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'occlusion_strata')
    fig,ax=plt.subplots(figsize=(10.8,4.8))
    for i,contrast in enumerate(contrasts):
        st=contrast['groups']['primary_Z_all']['dice']
        ax.errorbar(st['mean']*100,i,xerr=st['sd']*100,fmt='o',capsize=4,color='#24639A')
        ax.scatter([v*100 for v in st['values']],[i]*3,s=16,color='#B86B20',zorder=3)
    ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(len(contrasts)),[r['name'] for r in contrasts]);ax.invert_yaxis()
    ax.set_xlabel('配对种子的Dice差值 百分点');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();save(fig,'planned_contrasts')
    fig,axes=plt.subplots(3,2,figsize=(10.8,10),sharex=True,sharey=True)
    for ax,condition in zip(axes.flat,ORDER[:6]):
        histories=[json.loads((STUDY/'runs'/f'{condition}_seed_{seed}'/'checkpoints/history.json').read_text()) for seed in (42,43,44)]
        epochs=np.arange(1,121)
        for part,color,style in [('train','#B86B20','--'),('val','#24639A','-')]:
            arrays=np.array([[e[part]['dice'] for e in h] for h in histories]);mean=arrays.mean(0);sd=arrays.std(0,ddof=1)
            ax.plot(epochs,mean,color=color,ls=style,label='训练' if part=='train' else '患者内验证')
            ax.fill_between(epochs,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=color,alpha=.12)
        ax.set_title(LABELS[condition],fontsize=10);ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('Epoch');axes[-1,1].set_xlabel('Epoch')
    fig.tight_layout();save(fig,'learning_curves')
    heat=np.array([[raw[f'{condition}_seed_42']['per_sample'][next(i for i,r in enumerate(raw[f'{condition}_seed_42']['per_sample']) if r['filename']==f)]['dice']
                   for f in primary_ids] for condition in ORDER])
    fig,ax=plt.subplots(figsize=(11.5,4));im=ax.imshow(heat,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set_yticks(range(8),SHORT);ax.set_xticks(range(25),[f.replace('p2_revised_','').replace('.png','') for f in primary_ids],rotation=90,fontsize=8)
    ax.set_xlabel('患者2匿名帧编号 每帧并非独立病例');fig.colorbar(im,ax=ax,label='Dice')
    fig.tight_layout();save(fig,'per_frame_heatmap_seed42')
    # Fixed rule-driven examples: whole frames preserve the actual image context.
    for index,item in enumerate(selected,1):
        filename=item['filename'];stem=Path(filename).stem
        paths=[Path(protocol['data_root'])/'fullres/test/images'/filename,
               STUDY/'runs/unet_control_seed_42/test/overlays'/(stem+'.jpg'),
               STUDY/'runs/unet_resnet18_combined_seed_42/test/overlays'/(stem+'.jpg')]
        canvas=Image.new('RGB',(1956,440),'white');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
        for col,(p,label) in enumerate(zip(paths,['原图','基础方案 UD','改进方案 RB'])):
            with Image.open(p) as image:thumb=image.convert('RGB').resize((640,360),Image.Resampling.LANCZOS)
            x=col*652+6;draw.text((x,4),label,font=font,fill='black');canvas.paste(thumb,(x,38))
        draw.text((8,406),f"{item['selection']}  {filename}  Dice {item['baseline_dice']:.3f} → {item['proposed_dice']:.3f}",font=font,fill='black')
        canvas.save(figs/f'case_{index:02d}.jpg',quality=95)
    print(json.dumps({'analysis_complete':True,'conditions':8,'test_runs':24,
                      'primary_Dice_delta':contrasts[0]['groups']['primary_Z_all']['dice']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-analysis-only',action='store_true')
    args=parser.parse_args()
    freeze() if args.freeze_analysis_only else main()
