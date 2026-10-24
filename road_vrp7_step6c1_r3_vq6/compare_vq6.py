#!/usr/bin/env python3
"""Strict paired 30-sec Road.replay analysis. Descriptive, never treat seeds as tasks."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

ARMS=('C3','DEFER_RESID','E0_MATCHED','RESID_OOF50')
LOCK=('source_sha256','road_sha256','edge_sha256','solver_sha256',
      'section_sha256','wall_sec','max_iterations','anneal_iterations','top_k',
      'seeds','task_ids','scenario_ids','split')
COUPLED=('action_count','sampler_seed','r_score_every','r_max_scoring_sec','r_fallback')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def load(folder):
    folder=Path(folder)
    with (folder/'manifest.json').open() as f:manifest=json.load(f)
    with (folder/'summary.json').open() as f:summary=json.load(f)
    if summary.get('status')!='COMPLETE':raise ValueError('Run is incomplete: '+str(folder))
    with (folder/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=summary['records'] or len(rows)!=summary['expected_records']:
        raise ValueError('Online row count is inconsistent: '+str(folder))
    keys=[(r['task_id'],r['scenario_id'],int(r['seed'])) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate online experiment key')
    diagnostics=None
    if (folder/'r2_diagnostics.json').exists():
        diagnostics=json.loads((folder/'r2_diagnostics.json').read_text())
    return manifest,dict(zip(keys,rows)),diagnostics

def paired_report(sources):
    if set(sources)!=set(ARMS):raise ValueError('Exactly C3,DEFER_RESID,E0_MATCHED,RESID_OOF50 required')
    dat={k:load(v) for k,v in sources.items()};ref=dat['C3'][0]
    for k,(manifest,rows,diagnostics) in dat.items():
        if any(manifest.get(field)!=ref.get(field) for field in LOCK):
            raise ValueError('30-sec setup differs from C3: '+k)
        if k!='C3' and any(manifest.get(field)!=dat['E0_MATCHED'][0].get(field) for field in COUPLED):
            raise ValueError('Learned selector budgets/sampler differ: '+k)
        if set(rows)!=set(dat['C3'][1]):raise ValueError('Paired task/scenario/seed keys differ: '+k)
        if manifest.get('arm') not in (('C3',) if k=='C3' else ('learned',)):
            raise ValueError('Expected arm identity mismatch: '+k)
        if k!='C3':
            meta_path=Path(sources[k])/'vq6_provenance.json'
            if not meta_path.is_file():raise ValueError('Missing VQ6 arm provenance: '+k)
            arm_meta=json.loads(meta_path.read_text())
            if (arm_meta.get('stage')!='R3-VQ6-online-v1' or arm_meta.get('kind')!=k or
                arm_meta.get('model_sha256')!=manifest.get('model_sha256')):
                raise ValueError('Wrong VQ6 deployed model identity: '+k)
            if diagnostics is None or len(diagnostics.get('runs',[]))!=len(rows):
                raise ValueError('Missing real-C3 diagnostics: '+k)
            if k=='DEFER_RESID' and any(int(v['r2_effective_overrides']) for v in diagnostics['runs']):
                raise ValueError('Forced-defer control overrode C3')
            if not any(int(row['r_model_eval_calls'])>0 for row in rows.values()):
                raise ValueError('No learned scorer calls: '+k)
    keys=sorted(dat['C3'][1]); detail=[]
    def cost(k,arm):return float(dat[arm][1][k]['ex_post_road_travel_sec'])
    for k in keys:
        row={'task_id':k[0],'scenario_id':k[1],'seed':k[2]}
        for arm in ARMS:
            r=dat[arm][1][k]
            row[arm+'_road_sec']=cost(k,arm)
            row[arm+'_iterations']=int(r['iterations'])
            row[arm+'_search_wall_sec']=float(r['search_wall_sec'])
            row[arm+'_scoring_wall_sec']=float(r['r_scoring_seconds'])
            row[arm+'_model_actions']=int(r['r_model_actions'])
        detail.append(row)
    summaries={}
    for a,b in [('DEFER_RESID','C3'),('E0_MATCHED','C3'),('RESID_OOF50','C3'),
                ('RESID_OOF50','DEFER_RESID'),('RESID_OOF50','E0_MATCHED'),('E0_MATCHED','DEFER_RESID')]:
        delta=np.array([r[b+'_road_sec']-r[a+'_road_sec'] for r in detail],float)
        by_task=defaultdict(list);by_case=defaultdict(list)
        for r,y in zip(detail,delta):
            by_task[r['task_id']].append(float(y));by_case[(r['task_id'],r['scenario_id'])].append(float(y))
        summaries[f'{a}_vs_{b}']={'n_pairs':len(delta),
            'mean_road_improvement_sec':float(delta.mean()),'median_road_improvement_sec':float(np.median(delta)),
            'wins':int((delta>1e-9).sum()),'losses':int((delta< -1e-9).sum()),'ties':int((abs(delta)<=1e-9).sum()),
            'mean_by_task_sec':{t:float(np.mean(v)) for t,v in sorted(by_task.items())},
            'mean_by_task_scenario_sec':{f'{t}|{s}':float(np.mean(v)) for (t,s),v in sorted(by_case.items())},
            'mean_overhead_scoring_sec':float(np.mean([r[a+'_scoring_wall_sec'] for r in detail])),
            'mean_effective_model_actions':float(np.mean([r[a+'_model_actions'] for r in detail]))}
    return {'stage':'R3-VQ6-FROZEN-30SEC-PAIRED-v1','status':'COMPLETE',
        'n_task_groups':len({x['task_id'] for x in detail}),'n_scenario_groups':len({x['scenario_id'] for x in detail}),
        'n_pairs':len(detail),'summaries':summaries,'details':detail,
        'source_sha256':{k:{'manifest':sha256(Path(v)/'manifest.json'),
                            'metrics':sha256(Path(v)/'metrics.csv')} for k,v in sources.items()},
        'limits':['True 30-sec online Road.replay cost; NOT offline H-step restart benefit',
                  'Training states only from original TRAIN tasks',
                  'Cases with the same task/scenario but different seeds are correlated',
                  'A force-defer model computes features and predictions and commits real C3; wall-clock overhead can change search iterations',
                  'July validation has been inspected; this is DEVELOPMENT, not blind independent testing']}

def main():
    p=argparse.ArgumentParser()
    for arm in ARMS:p.add_argument('--'+arm.lower().replace('_','-'),required=True)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=Path(a.output_dir)
    if out.exists():raise FileExistsError('New independent comparison directory required')
    paths={arm:getattr(a,arm.lower()) for arm in ARMS}
    report=paired_report(paths);out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    md=['# VQ6 冻结30秒ALNS配对实验','','> Road.replay真实道路成本；较小越好。下面正改善表示左策略比右策略好。','',
        '| 对照 | 均值改善(s) | 胜/负/平 | 模型评分均耗时(s) | 平均介入次数 |','|---|---:|---:|---:|---:|']
    for comp,z in report['summaries'].items():
        md.append(f"| {comp} | {z['mean_road_improvement_sec']:+.3f} | {z['wins']}/{z['losses']}/{z['ties']} | {z['mean_overhead_scoring_sec']:.3f} | {z['mean_effective_model_actions']:.2f} |")
    md+=['','**注意：** 同任务/场景下多种子不独立；此次为曾查看过的七月开发场景。']
    (out/'report.md').write_text('\n'.join(md)+'\n')
    print('VQ6 paired comparison COMPLETE:',out)
    print('\n'.join(md))
if __name__=='__main__':main()
