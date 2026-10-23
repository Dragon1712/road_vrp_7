#!/usr/bin/env python3
"""Independent shard-resumable 30-second R3 paired pilot, SHA-pinned end to end.

Defaults are July-25 development scenarios with NEW solver seeds (60-64).
Not a blind/generalization test, and no July-22/24 dataset selection.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,subprocess,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1'
OLD_R2='runs/chicago_july_v1/step6c1_r2/r2_task_seed42_v1/R2_F2.joblib'
M0='runs/chicago_july_v1/step6b2/b2_full_seed42_v1/task/P2_F2_model.joblib'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model_info(a):
    if a.arm=='C3':return None,'F0','C3'
    if a.arm=='M0':return M0,'F2','M0'
    if a.arm=='T0_R2_F2':return OLD_R2,'F2','R2'
    if a.arm not in ('T1_CB_RMSE','T2_CB_HUBER','T3_PAIRWISE',
      'T4_WEIGHTED_PAIR','T5_HUBER_MARGIN','T6_WEIGHTED_MARGIN','T7_BLEND_MARGIN'):
        raise ValueError('Unknown arm '+a.arm)
    return str(Path(a.model_study)/(a.arm+'.joblib')),'F2','R3'


def plan(a):
    model,mode,engine=model_info(a)
    tasks=a.tasks.split(',');scenarios=a.scenarios.split(',');seeds=[int(x) for x in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or not all(tasks+scenarios) or len(set(tasks))!=len(tasks) or len(set(scenarios))!=len(scenarios):
        raise ValueError('Duplicate/bad task, scenario or seed list')
    frozen={
        'frozen_solver_sha256':digest(ROOT/'road_vrp7_step5b_v3/run_experiment.py'),
        'frozen_selector_sha256':digest(ROOT/'road_vrp7_step5b_v3/section_v2.py'),
        'base_adapter_sha256':digest(ROOT/'road_vrp7_step6b2_r/benchmark_30s_r.py'),
        'original_online_sha256':digest(ROOT/'road_vrp7_step6c1_r2/online_r2.py'),
        'r3_online_sha256':digest(ROOT/'road_vrp7_step6c1_r3_tricks/online_r3.py'),
        'r3_models_sha256':digest(ROOT/'road_vrp7_step6c1_r3_tricks/r3_models.py'),
        'study_code_sha256':digest(Path(__file__)),
        'merge_code_sha256':digest(Path(__file__).with_name('r3_merge.py'))}
    return {'stage':'step6c1r3t-pilot-v1','arm':a.arm,'tasks':tasks,'scenarios':scenarios,
       'seeds':seeds,'split':a.split,'model_path':model,'model_sha256':digest(model) if model else None,
       'model_mode':mode,'engine':engine,
       'source_path':a.model_source,'source_sha256':digest(Path(a.model_source)/'run_config.json'),
       **frozen,'wall_sec':30,'max_iterations':20000,'anneal_iterations':5000,
       'top_k':12,'action_count':8,'sampler_seed':20261019,'score_every':8,
       'max_scoring_sec':2.,'cache_size':64 if a.arm=='M0' else 0,
       'notes':['Development July-25 / new solver seeds; not blind final test',
                'C3 top-20% vs learned stratified 8 candidate mechanisms unchanged',
                'M0 uses its historical cache64; R2/R3 cache0 for RNG-dependent real C3 actions']}


def command(c,task,sc,folder):
    exe={'C3':'road_vrp7_step6b2_r/benchmark_30s_r.py',
         'M0':'road_vrp7_step6b2_r/benchmark_30s_r.py',
         'R2':'road_vrp7_step6c1_r2/online_r2.py',
         'R3':'road_vrp7_step6c1_r3_tricks/online_r3.py'}[c['engine']]
    cmd=[sys.executable,exe,'--output-dir',str(folder),'--control','C3' if c['engine']=='C3' else 'learned',
         '--model-method','P2','--model-mode',c['model_mode'],'--model-protocol','task',
         '--model-source',c['source_path'],'--seeds',','.join(map(str,c['seeds'])),
         '--split',c['split'],'--task-ids',task,'--scenario-ids',sc,
         '--wall-sec','30','--max-iterations','20000','--anneal-iterations','5000',
         '--top-k','12','--action-count','8','--sampler-seed','20261019',
         '--score-every','8','--max-scoring-sec','2','--cache-size',str(c['cache_size']),
         '--min-static-fraction','0','--fallback','c3']
    if c['model_path']:cmd.extend(['--model-path',c['model_path']])
    return cmd


def check_shard(folder,task,sc,seeds,plan):
    if not (folder/'summary.json').exists() or not (folder/'metrics.csv').exists() or not (folder/'manifest.json').exists():return False
    summary=json.loads((folder/'summary.json').read_text())
    if summary.get('status')!='COMPLETE' or summary.get('records')!=len(seeds):return False
    mf=json.loads((folder/'manifest.json').read_text())
    for actual,expect in (('source_sha256','source_sha256'),('solver_sha256','frozen_solver_sha256'),
              ('section_sha256','frozen_selector_sha256'),('model_sha256','model_sha256'),
              ('wall_sec','wall_sec'),('max_iterations','max_iterations'),('top_k','top_k')):
        if mf[actual]!=plan[expect]:raise ValueError('SHA/frozen constraint changed '+actual)
    with (folder/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    expected={(task,sc,int(s)) for s in seeds}
    if {(r['task_id'],r['scenario_id'],int(r['seed'])) for r in rows}!=expected or len(rows)!=len(expected):return False
    if not all(math.isfinite(float(r['ex_post_road_travel_sec'])) for r in rows):return False
    return True


def run(a):
    c=plan(a);root=Path(a.output_dir)
    if root.exists():
        if not a.resume:raise FileExistsError(root)
        if not (root/'plan.json').is_file() or json.loads((root/'plan.json').read_text())!=c:
            raise ValueError('Resume refused: plan / model SHA changed')
    elif not a.dry_run:
        root.mkdir(parents=True)
        (root/'plan.json').write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n')
    counter=0
    for task in c['tasks']:
        for sc in c['scenarios']:
            sh=root/'shards'/(task+'__'+sc)
            if sh.exists():
                if check_shard(sh,task,sc,c['seeds'],c):
                    print('SKIP verified',sh);continue
                raise ValueError('Shard exists but incomplete. Inspect/remove shard explicitly: '+str(sh))
            cli=command(c,task,sc,sh)
            if a.dry_run:print('DRY',' '.join(cli));continue
            sh.parent.mkdir(parents=True,exist_ok=True)
            print('RUN',task,sc,a.arm,flush=True)
            subprocess.run(cli,cwd=ROOT,check=True)
            if not check_shard(sh,task,sc,c['seeds'],c):raise ValueError('Shard output incomplete '+str(sh))
            counter+=1
            if a.max_new_shards and counter>=a.max_new_shards:break
        if a.max_new_shards and counter>=a.max_new_shards:break
    if a.dry_run:return
    complete=all(check_shard(root/'shards'/(t+'__'+sc),t,sc,c['seeds'],c)
          for t in c['tasks'] for sc in c['scenarios'])
    if complete:
        if (root/'summary.json').exists():
            expected=len(c['tasks'])*len(c['scenarios'])*len(c['seeds'])
            if json.loads((root/'summary.json').read_text()).get('records')!=expected:
                raise ValueError('Existing merge incomplete')
        else:
            from r3_merge import merge
            merge(root)
    else:print('PARTIAL, continue with RESUME=1',root)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--model-study',default='runs/chicago_july_v1/step6c1_r3_tricks/r3_seed42_v1')
    p.add_argument('--model-source',default=SRC)
    p.add_argument('--tasks',default='task_0008,task_0009')
    p.add_argument('--scenarios',default='2025-07-25_0800,2025-07-25_1700')
    p.add_argument('--seeds',default='60,61,62,63,64')
    p.add_argument('--split',choices=('train','validation','test'),default='validation')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--max-new-shards',type=int,default=0)
    run(p.parse_args())
