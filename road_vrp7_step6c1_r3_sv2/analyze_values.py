#!/usr/bin/env python3
"""State-clustered, paired analysis of SV2 frozen-C3 restart rollouts.

Do not treat repeated RNG seeds or multiple actions from the same captured state
as independent examples. This is a restart-surrogate diagnostic, not Q-learning.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

ARMS=('C3','M0_PROXY','R2_F2','T2_HUBER')

def readjson(p):return json.loads(Path(p).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def clustered_interval(values,seed=20261024,reps=4000):
    v=np.asarray(values,float)
    if not len(v):return None
    rng=np.random.default_rng(seed)
    i=rng.integers(0,len(v),size=(reps,len(v)))
    boot=np.mean(v[i],axis=1)
    return [float(x) for x in np.quantile(boot,[.025,.975])]

def analyze(rollouts,one_step,bootstrap=4000,seed=20261024):
    rows=[json.loads(x) for x in Path(rollouts,'rollouts.jsonl').read_text().splitlines() if x.strip()]
    manifest=readjson(Path(rollouts,'manifest.json'))
    if readjson(Path(rollouts,'report.json')).get('status')!='COMPLETE':raise ValueError('Rollout not COMPLETE')
    if manifest.get('stage')!='R3-SV-paired-restart-v1':raise ValueError('Unsupported restart stage')
    if sha(Path(one_step,'decisions.jsonl'))!=manifest['one_step_sha256'] or sha(Path(one_step,'pilot_states.json'))!=manifest['pilot_sha256']:
        raise ValueError('Source one-step/pilot differs from rollout manifest')
    pilot=readjson(Path(one_step,'pilot_states.json'))
    expected=set(pilot['selected_decision_ids']);Hs=set(map(int,manifest['horizons']));R=int(manifest['replicates'])
    if set(manifest['states'])!=expected:raise ValueError('Selected states differ from saved plan')
    choices={}
    for line in Path(one_step,'decisions.jsonl').read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line);k=(r['decision_id'],r['arm'])
        if k in choices:raise ValueError('Duplicate one-step decision '+str(k))
        choices[k]=r
    index={}
    for r in rows:
        k=(r['decision_id'],int(r['horizon']),int(r['replicate']),r['arm'])
        if k in index:raise ValueError('Duplicate rollout '+str(k))
        if r['decision_id'] not in expected or int(r['horizon']) not in Hs or int(r['replicate']) not in range(R) or r['arm'] not in ARMS:
            raise ValueError('Unexpected rollout key '+str(k))
        index[k]=r
    if len(index)!=len(expected)*len(Hs)*R*len(ARMS):raise ValueError('Incomplete paired rollouts')
    road_avail=all('final_ex_post_road_travel_sec' in r for r in rows)
    if manifest.get('road_replay') and not road_avail:raise ValueError('Declared road replay, but missing some values')
    metrics=['td_od']+(['road_replay'] if road_avail else [])
    results={}
    labels=[];placebo_total=0;placebo_violations=0
    for di in sorted(expected):
        for arm in ARMS[1:]:
            if (di,arm) not in choices or (di,'C3') not in choices:raise ValueError('Missing one-step row')
            for h in sorted(Hs):
                for rep in range(R):
                    a=index[(di,h,rep,arm)];b=index[(di,h,rep,'C3')]
                    if int(a['rollout_seed'])!=int(b['rollout_seed']):raise ValueError('CRN seed mismatch')
                    gain0=float(b['initial_td_od_cost'])-float(a['initial_td_od_cost'])
                    if abs(gain0-float(choices[(di,arm)]['gain_sec']))>.12:raise ValueError('H0 label mismatch')
                    if a['action_identity']==b['action_identity']:
                        placebo_total+=1
                        if abs(float(a['final_td_od_cost'])-float(b['final_td_od_cost']))>1e-6:
                            placebo_violations+=1
    if placebo_violations:raise ValueError('Identical action+seed yielded different TD-OD result; audit run determinism')
    # First average over RNG repeats, then compare across independent captured states.
    for metric in metrics:
        col='final_td_od_cost' if metric=='td_od' else 'final_ex_post_road_travel_sec'
        results[metric]={}
        for h in sorted(Hs):
            target={}
            for arm in ARMS[1:]:
                state_rec=[]
                for di in sorted(expected):
                    step=float(choices[(di,arm)]['gain_sec'])
                    gains=[float(index[(di,h,rep,'C3')][col])-float(index[(di,h,rep,arm)][col]) for rep in range(R)]
                    g0=[float(index[(di,h,rep,'C3')]['initial_td_od_cost'])-float(index[(di,h,rep,arm)]['initial_td_od_cost']) for rep in range(R)]
                    r=choices[(di,arm)]
                    state_rec.append({'decision_id':di,'task_id':r['task_id'],'scenario_id':r['scenario_id'],
                           'step_gain_td_od':step,'h0_td_od':float(np.mean(g0)),
                           'multi_gain':float(np.mean(gains)),
                           'replicate_sd':float(np.std(gains,ddof=1)) if R>1 else 0.,
                           'gains_by_replicate':gains,'same_as_c3':bool(int(r['selected_index'])==-1)})
                v=np.asarray([x['multi_gain'] for x in state_rec]);step=np.asarray([x['step_gain_td_od'] for x in state_rec]);
                # Directional disagreement across state MEANS (exclude exactly-zero signs).
                nz=(np.abs(step)>1e-6)&(np.abs(v)>1e-6)
                altered=np.asarray([not x['same_as_c3'] for x in state_rec])
                flips=int(np.count_nonzero(np.sign(v[nz])!=np.sign(step[nz])))
                signs_by_rep=[]
                for x in state_rec:
                    z=np.asarray(x['gains_by_replicate']);nz2=z[np.abs(z)>1e-6]
                    if len(nz2)>1:signs_by_rep.append(bool(np.any(nz2>0) and np.any(nz2<0)))
                task_means={t:float(np.mean([x['multi_gain'] for x in state_rec if x['task_id']==t])) for t in sorted(set(x['task_id'] for x in state_rec))}
                target[arm]={'n_states':len(v),'n_restarts':len(v)*R,
                  'mean_h0_step_sec':float(np.mean(step)),'mean_h_gain_sec':float(np.mean(v)),
                  'state_bootstrap_95pct_sec':clustered_interval(v,seed,bootstrap),
                  'state_wins_losses_ties':[int(sum(v>1e-6)),int(sum(v< -1e-6)),int(sum(np.abs(v)<=1e-6))],
                  'n_selected_different_action':int(np.sum(altered)),
                  'sign_flip_fraction_per_state':float(flips/int(np.sum(nz))) if np.any(nz) else None,
                  'n_nonzero_state_sign_pairs':int(np.sum(nz)),
                  'rng_sign_instability_fraction':float(np.mean(signs_by_rep)) if signs_by_rep else None,
                  'n_rng_sign_evaluable':len(signs_by_rep),
                  'mean_within_state_rng_sd_sec':float(np.mean([x['replicate_sd'] for x in state_rec])),
                  'per_task_mean':task_means}
                labels.extend([{**x,'arm':arm,'horizon':h,'metric':metric} for x in state_rec])
            results[metric][str(h)]=target
    report={'status':'COMPLETE','stage':'R3-SV2-state-aggregated-v1',
      'selection':{'type':pilot.get('sampling'),'role':pilot.get('role'),'n_states':len(expected),
                   'seed':pilot.get('seed')},
      'n_runs':len(index),'n_replicates':R,'horizons':sorted(Hs),
      'identical_action_placebo':{'n_checked':placebo_total,'violations':placebo_violations},
      'metrics':results,
      'interpretation_limits':['Relaunch from repaired route; original ALNS state/RNG/temperature not restored',
          'Future TD costs are retrospective evaluation only',
          'Captured states are correlated within original search runs and tasks',
          'This preexamined development/internal_test split is NOT a blind test',
          '48 or more states do not create extra independent VRP tasks/dates',
          'Road.replay is an ex-post metric, not supplied to search']}
    return report,labels

def write_analysis(rollout_dir,one_step_dir,out_dir,bootstrap=4000,seed=20261024):
    output=Path(out_dir)
    if output.exists():raise FileExistsError('New output directory required')
    report,labels=analyze(rollout_dir,one_step_dir,bootstrap,seed)
    output.mkdir(parents=True)
    provenance={'source_rollout_report_sha256':sha(Path(rollout_dir,'report.json')),
                'source_rollout_rows_sha256':sha(Path(rollout_dir,'rollouts.jsonl')),
                'source_pilot_sha256':sha(Path(one_step_dir,'pilot_states.json')),
                'analysis_code_sha256':sha(__file__),'bootstrap_samples':bootstrap,'bootstrap_seed':seed}
    (output/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (output/'value_labels.jsonl').open('w') as f:
        for z in labels:f.write(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n')
    lines=['# R3-SV2 按状态聚合的搜索重启诊断','',
           '正值 = 相对冻结C3更低的成本；区间按状态Bootstrap，仅供开发诊断。','']
    for metric,t in results_iter(report['metrics']):
        lines.extend(['## '+metric,'','| H | 方法 | n状态 | 平均改善s | 95%状态Bootstrap区间 | 符号翻转 | 重复随机数符号不稳 |','|---|---|---:|---:|---|---:|---:|'])
        for h,d in t.items():
            for arm,z in d.items():
                ci=z['state_bootstrap_95pct_sec'];sflip=z['sign_flip_fraction_per_state'];v=z['rng_sign_instability_fraction']
                fmt=lambda x:'—' if x is None else f'{100*x:.1f}%'
                lines.append(f"|{h}|{arm}|{z['n_states']}|{z['mean_h_gain_sec']:+.3f}|[{ci[0]:+.2f},{ci[1]:+.2f}]|{fmt(sflip)}|{fmt(v)}|")
        lines.append('')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return report

def results_iter(d):return d.items()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout-dir',required=True);p.add_argument('--one-step-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--bootstrap',type=int,default=4000)
    p.add_argument('--seed',type=int,default=20261024)
    a=p.parse_args();r=write_analysis(a.rollout_dir,a.one_step_dir,a.output_dir,a.bootstrap,a.seed)
    print('R3-SV2 analysis DONE',a.output_dir,'runs',r['n_runs'])
