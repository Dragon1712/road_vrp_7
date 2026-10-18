#!/usr/bin/env python3
"""Offline synthetic tests: modules, patch anchors, snapshot analysis, calibration.
The genuine official-ALNS integration still requires server smoke.
"""
from __future__ import annotations
import importlib
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from snapshots_v3 import BestTrace,parse_times
from install_v3 import build,EXPECTED_RUNNER
from calibrate_cost import build as calibrate
from analyze_v3 import analyze
from repeat_compare import compare


def test_snapshots():
    trace=BestTrace(((1,2),(3,)),100.)
    ticks=iter([10.,12.,14.])
    with patch('snapshots_v3.time.perf_counter',side_effect=lambda:next(ticks)):
        trace.begin();trace.note_best(90.,((2,1),(3,)));trace.note_best(80.,((1,2),(3,)))
    snapshots=trace.snapshots((1.,3.,5.),search_elapsed=4.5)
    assert [x['status'] for x in snapshots]==['reached','reached','censored']
    assert [x['planned_cost'] for x in snapshots][:2]==[100.,90.]
    assert all(snapshots[-1][k] is None for k in ('routes','planned_cost'))
    assert parse_times('5,15,30')==(5.,15.,30.)
    for s in ('30,15','2,2','0,2','3,inf',''):
        try:parse_times(s);raise AssertionError('Must reject '+s)
        except ValueError:pass
    print('PASS snapshots: timestamping, censorship, monotonic best, invalid grids')


def test_selector():
    # Faithful TimeUCB shell; UCB policy inherited unchanged from v2.
    base=types.ModuleType('time_selector')
    class StubBase:
        def __init__(self,nd,nr,*,c,epsilon):
            self.nd=nd;self.nr=nr;self.c=c;self.epsilon=epsilon
            self.n=np.zeros((nd,nr),int);self.avg=np.zeros((nd,nr),float)
            self._next=None;self._candidate_gain=0.;self._current_cost=1.
            self.selection_start=0.;self._last_elapsed=0.;self.reward_sum=0.
        def diagnostics(self):return {'pair_counts':self.n.tolist(),'mean_reward':self.avg.tolist()}
    base.TimeUCB=StubBase
    with patch.dict(sys.modules,{'time_selector':base}):
        sys.modules.pop('selectors_v3',None)
        mod=importlib.import_module('selectors_v3')
        costs={f'{d}|{r}':.02+.01*i for i,d in enumerate(mod.NAMES_D) for r in mod.NAMES_R}
        profile={'pair_seconds':costs,'global_seconds':.035}
        nums=[]
        for mode in ('U0_gain','U1_walltime','U2_calibrated'):
            solver=mod.ControlledUCB(4,2,mode=mode,calibration=profile)
            solver._next=(0,0);solver._candidate_gain=.001;solver._current_cost=100.
            solver.selection_start=time.perf_counter()-.04
            solver.update(None,0,0,None)
            assert solver.n[0,0]==1
            assert solver.elapsed_sum[0,0]>.03
            assert solver.avg[0,0]>0
            assert solver.diagnostics()['mode']==mode
            nums.append(solver.avg[0,0])
        assert nums[2]!=nums[0]
        for bad in [{'pair_seconds':{}}, {'pair_seconds':costs,'global_seconds':-1}]:
            try:mod.ControlledUCB(4,2,mode='U2_calibrated',calibration=bad);raise AssertionError
            except ValueError:pass
        print('PASS UCB modes: common policy, calibrated positive divisor, telemetry')
    sys.modules.pop('selectors_v3',None)


def fake_sol(solver,task,scenario,seed,road=1000.,planned=900.):
    aud={'destroy':{d:{'seconds':.15,'calls':15} for d in ('random','worst','related','section')},
         'repair':{d:{'seconds':.1,'calls':15} for d in ('greedy','regret2')}}
    return {'task_id':task,'scenario_id':scenario,'solver':solver,'planner':'oracle','seed':seed,
            'routes':[[1,2]],'operator_audit':aud,'selector_stats':{'pair_counts':[[1,1]]*4,
                     'mean_reward':[[1,2]]*4,'measured_pair_elapsed_sec':[[.1,.2]]*4}}


def test_calibrate(tmp):
    source=tmp/'dev_sol.jsonl';dest=tmp/'profile.json'
    records=[fake_sol('C3_topk','task_0000','2025-07-22_0800',42)]
    source.write_text('\n'.join(map(json.dumps,records))+'\n')
    x=calibrate(source,dest)
    assert len(x['pair_seconds'])==8
    assert abs(x['pair_seconds']['section|greedy']-.15/15-.1/15)<1e-10
    assert x['global_seconds']>0
    assert calibrate(source,dest)==x
    print('PASS frozen cost calibration SHA, complete operator grid, no silent overwrite')


def synthetic_run(path,changed=False,missing=False):
    path.mkdir(parents=True)
    variants=['C0_baseline','C3_topk','U0_gain']
    tasks=['task_0000.json','task_0001.json']
    cfg={'stage':'step5b-v3','tasks':{name:'hash' for name in tasks},'scenarios':['2025-07-22_0800'],
         'variants':variants,'planners':['oracle'],'seeds':[42],
         'expected_count':6,'snapshot_times':[5.,15.,30.,60.],
         'max_iterations':20000,'anneal_iterations':5000}
    (path/'run_config.json').write_text(json.dumps(cfg))
    (path/'benchmark_summary.json').write_text(json.dumps({'status':'COMPLETE'}))
    import csv
    met=[];sol=[];snap=[]
    for taskname in tasks:
        for i,solver in enumerate(variants):
            key={'task_id':taskname.removesuffix('.json'),'scenario_id':'2025-07-22_0800',
                 'solver':solver,'planner':'oracle','seed':42}
            road=1000.+50*i+('task_0001' in taskname)*100
            if changed:road+=i*5
            met.append({**key,'ex_post_road_travel_sec':road,
                        'surrogate_travel_sec':road-50,'initial_surrogate_travel_sec':road+5,
                        'search_wall_sec':30.02,'iterations':1000})
            sol.append({**key,'selector_stats':{}})
            t=[]
            for s in cfg['snapshot_times']:
                if s==60.:
                    t.append({'target_sec':s,'status':'censored','planned_cost':None,'ex_post_road_travel_sec':None})
                else:
                    t.append({'target_sec':s,'status':'reached','planned_cost':road-s,
                              'ex_post_road_travel_sec':road-s+.5})
            snap.append({**key,'snapshots':t})
    with (path/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(met[0]));w.writeheader();w.writerows(met)
    if missing:snap.pop()
    (path/'solutions.jsonl').write_text('\n'.join(map(json.dumps,sol))+'\n')
    (path/'snapshot_records.jsonl').write_text('\n'.join(map(json.dumps,snap))+'\n')


def test_analyze(tmp):
    one=tmp/'one';two=tmp/'two';bad=tmp/'bad'
    synthetic_run(one);synthetic_run(two,changed=True);synthetic_run(bad,missing=True)
    a=analyze(one,one/'analysis')
    assert len(a['snapshot'])==3*4
    assert len(a['progress'])==3*3
    assert all(x['paired_reached']==0 for x in a['progress'] if x['to_sec']==60.)
    assert all(x['mean_planned_saved_sec']>=-1e-6 for x in a['progress'] if x['paired_reached'])
    rep=compare(one,two)
    assert rep['C0_baseline']['changed_road_records']==0
    assert rep['C3_topk']['changed_road_records']==2
    try:analyze(bad,bad/'analysis');raise AssertionError('Missing record accepted')
    except ValueError:pass
    print('PASS analysis: complete grid, same-run progress, censoring, repeated run diff, bad checkpoint rejection')


def test_installer():
    assert len(EXPECTED_RUNNER)==64
    src=HERE.parent/'road_vrp7_step5b_v2'/'run_experiment.py'
    if src.is_file():
        text=src.read_text()
        result=build(text)
        compile(result,str(HERE/'run_experiment.py'),'exec')
        assert 'snapshot_records.jsonl' in result
        assert 'anneal_iterations' in result
        print('PASS repository v2 -> v3 build and syntax compilation')
    else:
        import ast
        code=(HERE/'install_v3.py').read_text()
        tree=ast.parse(code)
        changes=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='change']
        anchors=[c.args[1].value for c in changes if len(c.args)>2 and isinstance(c.args[1],ast.Constant)]
        assert len(anchors)>18
        # Exercise all literal replacements against a synthetic anchor corpus.
        fake='\n'.join(anchors)
        build(fake)
        print('PASS all installer patch anchors (real frozen v2 source checked on user server)')


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d:
        tmp=Path(d)
        test_snapshots();test_selector();test_calibrate(tmp)
        test_analyze(tmp);test_installer()
    print('ALL SYNTHETIC TESTS PASSED; real ALNS-7 integration requires smoke.')
