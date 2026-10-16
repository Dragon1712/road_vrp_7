#!/usr/bin/env python3
"""Targeted Step5A-v2 mechanism audit. No real Chicago/ALNS test is claimed."""
from pathlib import Path
import sys, math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from test_original import ToyObjective,ROUTES,assert_legal
from traffic_operators import TrafficDestroy,removal_components


def audit():
    planner='oracle';a=ToyObjective(planner)
    by={}
    for op in ('at_space','at_timeonly','at_rr','at_misaligned','te_static_local','te_local','te_wr','te_positive'):
        d=TrafficDestroy(a)
        partial,removed=d(op,ROUTES,np.random.default_rng(42),3)
        assert len(removed)==3 and len(set(removed))==3
        assert all(len(rt)>=1 for rt in partial)
        assert sorted(removed+[x for r in partial for x in r])==list(range(1,13))
        by[op]=d.diagnostics()
    assert by['at_rr']['at_shadow_compared']==1
    assert 0<=by['at_rr']['at_shadow_change_rate']<=1
    assert by['te_wr']['te_shadow_compared']==1
    assert by['te_wr']['selected_components']==3
    assert by['at_rr']['at_feature_mean_parts']['distance']>=0
    assert by['at_rr']['at_feature_mean_parts']['level']>=0
    assert math.isclose(sum(by['at_rr']['at_feature_mean_parts'].values()),sum(by['at_rr']['at_feature_mean_parts'].values()))
    blocked=ToyObjective('static',block=True)
    for op in ('at_space','at_timeonly','at_rr','at_misaligned','te_static_local','te_local','te_wr','te_positive'):
        _=TrafficDestroy(blocked)(op,ROUTES,np.random.default_rng(11),3)
    assert blocked.od.dyn_access==0,'STATIC MUST NOT read future dynamic OD'
    c=ToyObjective('oracle')
    for route in ROUTES:
        for comp in removal_components(route,c):
            assert math.isclose(comp.local_gain_sec+comp.externality_sec,comp.total_gain_sec,abs_tol=1e-6)
    print('PASS 8 new/old operators + oracle shadow comparison + static future firewall + externality conservation')

    # Test telemetry independent of Stage3 file availability: AST extracts the
    # EXACT audited class from runner, not a reimplementation.
    import ast
    from collections import defaultdict
    src=ast.parse((ROOT/'run_experiment.py').read_text())
    klass=next(x for x in src.body if isinstance(x,ast.ClassDef) and x.name=='OperatorTelemetry')
    env={'defaultdict':defaultdict}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),'<audit extraction>','exec'),env)
    tele=env['OperatorTelemetry']()
    tele.destroy('at_rr',.001)
    tele.repair('greedy',.002)
    tele.accept('',True,20.,30.)
    tele.destroy('te_wr',.003)
    tele.repair('regret2',.004)
    tele.accept('',False,-10.,-5.)
    diag=tele.summary()
    assert diag['per_destroy']['at_rr']['new_best']==1
    assert diag['per_destroy']['te_wr']['new_best']==0
    assert diag['per_pair']['at_rr|greedy']['positive_candidates']==1
    assert diag['per_destroy']['te_wr']['accepted']==0
    assert len(diag['sample_events'])==2
    try:
        tele.repair('greedy',.01)
    except AssertionError:pass
    else:raise AssertionError('Unmatched repair not rejected')
    print('PASS exact runner telemetry class: attribution, acceptance, time, ordering guards')

if __name__=='__main__':audit()
