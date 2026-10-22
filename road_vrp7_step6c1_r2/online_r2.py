#!/usr/bin/env python3
"""Actual-C3-aware learned Section selector inside frozen 30s ALNS.

A C3 action is computed on a CLONED RNG and cloned Selector stats. If the
model defers, ORIGINAL C3 is called with ORIGINAL RNG exactly once. If the
model overrides, C3's cloned RNG draws do not touch the solver RNG stream.
"""
from __future__ import annotations
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for f in ('road_vrp7_step6c1_r2','road_vrp7_step6b2_r','road_vrp7_step6b1_v2','road_vrp7_step6b2','road_vrp7_step6a','road_vrp7_step5b_v3','road_vrp7_stage3'):
    sys.path.insert(0,str(ROOT/f))
from r2_core import shadow_c3,transfer_shadow,matches
from r2_policy import R2Advantage
import benchmark_30s_r as base
from dataset import sha256
from section_dataset import select_actions
from section_v2 import Stats

ORIGINAL_LOADER=base.load_artifact
HANDLERS=[]
FORCE_DEFER=False

class R2Section(base.OnlineSectionR):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stats.update({'r2_shadow_calls':0,'r2_shadow_sec':0.,'r2_effective_overrides':0,
              'r2_same_as_C3':0,'r2_predicted_advantage_sum':0.,'r2_policy_rejected':0,
              'r2_c3_missing_or_invalid':0,'r2_score_budget_prevented':0})
        HANDLERS.append(self)

    def destroy(self,selector,mode,routes,rng):
        self.stats['calls']+=1
        if mode!='static' or self.obj.planner!='static':raise ValueError('R2 requires static-planner C3')
        if self.budget is not None and time.perf_counter()>=self.budget.deadline-.02:
            self.stats['budget_short_count']+=1
            return base.no_section(routes,self.obj)
        if self.stats['calls']%self.score_every:
            return self._fallback(selector,mode,routes,rng,'cadence')
        if self.scoring_elapsed>=self.max_scoring_sec:
            return self._fallback(selector,mode,routes,rng,'budget')
        begin=time.perf_counter();prior_fallback=self.fallback_elapsed
        shadow_payload=None
        def commit_defer(reason):
            # Once shadow has been drawn, do NOT re-run the expensive C3 scorer.
            # Commit exactly the cloned C3 RNG draws and stats, then return its
            # original Partial. This is the next real-C3 action at this state.
            _,partial,post_state,stat_delta=shadow_payload
            transfer_shadow(rng,selector,post_state,stat_delta,True)
            self.stats['r_c3_fallbacks']+=1
            if reason in ('budget','gate','cheap','cadence'):
                self.stats['r_'+reason+'_fallbacks']+=1
            return partial
        def exhausted():
            return (self.scoring_elapsed+time.perf_counter()-begin>=self.max_scoring_sec or
                (self.budget is not None and time.perf_counter()>=self.budget.deadline-.02))
        try:
            if exhausted():return self._fallback(selector,mode,routes,rng,'budget')
            # FROZEN C3 draw on a cloned RNG. The unchanged real RNG makes any
            # fallback exactly the original C3 draw from the same search state.
            tick=time.perf_counter()
            shadow_payload=shadow_c3(selector,mode,routes,rng,base.FROZEN_C3_DESTROY,Stats,True)
            baseline=shadow_payload[0]
            self.stats['r2_shadow_calls']+=1
            self.stats['r2_shadow_sec']+=time.perf_counter()-tick
            if baseline is None:
                self.stats['r2_c3_missing_or_invalid']+=1
                return commit_defer('cheap')
            if FORCE_DEFER:
                # Negative control: preserve the exact original C3 selected
                # action and RNG draws, but measure R2 shadow runtime.
                self.stats['r2_policy_rejected']+=1
                self.stats['gate_rejected']+=1
                return commit_defer('gate')
            if exhausted():
                self.stats['r2_score_budget_prevented']+=1
                return commit_defer('budget')
            candidates=selector.candidates(routes,traffic=False)
            c3=next((c for c in candidates if matches(c,baseline)),None)
            if c3 is None:
                self.stats['r2_c3_missing_or_invalid']+=1
                return commit_defer('cheap')
            sid=base.state_identifier(self.tid,self.scid,routes)
            sampled=select_actions(candidates,self.action_count,sid,self.sampler_seed)
            cost=float(self.obj.cost(routes))
            fx=[];cand=[]
            tick=time.perf_counter()
            bvec=self.form(routes,c3,cost)
            self.stats['feature_wall_sec']+=time.perf_counter()-tick
            if bvec is None:
                self.stats['r2_c3_missing_or_invalid']+=1
                return commit_defer('cheap')
            for _,c in sampled:
                if exhausted():
                    self.stats['r2_score_budget_prevented']+=1
                    return commit_defer('budget')
                t=time.perf_counter();v=self.form(routes,c,cost)
                self.stats['feature_wall_sec']+=time.perf_counter()-t
                if v is None:self.stats['no_proxy_count']+=1;continue
                fx.append(v);cand.append(c)
            if not cand:return commit_defer('cheap')
            if exhausted():return commit_defer('budget')
            same=np.array([matches(c,c3) for c in cand],bool)
            start=time.perf_counter()
            j,adv,reason=self.artifact['model'].select(np.stack(fx),bvec,same)
            self.stats['inference_wall_sec']+=time.perf_counter()-start
            self.stats['r_model_eval_calls']+=1
            self.stats['actions_scored']+=len(cand)
            if j is None:
                self.stats['r2_policy_rejected']+=1
                self.stats['gate_rejected']+=1
                return commit_defer('gate')
            selected=cand[j]
            if matches(selected,c3):
                self.stats['r2_same_as_C3']+=1
                return commit_defer('gate')
            # Preserve the same next RNG state as C3 for a tightly coupled
            # common-random-numbers comparison; but do NOT add C3 stats.
            transfer_shadow(rng,selector,shadow_payload[2],shadow_payload[3],False)
            self.stats['r_model_actions']+=1
            self.stats['r2_effective_overrides']+=1
            self.stats['r2_predicted_advantage_sum']+=float(adv)
            selector.stats.add('destroy_calls');selector.stats.add('candidate_count',len(candidates))
            return base.make_candidate_partial(routes,selected,self.obj)
        finally:
            elapsed=max(0.,time.perf_counter()-begin-(self.fallback_elapsed-prior_fallback))
            self.scoring_elapsed+=elapsed
            self.stats['r_scoring_calls']+=1
            self.stats['r_scoring_seconds']+=elapsed


def checked_load(path,mode,method,protocol,source):
    a=ORIGINAL_LOADER(path,mode,method,protocol,source)
    if a.get('stage')!='step6c1r2-c3-advantage-train-v1':raise ValueError('Wrong R2 model artifact')
    if not isinstance(a.get('model'),R2Advantage) or a['model'].mode!=mode:
        raise ValueError('Unexpected R2 predictor schema')
    if method!='P2' or protocol!=a['protocol']:raise ValueError('Wrong R2 protocol/method')
    labeldir=Path(a['r2_source'])
    if sha256(labeldir/'paired.npz')!=a['r2_dataset_sha256']:
        raise ValueError('R2 label artifact changed since training')
    return a


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--r2-force-defer',action='store_true')
    args,forward=p.parse_known_args()
    global FORCE_DEFER
    FORCE_DEFER=args.r2_force_defer
    if '--control' not in forward or forward[forward.index('--control')+1]!='learned':
        raise ValueError('Only learned arm handled by R2; use official B2-R for C3/M0/Random')
    old_loader,old_class,old_argv=base.load_artifact,base.OnlineSectionR,sys.argv
    base.load_artifact=checked_load;base.OnlineSectionR=R2Section
    HANDLERS.clear()
    try:
        sys.argv=['benchmark_30s_r.py','--output-dir',args.output_dir]+forward
        base.main()
    finally:
        base.load_artifact=old_loader;base.OnlineSectionR=old_class;sys.argv=old_argv
    out=Path(args.output_dir)
    import csv
    with (out/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(HANDLERS):raise AssertionError('R2 handler count does not match online records')
    keys=('r2_shadow_calls','r2_shadow_sec','r2_effective_overrides','r2_same_as_C3',
          'r2_predicted_advantage_sum','r2_policy_rejected','r2_c3_missing_or_invalid',
          'r2_score_budget_prevented')
    result=[]
    for row,handler in zip(rows,HANDLERS):
        result.append({'task_id':row['task_id'],'scenario_id':row['scenario_id'],'seed':int(row['seed']),
                       'road_cost_sec':float(row['ex_post_road_travel_sec']),
                       **{k:handler.stats[k] for k in keys}})
    modelpath=Path(forward[forward.index('--model-path')+1]);data={'stage':'step6c1r2-online-v1',
        'model_sha256':sha256(modelpath),'code_sha256':{'online_r2.py':sha256(Path(__file__)),
          'r2_policy.py':sha256(Path(__file__).with_name('r2_policy.py'))},
        'runs':result,'force_defer':FORCE_DEFER,'notes':['Cloned RNG/Stats shadow C3 is paired to real C3 fallback at the same state.',
         'Shadow draw and paired feature extraction are included INSIDE 30s wall budget.',
         'Online behavior changes the later ALNS trajectory; offline one-step proxy is not 30s gain.',
         'The scoring budget limits R2 shadow+feature+prediction CPU time; no cached RNG-dependent C3 actions.']}
    (out/'r2_diagnostics.json').write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
