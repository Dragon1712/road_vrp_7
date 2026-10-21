#!/usr/bin/env python3
"""30-second OFFICIAL ALNS/C3 static-planner adapter, cold context separately timed.

Core original Step5-B v3 solver is NOT edited. The learned treatment patches
only the per-instance Section destroy dispatch for the duration of one solve;
all other destroy/repair operators, roulette selection, SA, and stop rule remain
identical. Feature+prediction+static proxy cost IS measured inside 30s search.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,time
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import joblib
from features import (ROOT,FEATURE_MODES,CORE,OLD_HIST,GEOMETRY,TRAFFIC,RELIABILITY,
                      core_online,old_hist_online,extract_boundary)
from dataset import sha256
from section_dataset import HistoricalCorridorFeatures,read_raw_cost,select_actions
from road_vrp import Road,OD,Objective,legal,read_json
from benchmark import load_tasks
import section_v2
import run_experiment as v3


def state_identifier(task_id,scenario_id,routes):
    return hashlib.sha256(json.dumps([task_id,scenario_id,tuple(tuple(r) for r in routes)],sort_keys=True).encode()).hexdigest()[:24]

def make_candidate_partial(routes,c,obj):
    r,p,n=int(c['route']),int(c['pos']),int(c['size'])
    fr=section_v2.frozen(routes)
    block=section_v2.Block(tuple(fr[r][p:p+n]),r,p)
    after=section_v2.mutable(fr);del after[r][p:p+n]
    if not after[r]:raise ValueError('Cannot empty route')
    return section_v2.Partial(section_v2.frozen(after),obj,(block,),fr)


def no_section(routes,obj):
    fr=section_v2.frozen(routes)
    return section_v2.Partial(fr,obj,(),fr)

class OnlineSection:
    def __init__(self,artifact,obj,hist,task_id,scenario_id,departure_local,*,action_count=8,
                 sampler_seed=20261019,control=False):
        self.artifact=artifact;self.obj=obj;self.hist=hist;self.tid=task_id;self.scid=scenario_id
        from pandas import Timestamp
        t=Timestamp(departure_local);self.hour=t.hour+t.minute/60.;self.weekday=t.weekday()
        self.action_count=action_count;self.sampler_seed=sampler_seed;self.control=control
        self.stats={'calls':0,'no_section':0,'gate_rejected':0,'feature_wall_sec':0.,
                    'inference_wall_sec':0.,'no_proxy_count':0,'actions_scored':0,
                    'budget_short_count':0,'selected_score_sum':0.}
        self.budget=None
    def form(self,routes,c,base_cost):
        mode=self.artifact['mode'] if self.artifact else 'F0'
        x=core_online(routes,c,self.obj,self.hour,self.weekday,base_cost)
        if mode=='F1':
            if self.hist is None:raise ValueError('Missing historical input')
            x.update(old_hist_online(routes,c,self.obj,self.hist))
        elif mode in ('F2','F3','F4'):
            extras=extract_boundary(routes,c,self.obj,self.hist if mode in ('F3','F4') else None)
            if extras is None:return None
            x.update(extras)
        names=FEATURE_MODES[mode]
        values=np.array([x[name] for name in names],float)
        if len(values)!=len(names) or not np.isfinite(values).all():raise ValueError('Bad online features')
        return values
    def destroy(self,selector,mode,routes,rng):
        self.stats['calls']+=1
        if mode!='static' or self.obj.planner!='static':raise ValueError('Only static-planner C3 model is supported')
        if self.budget is not None and time.perf_counter()>=self.budget.deadline-0.01:
            self.stats['budget_short_count']+=1
            return no_section(routes,self.obj)
        tick=time.perf_counter();cs=selector.candidates(routes,traffic=False)
        selector.stats.add('destroy_calls')
        selector.stats.add('candidate_count',len(cs))
        if not cs:
            selector.stats.add('destroy_sec',time.perf_counter()-tick)
            self.stats['no_section']+=1
            return no_section(routes,self.obj)
        sid=state_identifier(self.tid,self.scid,routes)
        options=select_actions(cs,self.action_count,sid,self.sampler_seed)
        indices=[];vectors=[]
        t_cost=time.perf_counter()
        base_cost=float(self.obj.cost(routes))
        self.stats['feature_wall_sec']+=time.perf_counter()-t_cost
        for j,c in options:
            t=time.perf_counter()
            vals=self.form(routes,c,base_cost)
            self.stats['feature_wall_sec']+=time.perf_counter()-t
            if vals is None:self.stats['no_proxy_count']+=1;continue
            indices.append(j);vectors.append(vals)
            if self.budget is not None and time.perf_counter()>=self.budget.deadline-0.01:break
        if not indices:
            selector.stats.add('destroy_sec',time.perf_counter()-tick)
            self.stats['no_section']+=1
            return no_section(routes,self.obj)
        self.stats['actions_scored']+=len(indices)
        if self.control:
            local=int(rng.integers(len(indices)));gate=float('inf');rank=0.
        else:
            t=time.perf_counter()
            scores,gates,_=self.artifact['model'].predict(np.stack(vectors))
            self.stats['inference_wall_sec']+=time.perf_counter()-t
            local=int(np.argmax(scores));gate=float(gates[local]);rank=float(scores[local])
            threshold=self.artifact['threshold']
            if threshold is None or gate<float(threshold):
                self.stats['gate_rejected']+=1
                selector.stats.add('destroy_sec',time.perf_counter()-tick)
                return no_section(routes,self.obj)
        selected=cs[indices[local]]
        self.stats['selected_score_sum']+=rank
        # A true actionable Section in the original frozen C3 Repair pipeline.
        selector.stats.add('selected_support_sum',0.)
        selector.stats.add('selected_support_n')
        selector.stats.add('destroy_sec',time.perf_counter()-tick)
        return make_candidate_partial(routes,selected,self.obj)

@contextmanager
def temporary_hook(handler):
    """Restore both class methods after each solve; never change frozen files."""
    cls=v3.NewSelector
    if cls is not section_v2.Selector:raise RuntimeError('Frozen solver selector class mismatch')
    original=cls.destroy;original_budget=v3.BothBudget
    def hook(self,mode,routes,rng):return handler.destroy(self,mode,routes,rng)
    class TrackingBudget(original_budget):
        def __init__(self,iters,wall):
            super().__init__(iters,wall);handler.budget=self
    cls.destroy=hook;v3.BothBudget=TrackingBudget
    try:yield
    finally:
        cls.destroy=original;v3.BothBudget=original_budget


def load_artifact(path,mode,method,protocol,source):
    d=joblib.load(path) # ONLY load artifacts produced and trusted by the local training run.
    if d.get('mode')!=mode or d.get('method')!=method or d.get('protocol')!=protocol:
        raise ValueError('Model identity/protocol mismatch')
    if not d.get('online_compatible') or mode=='F5':raise ValueError('F5 shuffled negative control cannot deploy online')
    if d.get('feature_names')!=list(FEATURE_MODES[mode]):raise ValueError('Model feature schema mismatch')
    if d.get('training_label')!='static_repair_td_gain_sec':raise ValueError('Wrong model label')
    data=d.get('source_sha256',{})
    for k,v in data.items():
        if k in {'config','schema','X','groups','y','summary'}:
            source_file={'config':'run_config.json','schema':'analysis/dataset_schema.json',
              'X':'analysis/features_X.npz','y':'analysis/labels_y.npz',
              'groups':'analysis/groups.csv','summary':'benchmark_summary.json'}[k]
            if sha256(Path(source)/source_file)!=v:raise ValueError('Train/source mismatch '+k)
    return d

def verify_source_hashes(road,task_manifest,model_source):
    c=read_json(Path(model_source)/'run_config.json')
    if c['source_cost_hash']!=road.cost_file_hash or c['source_edge_hash']!=road.edge_file_hash:
        raise ValueError('Model and road cost manifest mismatch')
    if task_manifest['cost_edge_index_sha256']!=road.edge_file_hash or task_manifest['cost_edge_npz_sha256']!=road.cost_file_hash:
        raise ValueError('Road/task manifest mismatch')

def run(args):
    v3.check_alns()
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError('New study directory required: '+str(out))
    if args.wall_sec!=30:raise ValueError('Fixed 30s study requires --wall-sec 30')
    if args.model_mode=='F5':raise ValueError('F5 cannot be deployed')
    if args.model_method not in ('P2','P4'):raise ValueError('Only P2 and P4 supported')
    if args.max_iterations<=0 or args.anneal_iterations<=0 or args.top_k<1:raise ValueError('Bad solver settings')
    if args.control!='C3' and args.control!='sampled_random' and args.control!='learned':
        raise ValueError('Invalid arm')
    trained=None
    if args.control=='learned':
        trained=load_artifact(args.model_path,args.model_mode,args.model_method,args.model_protocol,args.model_source)
    road=Road(args.cost_dir)
    paths,tasks,manifest=load_tasks(args.task_dir)
    # Baseline has same frozen model-source road/input distribution requirement.
    verify_source_hashes(road,manifest,args.model_source)
    scenarios=[s for s in tasks[0]['scenarios'] if s['split']==args.split and s['eligible']]
    if args.max_tasks:tasks=tasks[:args.max_tasks]
    if args.max_scenarios:scenarios=scenarios[:args.max_scenarios]
    if not scenarios:raise ValueError('No eligible scenarios')
    if args.task_ids:
        wanted=set(args.task_ids.split(','));tasks=[t for t in tasks if t['task_id'] in wanted]
        if set(t['task_id'] for t in tasks)!=wanted:raise ValueError('Unknown task IDs')
    if args.scenario_ids:
        wanted=set(args.scenario_ids.split(','));scenarios=[s for s in scenarios if s['scenario_id'] in wanted]
        if set(s['scenario_id'] for s in scenarios)!=wanted:raise ValueError('Unknown scenario IDs')
    if not tasks or not scenarios:raise ValueError('Empty evaluation')
    for task in tasks:
        others={s['scenario_id']:s for s in task['scenarios']}
        if any(others.get(s['scenario_id'])!=s for s in scenarios):raise ValueError('Scenario mismatch across tasks')
    seeds=[int(s) for s in args.seeds.split(',')]
    if len(set(seeds))!=len(seeds):raise ValueError('Repeated seed')
    src_cfg=read_json(Path(args.model_source)/'run_config.json')
    if int(src_cfg['lookback_slots'])!=6:raise ValueError('Expected history 6 slots')
    pin=src_cfg.get('code_sha256',{})
    for relative in ('road_vrp7_step5b_v3/section_v2.py','road_vrp7_stage3/road_vrp.py'):
        if relative not in pin or sha256(ROOT/relative)!=pin[relative]:
            raise ValueError('Frozen Step6-A source implementation changed: '+relative)
    raw=None;hist=None;history_setup_sec=0.
    if args.control!='C3' and args.model_mode in ('F1','F3','F4'):
        t_setup=time.perf_counter()
        raw=read_raw_cost(args.cost_dir)
        hist=HistoricalCorridorFeatures(road,raw,6)
        history_setup_sec=time.perf_counter()-t_setup
    output_manifest={'stage':'step6b2-30s-pilot-v1','arm':args.control,'model_method':args.model_method,
      'model_mode':args.model_mode,'model_protocol':args.model_protocol,
      'model_sha256':sha256(args.model_path) if trained else None,
      'source_sha256':sha256(Path(args.model_source)/'run_config.json'),
      'road_sha256':road.cost_file_hash,'edge_sha256':road.edge_file_hash,
      'solver_sha256':sha256(ROOT/'road_vrp7_step5b_v3/run_experiment.py'),
      'adapter_sha256':sha256(Path(__file__)),
      'feature_code_sha256':sha256(ROOT/'road_vrp7_step6b2/features.py'),
      'model_code_sha256':sha256(ROOT/'road_vrp7_step6b1_v2/models.py'),
      'section_sha256':sha256(ROOT/'road_vrp7_step5b_v3/section_v2.py'),
      'history_setup_sec':history_setup_sec,
      'seeds':seeds,'task_ids':[x['task_id'] for x in tasks],
      'scenario_ids':[x['scenario_id'] for x in scenarios],
      'split':args.split,'wall_sec':args.wall_sec,'max_iterations':args.max_iterations,
      'anneal_iterations':args.anneal_iterations,'top_k':args.top_k,
      'action_count':args.action_count,'sampler_seed':args.sampler_seed,
      'notes':['OD and historical loading time excluded from 30-second SEARCH budget and reported separately.',
               'Model scoring+features inside 30-second C3 search stop rule.',
               'Road replay occurs after search and cannot influence operators.',
               'Sampled_random and learned use stratified 8-section Section sampler; C3 uses its original top20pct sampler.',
               'Global historic tensor read/preparation is outside search and explicitly in manifest history_setup_sec.']}
    out.mkdir(parents=True,exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(output_manifest,indent=2,ensure_ascii=False)+'\n')
    cols=['task_id','scenario_id','date','seed','arm','model_method','model_mode',
          'planned_cost_sec','ex_post_road_travel_sec','ex_post_total_vehicle_elapsed_sec',
          'search_wall_sec','precompute_wall_sec','iterations','termination_reason',
          'section_calls','gate_rejected','section_no_proxy','section_feature_wall_sec',
          'section_inference_wall_sec','section_actions_scored','section_budget_short_count',
          'observed_edge_share']
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        solutions=[]
        for task in tasks:
            for sc in scenarios:
                t=time.perf_counter()
                od=OD(road,task,sc,step_min=int(src_cfg['od_step_min']),hours=float(src_cfg['od_hours']),
                      cache_dir=args.od_cache_dir)
                if hist is not None:hist.enter(task['task_id'],od,sc['departure_local'])
                precompute=time.perf_counter()-t
                for seed in seeds:
                    obj=Objective(task,od,'static')
                    h=OnlineSection(trained,obj,hist,task['task_id'],sc['scenario_id'],sc['departure_local'],
                                    action_count=args.action_count,sampler_seed=args.sampler_seed,
                                    control=args.control=='sampled_random') if args.control!='C3' else None
                    if h is None:
                        answer=v3.solve('C3_topk',task,obj,seed,30,args.max_iterations,
                            top_k=args.top_k,anneal_iterations=args.anneal_iterations)
                    else:
                        with temporary_hook(h):
                            answer=v3.solve('C3_topk',task,obj,seed,30,args.max_iterations,
                                top_k=args.top_k,anneal_iterations=args.anneal_iterations)
                    legal(task,answer['routes'])
                    # Post-hoc replay ONLY; never in learned operator selection.
                    actual=road.replay(task,answer['routes'],sc['departure_local'])
                    stats=h.stats if h is not None else {}
                    row=dict(task_id=task['task_id'],scenario_id=sc['scenario_id'],date=sc['date'],seed=seed,
                        arm=args.control,model_method=args.model_method if h and not h.control else '',
                        model_mode=args.model_mode if h and not h.control else '',
                        planned_cost_sec=answer['planned_cost'],
                        ex_post_road_travel_sec=actual['ex_post_road_travel_sec'],
                        ex_post_total_vehicle_elapsed_sec=actual['ex_post_total_vehicle_elapsed_sec'],
                        observed_edge_share=actual['observed_edge_share'],
                        search_wall_sec=answer['search_wall_sec'],precompute_wall_sec=precompute,
                        iterations=answer['iterations'],termination_reason=answer['termination_reason'],
                        section_calls=answer['stats']['destroy_calls'],
                        gate_rejected=stats.get('gate_rejected',0),section_no_proxy=stats.get('no_proxy_count',0),
                        section_feature_wall_sec=stats.get('feature_wall_sec',0.),
                        section_inference_wall_sec=stats.get('inference_wall_sec',0.),
                        section_actions_scored=stats.get('actions_scored',0),
                        section_budget_short_count=stats.get('budget_short_count',0))
                    writer.writerow(row);f.flush()
                    solutions.append({**row,'routes':answer['routes']})
                    print('SEARCH',args.control,task['task_id'],sc['scenario_id'],seed,
                          'cost',round(row['ex_post_road_travel_sec'],2),'wall',round(row['search_wall_sec'],2),flush=True)
    (out/'solutions.json').write_text(json.dumps(solutions,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','records':len(solutions),
      'expected_records':len(tasks)*len(scenarios)*len(seeds)},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--control',choices=('C3','sampled_random','learned'),required=True)
    p.add_argument('--model-path')
    p.add_argument('--model-mode',choices=('F0','F1','F2','F3','F4','F5'),default='F0')
    p.add_argument('--model-method',choices=('P2','P4'),default='P2')
    p.add_argument('--model-protocol',choices=('task','date'),default='task')
    p.add_argument('--model-source',default='runs/chicago_july_v1/step6a_v2/labeled_current_static_full_v1')
    p.add_argument('--cost-dir',default='datasets/processed/chicago/july_v1/dynamic_costs')
    p.add_argument('--task-dir',default='datasets/processed/chicago/july_v1/tdcvrp30/tasks_10')
    p.add_argument('--od-cache-dir',default='runs/chicago_july_v1/stage3/od_cache')
    p.add_argument('--split',choices=('train','validation','test'),default='validation')
    p.add_argument('--task-ids',default='');p.add_argument('--scenario-ids',default='')
    p.add_argument('--max-tasks',type=int);p.add_argument('--max-scenarios',type=int)
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--wall-sec',type=int,default=30);p.add_argument('--max-iterations',type=int,default=20000)
    p.add_argument('--anneal-iterations',type=int,default=5000);p.add_argument('--top-k',type=int,default=12)
    p.add_argument('--action-count',type=int,default=8);p.add_argument('--sampler-seed',type=int,default=20261019)
    a=p.parse_args()
    if a.control=='learned' and not a.model_path:p.error('learned requires --model-path')
    if a.action_count<2:p.error('need at least 2 candidates per Section')
    run(a)
if __name__=='__main__':main()
