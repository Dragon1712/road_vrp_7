#!/usr/bin/env python3
"""Strict paired 30s R2 vs ORIGINAL C3 and strong B2-R M0 (descriptive only)."""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path


def load(path):
    folder=Path(path)
    with (folder/'metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    manifest=json.loads((folder/'manifest.json').read_text())
    if not rows or not len({(r['task_id'],r['scenario_id'],r['seed']) for r in rows})==len(rows):
        raise ValueError('Empty or duplicated online runs in '+str(folder))
    return manifest,rows


def paired(m0,c3,r2):
    studies={'M0':m0,'C3':c3,'R2':r2}
    man={n:load(p)[0] for n,p in studies.items()}
    rows={n:load(p)[1] for n,p in studies.items()}
    for n in ('C3','R2'):
        for field in ('source_sha256','road_sha256','edge_sha256','solver_sha256',
          'section_sha256','wall_sec','max_iterations','anneal_iterations','top_k',
          'seeds','task_ids','scenario_ids','split'):
            if man[n].get(field)!=man['M0'].get(field):
                raise ValueError(f'{n} differs from M0 in {field}')
    # C3 intentionally uses a DIFFERENT C3 top20pct sampler. Never pretend
    # to validate action-sampler equality with M0/R2.
    for field in ('action_count','sampler_seed','r_score_every','r_max_scoring_sec','r_fallback'):
        if man['M0'].get(field)!=man['R2'].get(field):raise ValueError('M0/R2 config mismatches '+field)
    def key(r):return (r['task_id'],r['scenario_id'],int(r['seed']))
    d={n:{key(r):r for r in arr} for n,arr in rows.items()}
    if not (set(d['M0'])==set(d['C3'])==set(d['R2'])):raise ValueError('R2/C3/M0 keys differ')
    detail=[]
    for k in sorted(d['M0']):
        values={n:float(d[n][k]['ex_post_road_travel_sec']) for n in studies}
        detail.append({'task_id':k[0],'scenario_id':k[1],'seed':k[2],**values,
            'r2_improvement_vs_c3':values['C3']-values['R2'],
            'r2_improvement_vs_m0':values['M0']-values['R2']})
    def summ(key):
        ds=[x[key] for x in detail]
        return {'n':len(ds),'mean_improvement_sec':statistics.mean(ds),
                'median_improvement_sec':statistics.median(ds),
                'win':sum(x>1e-8 for x in ds),'loss':sum(x< -1e-8 for x in ds),
                'tie':sum(abs(x)<=1e-8 for x in ds),'min_sec':min(ds),'max_sec':max(ds)}
    return {'stage':'step6c1r2-paired-analysis-v1','summary':{
        'vs_original_C3':summ('r2_improvement_vs_c3'),
        'vs_frozen_M0':summ('r2_improvement_vs_m0')},'cases':detail,
        'caveat':['Only paired instances; 8-case pilot cannot establish significance.',
             'Original C3 uses top20pct sampling; M0/R2 use stratified 8 candidate pool.',
             'Original M0 may use cache_size=64 while R2 deliberately disables caching of RNG-dependent C3 actions; not a cost-isolated test.',
             '30s search also stops at max_iterations=20000.',
             'Future ex_post_road_travel_sec is report-only and NEVER used online.',
             'R2 shadow C3 sampling changes runtime and uses copied RNG stats.']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--m0',default='runs/chicago_july_v1/step6b2_r/b2r_online_P2_F2_match8_v1')
    p.add_argument('--c3',required=True);p.add_argument('--r2',required=True)
    p.add_argument('--output-dir',required=True);a=p.parse_args()
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError(out)
    result=paired(a.m0,a.c3,a.r2);out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
