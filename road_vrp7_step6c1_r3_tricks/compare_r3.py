#!/usr/bin/env python3
"""Pairwise development road replay summary; includes cluster bootstrap intervals."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from r3_stats import paired_stats

def read(folder):
    folder=Path(folder)
    plan=json.loads((folder/'plan.json').read_text())
    summ=json.loads((folder/'summary.json').read_text())
    if summ.get('status')!='COMPLETE':raise ValueError('Incomplete study '+str(folder))
    with (folder/'metrics.csv').open(newline='') as h:rows=list(csv.DictReader(h))
    pairs={(r['task_id'],r['scenario_id'],int(r['seed'])):r for r in rows}
    if len(pairs)!=len(rows) or len(rows)!=summ['records']:raise ValueError('Bad study '+str(folder))
    return plan,pairs

def main(a):
    paths={}
    for raw in a.studies:
        if '=' not in raw:raise ValueError('Expected ARM=PATH '+raw)
        k,v=raw.split('=',1)
        if k in paths:raise ValueError('Repeated arm '+k)
        paths[k]=v
    if 'M0' not in paths or 'C3' not in paths:raise ValueError('Need C3 and M0 reference')
    data={k:read(v) for k,v in paths.items()}
    c=data['C3'][0]
    for arm,(p,rows) in data.items():
        for key in ('tasks','scenarios','seeds','source_sha256','frozen_solver_sha256',
            'frozen_selector_sha256','wall_sec','max_iterations','top_k','split'):
            if p.get(key)!=c.get(key):raise ValueError('Experiment design mismatch '+arm+' '+key)
    res={'stage':'step6c1r3t-paired-pilot-v1','inputs':paths,'comparisons':{},
      'warning':'Development / nonblind, shared road situations and only two tasks; CI descriptive only.'}
    for arm,(_,rows) in data.items():
        if arm in ('C3','M0'):continue
        res['comparisons'][arm]={baseline:paired_stats(data[baseline][1],rows)
             for baseline in ('C3','M0')}
    output=Path(a.output_dir)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
    md=['# R3-T 第1批固定30秒开发集配对对照','','正收益=道路成本降低。','','|方法|参考|平均改善(秒)|胜/负/平|簇Bootstrap区间|','|---|---|---:|---|---|']
    for arm,rr in res['comparisons'].items():
        for b,v in rr.items():
            md.append(f'|{arm}|{b}|{v["mean_improvement_sec"]:+.3f}|{v["win"]}/{v["loss"]}/{v["tie"]}|{v["cluster_bootstrap_95pct_interval_sec"]}|')
    (output/'report.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True)
    p.add_argument('--output-dir',required=True);main(p.parse_args())
