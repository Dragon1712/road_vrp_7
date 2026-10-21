#!/usr/bin/env python3
"""Six-seed offline aggregation; preserve independent-group and reward caveats."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np

def run(root,prefix,seeds,out):
    root,out=Path(root),Path(out)
    if out.exists():raise FileExistsError(out)
    rows=[];reference=None
    for seed in seeds:
        path=root/(prefix.format(seed=seed))
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest['seed']!=seed or manifest['stage']!='step6c1r-anchored-selective-defer-v1':
            raise ValueError('Wrong report seed/stage')
        fingerprint=(manifest['source_sha256'],manifest['F4_sha256'],manifest['graph_sha256'])
        if reference is None:reference=fingerprint
        if reference!=fingerprint:raise ValueError('Source/feature/graph SHA changed across seeds')
        for proto in manifest['protocols']:
            report=json.loads((path/proto/'report.json').read_text())
            m0=float(report['M0']['test_policy']['mean_signed_gain_sec'])
            for key,v in report.items():
                if not isinstance(v,dict) or 'test_M0_proxy_fallback' not in v:continue
                proxy=v['test_M0_proxy_fallback']
                rows.append({'protocol':proto,'seed':seed,'arm':key,
                    'zero_noop_gain':float(v['test_zero_noop']['mean_signed_gain_sec']),
                    'M0_proxy_gain':float(proxy['mean_signed_gain_sec_per_state']),
                    'M0_proxy_delta':float(proxy['mean_improvement_vs_m0_proxy_sec_per_state']),
                    'original_M0_gain':m0,'delegation_rate':proxy['delegation_rate']})
    by=defaultdict(list)
    for r in rows:by[(r['protocol'],r['arm'])].append(r)
    report=[]
    for (protocol,arm),rr in sorted(by.items()):
        gain=np.array([r['M0_proxy_delta'] for r in rr]);base=np.array([r['original_M0_gain'] for r in rr]);
        proxy=np.array([r['M0_proxy_gain'] for r in rr]);zero=np.array([r['zero_noop_gain'] for r in rr]);
        report.append({'protocol':protocol,'arm':arm,'seeds':len(rr),
         'mean_delta_vs_original_M0_proxy':float(gain.mean()),
         'positive_seed_count':int((gain>1e-7).sum()),'negative_seed_count':int((gain< -1e-7).sum()),
         'mean_original_M0_gain':float(base.mean()),'mean_selective_M0_proxy_gain':float(proxy.mean()),
         'mean_zero_noop_gain':float(zero.mean()),
         'mean_delegation_rate':float(np.mean([r['delegation_rate'] for r in rr])),
         'seedwise_deltas':gain.tolist()})
    out.mkdir(parents=True,exist_ok=False)
    payload={'stage':'step6c1r-aggregate-v1','items':report,
     'warnings':['M0-proxy fallback is not actual online C3 fallback.',
       'July is repeatedly used development data, NOT a blind test.',
       'Repeated train seeds share the same test state set; do not treat seed count as independent sample size.',
       'Paired online Road.replay is required before claiming an ALNS benefit.']}
    (out/'report.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/chicago_july_v1/step6c1_r');p.add_argument('--prefix',default='c1r_seed{seed}_v1')
    p.add_argument('--seeds',default='42,43,44,45,46,47');p.add_argument('--output-dir',required=True);a=p.parse_args()
    run(a.root,a.prefix,[int(x) for x in a.seeds.split(',')],a.output_dir)
