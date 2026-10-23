#!/usr/bin/env python3
"""Summarize repeated training seeds without treating them as independent datasets."""
import argparse,json,statistics
from pathlib import Path

def run(studies,out):
    data=[]
    for path in studies:
        p=Path(path);mf=json.loads((p/'manifest.json').read_text())
        summary=json.loads((p/'summary.json').read_text())
        if summary['status']!='COMPLETE':raise ValueError('Incomplete study '+str(p))
        rep=json.loads((p/'report.json').read_text())
        data.append((mf,rep))
    base=data[0][0]
    if len({x[0]['seed'] for x in data})!=len(data):raise ValueError('Repeated training seeds')
    for mf,r in data:
        for k in ('label_sha256','groups_sha256','protocol','folds','model_source','epoch_count','dataset_dir'):
            if mf[k]!=base[k]:raise ValueError('Different data/config across seeds '+k)
        if set(r)!=set(data[0][1]):raise ValueError('Different arm lists')
    arms={}
    for arm in data[0][1]:
        vals=[rep[arm]['scores']['internal_test'] for mf,rep in data]
        metrics=('mean_signed_advantage_sec','mean_regret_sec','top1_with_C3_accuracy',
                 'pairwise_accuracy','harm_per_state','override_rate')
        arms[arm]={k:{'mean':float(statistics.mean(v[k] for v in vals)),'min':float(min(v[k] for v in vals)),
                         'max':float(max(v[k] for v in vals))} for k in metrics if all(v[k] is not None for v in vals)}
        arms[arm]['seeds']=[mf['seed'] for mf,rep in data]
    result={'stage':'step6c1r3t-seed-aggregate-v1','same_data_labels_sha':base['label_sha256'],
      'n_training_seeds':len(data),'arms':arms,
      'note':'Seeds change fitting randomness, not independent task/data resampling. INTERNAL_TEST diagnostic only.'}
    output=Path(out)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('R3 aggregate complete',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();run(a.studies,a.output_dir)
