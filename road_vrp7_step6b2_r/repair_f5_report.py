#!/usr/bin/env python3
"""Non-destructive correction of frozen Step6-B2 v1 date/report.json F5 carryover."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def repair(study,out):
    study=Path(study);out=Path(out)
    if out.exists():raise FileExistsError(out)
    mani=json.loads((study/'manifest.json').read_text())
    if mani.get('stage')!='step6b2-ablation-v1':raise ValueError('Only original B2 v1 applies')
    taskf=study/'task/report.json';datef=study/'date/report.json'
    task=json.loads(taskf.read_text());date=json.loads(datef.read_text())
    f5=[k for k in date if k.endswith('_F5')]
    if not f5:raise ValueError('No F5 contamination to correct')
    if any(k not in task for k in f5):raise ValueError('Unknown F5 entry')
    for k in f5:
        # Strong proof: exact old task report entry carried over into date JSON.
        if task[k]!=date[k]:raise ValueError('F5 entry differs from task and needs manual audit')
    corrected={k:v for k,v in date.items() if k not in f5}
    manifest={'original_study':str(study),'status':'CORRECTED_REPORT_ONLY',
       'date_report_sha256':digest(datef),'task_report_sha256':digest(taskf),
       'removed_invalid_date_entries':f5,
       'warning':'All original files, models, labels and valid date F0-F4 metrics remain unchanged.'}
    out.mkdir(parents=True,exist_ok=False)
    (out/'date_report_corrected.json').write_text(json.dumps(corrected,indent=2,ensure_ascii=False)+'\n')
    (out/'audit.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();repair(a.study,a.output_dir)
