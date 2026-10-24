#!/usr/bin/env python3
"""VQ4 collector checkpoint reuse, nested 48/96/144/192 views, and preflight."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from vq4_core import (SIZES,read_json,verify_nested,reuse_checkpoints,create_views)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('preflight');q.add_argument('--old48',required=True);q.add_argument('--target192',required=True)
    q=sub.add_parser('reuse');q.add_argument('--old48',required=True);q.add_argument('--target192',required=True)
    q=sub.add_parser('views');q.add_argument('--old48',required=True);q.add_argument('--target192',required=True);q.add_argument('--out-root',required=True)
    args=p.parse_args()
    old=Path(args.old48);target=Path(args.target192)
    if args.command=='preflight':
        a=read_json(old/'manifest.json');b=read_json(target/'manifest.json')
        verify_nested(a,b,len(a['selected_decision_ids']),len(b['selected_decision_ids']))
        print(json.dumps({'status':'OK','old_states':len(a['selected_decision_ids']),
                          'target_states':len(b['selected_decision_ids']),
                          'horizons':b['horizons'],'reps':b['replicates'],
                          'safe_reuse':True},ensure_ascii=False,indent=2))
    elif args.command=='reuse':
        z=reuse_checkpoints(old,target)
        print(json.dumps({k:v for k,v in z.items() if k!='copies'},ensure_ascii=False,indent=2))
    elif args.command=='views':
        z=create_views(target,Path(args.out_root),old48=old,sizes=SIZES)
        print(json.dumps({'status':'COMPLETE','views':z},indent=2))

if __name__=='__main__':main()
