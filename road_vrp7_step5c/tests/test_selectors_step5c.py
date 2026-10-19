#!/usr/bin/env python3
"""Offline tests, not a substitute for official ALNS 7.0.0 Smoke."""
from __future__ import annotations
import ast
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from selectors_step5c import FrozenPairLRW,PAIRS
from learn_lrw import train
from install_step5c import build_runner,build_analyzer,replace_one


def test_lrw():
    with tempfile.TemporaryDirectory() as d:
        inp=Path(d)/'input.jsonl'
        item={'solver':'C3_topk','planner':'oracle','task_id':'task_0000',
              'scenario_id':'2025-07-22_0800','operator_audit':{'pairs':{
                  p:{'calls':20,'improving':6,'new_best':2} for p in PAIRS}}}
        inp.write_text(json.dumps(item)+'\n')
        profile=train(inp)
        assert profile['trained_records']==1
        lrw=FrozenPairLRW(profile)
        assert abs(lrw.probabilities.sum()-1)<1e-12
        rng=np.random.default_rng(42)
        for _ in range(100):
            d,r=lrw(rng,None,None)
            lrw.update(None,d,r,None)
        assert lrw.pair_counts.sum()==100
        profile['weights'][PAIRS[0]]=0
        try:FrozenPairLRW(profile)
        except ValueError:pass
        else:raise AssertionError('Should reject zero weights')


def _fixture_from_literal_calls(fn):
    tree=ast.parse(Path(HERE/'install_step5c.py').read_text(encoding='utf-8'))
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
    olds=[]
    aliases={}
    for node in ast.walk(f):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try: aliases[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    for node in ast.walk(f):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='replace_one':
            x=node.args[1]
            olds.append(aliases[x.id] if isinstance(x,ast.Name) else ast.literal_eval(x))
    return '\n\n'.join(olds)


def test_patcher():
    r=build_runner(_fixture_from_literal_calls('build_runner'))
    assert 'C9_lrw' in r
    assert "cfg={'stage':'step5c-v1'" in r
    assert 'selectors_step5c.py' in r
    assert 'lrw_profile_sha256' in r
    a=build_analyzer(_fixture_from_literal_calls('build_analyzer'))
    assert 'step5c-v1' in a
    assert "('C9_lrw','C3_topk')" in a
    try:replace_one('abc','DEF','x')
    except ValueError:pass
    else:raise AssertionError('Missing anchors must fail closed')


if __name__=='__main__':
    test_lrw();test_patcher()
    print('PASS LRW data, selector, provenance and patch anchors')
