"""No real Chicago inputs: strictly synthetic contract/negative-control tests."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vq4_core import (VQ_STAGE,VQ_REPORT,write_json,read_json,verify_nested,
                      reuse_checkpoints,create_views,validate_unit,sha256)
from analyze_curve import analyze


def ids(n):return [f'decision_{i:04d}' for i in range(n)]

def meta(n):
    return {'stage':VQ_STAGE,'selected_decision_ids':ids(n),'seed':20261025,
            'sampler_seed':20261019,'horizons':[20,50],'replicates':3,
            'frozen_solver_sha256':'frozen','label_sha256':'labels',
            'source_sha256':{'vq_core.py':'constant'},'train_tasks':['task_0000','task_0001','task_0002','task_0003']}

def row(di,h,rep,a):
    return {'decision_id':di,'horizon':h,'replicate':rep,'action_index':a,
            'final_road':float(3000+a*10+rep),'final_td_od':float(3100+a*10+rep),
            'one_step_gain_td_od':float(-a*10),'search_wall_sec':float(rep+0.01)}


def unit(di,h,rep):
    return {'decision_id':di,'horizon':h,'replicate':rep,'rows':[row(di,h,rep,a) for a in (-1,0,1)]}


def full_collection(folder,n,units=True):
    folder.mkdir(parents=True)
    write_json(folder/'manifest.json',meta(n))
    rows=[]
    for di in ids(n):
        for h in (20,50):
            for rep in range(3):
                u=unit(di,h,rep);rows.extend(u['rows'])
                if units:
                    (folder/'units').mkdir(exist_ok=True)
                    write_json(folder/'units'/f'{di}__H{h}__R{rep}.json',u)
    with (folder/'rollouts.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    write_json(folder/'report.json',{'stage':VQ_REPORT,'status':'COMPLETE',
        'n_selected_states':n,'horizons':[20,50],'replicates':3,
        'n_restart_action_rows':len(rows),'n_state_horizon_action_labels':n*2*3,
        'per_horizon_action_labels':{'20':n*3,'50':n*3}})


class CoreTests(unittest.TestCase):
    def test_nested_ok(self):
        self.assertTrue(verify_nested(meta(48),meta(192)))

    def test_bad_selection(self):
        m=meta(192);m['selected_decision_ids'][3],m['selected_decision_ids'][9]=m['selected_decision_ids'][9],m['selected_decision_ids'][3]
        with self.assertRaisesRegex(ValueError,'prefix'):verify_nested(meta(48),m)

    def test_changed_frozen_solver(self):
        m=meta(192);m['frozen_solver_sha256']='modified'
        with self.assertRaisesRegex(ValueError,'Different restart'):verify_nested(meta(48),m)

    def test_duplicate_or_absent_c3(self):
        v=unit('abc',20,0)
        validate_unit(v,'abc',20,0,3)
        v['rows']=[r for r in v['rows'] if r['action_index']!=-1]
        with self.assertRaisesRegex(ValueError,'C3'):validate_unit(v,'abc',20,0,3)

    def test_reuse_all_units_preserves_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old=base/'old';new=base/'new'
            full_collection(old,48)
            new.mkdir();write_json(new/'manifest.json',meta(192));(new/'units').mkdir()
            pre=unit(ids(48)[0],20,0)
            pre['rows'][0]['search_wall_sec']+=.1
            write_json(new/'units'/f'{ids(48)[0]}__H20__R0.json',pre)
            audit=reuse_checkpoints(old,new)
            self.assertEqual(audit['expected_units'],288)
            self.assertEqual(audit['copied_units'],287)
            self.assertEqual(audit['already_present_units'],1)
            self.assertEqual(len(audit['copies']),288)
            self.assertTrue((new/'vq4_reuse_audit.json').exists())
            with self.assertRaisesRegex(ValueError,'already finalized'):
                write_json(new/'report.json',{'status':'COMPLETE'});reuse_checkpoints(old,new)

    def test_reuse_rejects_cost_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old=base/'old';new=base/'new'
            full_collection(old,48)
            new.mkdir();write_json(new/'manifest.json',meta(192));(new/'units').mkdir()
            bad=unit(ids(48)[0],20,0);bad['rows'][0]['final_road']+=1.
            write_json(new/'units'/f'{ids(48)[0]}__H20__R0.json',bad)
            with self.assertRaisesRegex(ValueError,'different objective'):
                reuse_checkpoints(old,new)

    def test_views_complete_and_retain_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);src=base/'src';old=base/'old';dest=base/'views'
            full_collection(src,192,units=False);old.mkdir();write_json(old/'manifest.json',meta(48))
            a=create_views(src,dest,old48=old)
            self.assertEqual([z['size'] for z in a],[48,96,144,192])
            for n in (48,96,144,192):
                report=read_json(dest/f'vq4_view{n}_v1'/'report.json')
                self.assertEqual(report['n_restart_action_rows'],n*2*3*3)
                self.assertEqual(report['n_state_horizon_action_labels'],n*2*3)
            with self.assertRaises(FileExistsError):create_views(src,dest,old48=old)

    def test_views_mismatched_candidate_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);src=base/'src';full_collection(src,192,units=False)
            pp=src/'rollouts.jsonl';rs=[json.loads(s) for s in pp.read_text().splitlines()]
            rs=[r for r in rs if not(r['decision_id']==ids(192)[0] and r['replicate']==1 and r['action_index']==1)]
            pp.write_text('\n'.join(json.dumps(r) for r in rs)+'\n')
            with self.assertRaisesRegex(ValueError,'Candidate mismatch'):
                create_views(src,base/'views')


class CurveTests(unittest.TestCase):
    def test_analyze_multiseed_and_provenance_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);model=b/'models';code={'core':'abc'}
            policies=('C3','E0_FULL','E0_MATCHED','VQ_MEAN','RESID_OOF25','RESID_OOF50',
                      'PAIR_RANK','PAIR_REGRET','E0_RISK35','E0_SCORE_COVERAGE35')
            for n in (48,96):
                for seed in (42,43):
                    p=model/f'vq4_n{n}_seed{seed}_v1';p.mkdir(parents=True)
                    macros={};folds={}
                    for h in ('20','50'):
                        base={k:{'mean_gain_sec':float(n/48+(k!='C3')),
                           'mean_regret_sec':10.,'override_rate':.6,
                           'harm_gt40_per_state':.2,'harm_gt40_per_override':.3,'top1_hit':.3} for k in policies}
                        macros[h]=base
                        folds[h]={t:{'metrics':copy.deepcopy(base)} for t in ('task_0000','task_0001','task_0002','task_0003')}
                    write_json(p/'report.json',{'stage':'R3-VQ3-DECISION-RISK-LOTO-v1','status':'COMPLETE',
                        'seed':seed,'n_selected_states':n,'horizons':[20,50],
                        'policies':list(policies),'macro_task_metrics':macros,'folds':folds})
                    write_json(p/'provenance.json',{'collection_manifest_sha256':'view'+str(n),
                        'collection_rollouts_sha256':'rows'+str(n),
                        'dataset_paired_sha256':'same','dataset_groups_sha256':'same','code_sha256':code})
            res=analyze(model,b/'out',sizes=(48,96),seeds=(42,43))
            self.assertEqual(len(res),16)
            self.assertEqual(read_json(b/'out'/'report.json')['status'],'COMPLETE')
            self.assertTrue((b/'out'/'report.md').is_file())
            p=model/'vq4_n96_seed42_v1'/'provenance.json';v=read_json(p);v['code_sha256']['core']='evil';write_json(p,v)
            with self.assertRaisesRegex(ValueError,'Mixed VQ3 code'):
                analyze(model,b/'out2',sizes=(48,96),seeds=(42,43))

if __name__=='__main__':unittest.main()
