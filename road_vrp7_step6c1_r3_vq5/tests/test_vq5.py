"""VQ5 deterministic fixed-panel and original-VQ3 baseline regression tests."""
from __future__ import annotations
import json,sys,tempfile,unittest
from argparse import Namespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for d in (ROOT/'road_vrp7_step6c1_r3_vq5',ROOT/'road_vrp7_step6c1_r3_vq3',
          ROOT/'road_vrp7_step6c1_r3_vq2/tests',ROOT/'road_vrp7_step6c1_r3_vq2',
          ROOT/'road_vrp7_step6c1_r3_vq',ROOT/'road_vrp7_step6c1_r2'):
    if str(d) not in sys.path:sys.path.insert(0,str(d))
from vq5_core import (check_balanced_prefix,hybrid_outer_selection,
                     same_decision_rows,anchor_filter,read_json,write_json,STAGE)
import fixed_panel_vq5,summarize_vq5,train_vq3

class SelectionTests(unittest.TestCase):
    def setup_selection(self):
        groups=[];ids=[]
        for i in range(12):
            t=i%4;groups.append({'decision_id':str(i),'task_id':f't{t}'})
            ids.append(str(i))
        return groups,ids,{str(i):i for i in range(len(groups))}, {f't{x}':'train' for x in range(4)}
    def test_balanced_nested_train_and_fixed_eval(self):
        groups,ids,lookup,roles=self.setup_selection()
        self.assertEqual(check_balanced_prefix(ids,groups,lookup,roles,[8,12]),list(roles))
        hy,tr,ev=hybrid_outer_selection(ids,ids[:8],groups,lookup,'t0')
        self.assertEqual(len(tr),6);self.assertEqual(len(ev),3)
        self.assertEqual(set(ev),{'0','4','8'})
        self.assertEqual(set(tr)&set(ev),set())
        self.assertEqual(len(hy),9)
    def test_wrong_prefix_and_forbidden_role_rejected(self):
        groups,ids,lookup,roles=self.setup_selection()
        with self.assertRaises(ValueError):check_balanced_prefix(ids,groups,lookup,roles,[7,12])
        roles['t2']='internal_test'
        with self.assertRaises(ValueError):check_balanced_prefix(ids,groups,lookup,roles,[8,12])
        roles['t2']='train'
        ids[0]=ids[1]
        with self.assertRaises(ValueError):check_balanced_prefix(ids,groups,lookup,roles,[8,12])
    def test_anchor_detects_action_or_gain_change(self):
        old=[{'decision_id':'a','horizon':20,'policy':'P','selected_index':0,'realized_mean_road_gain':2.,'regret':7.}]
        newer=[dict(old[0])]
        self.assertEqual(same_decision_rows(newer,old),1)
        newer[0]['selected_index']=1
        with self.assertRaises(ValueError):same_decision_rows(newer,old)
        newer[0]=dict(old[0]);newer[0]['realized_mean_road_gain']=22
        with self.assertRaises(ValueError):same_decision_rows(newer,old)
    def test_no_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                fixed_panel_vq5.run(Namespace(output_dir=tmp))

class SyntheticEndToEnd(unittest.TestCase):
    def test_fixed_panel_and_frozen_anchor(self):
        from test_vq2 import SyntheticPipeline
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);src,col,selection=SyntheticPipeline()._make_data(home)
            # Synthetic selector from fixture was grouped by task; freeze a balanced round-robin order.
            bins={t:[di for di in selection if di.startswith(f'd{t:02d}_')]
                  for t in range(4)}
            balanced=[bins[t][j] for j in range(4) for t in range(4)]
            mf=read_json(col/'manifest.json');mf['selected_decision_ids']=balanced
            # Run one H for a fast end-to-end integration, still 3 restarts per action.
            mf['horizons']=[20];write_json(col/'manifest.json',mf)
            rows=[json.loads(x) for x in (col/'rollouts.jsonl').read_text().splitlines() if x.strip()]
            (col/'rollouts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows if x['horizon']==20))
            report=read_json(col/'report.json');report['n_restart_action_rows']=len(rows)//2
            write_json(col/'report.json',report)
            # Baseline n8 for strict action-level reproduction using unchanged VQ3 driver.
            from shutil import copytree
            col8=home/'collection8';copytree(col,col8)
            m8=read_json(col8/'manifest.json');m8['selected_decision_ids']=balanced[:8]
            write_json(col8/'manifest.json',m8)
            keep=set(balanced[:8]); rr=[json.loads(x) for x in (col8/'rollouts.jsonl').read_text().splitlines() if x.strip()]
            rr=[x for x in rr if x['decision_id'] in keep]
            (col8/'rollouts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
            z=read_json(col8/'report.json');z['n_restart_action_rows']=len(rr)
            write_json(col8/'report.json',z)
            orig=home/'frozen_vq3_n8'
            train_vq3.run(Namespace(dataset_dir=str(src),collection_dir=str(col8),
                                    output_dir=str(orig),seed=42,trees=10))
            out=home/'vq5'
            fixed_panel_vq5.run(Namespace(dataset_dir=str(src),collection_dir=str(col),
                output_dir=str(out),seed=42,trees=10,sizes='8,12,16',anchor_dir=str(orig)))
            result=read_json(out/'report.json')
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(result['anchor_48_verification']['status'],'MATCH')
            self.assertEqual(len((out/'fold_decisions.jsonl').read_text().splitlines()),
                             3*16*len(result['policies']))
            for size,ntrain in [(8,6),(12,9),(16,12)]:
                for task,detail in result['folds'][str(size)]['20'].items():
                    self.assertEqual(detail['counts']['n_train_states'],ntrain)
                    self.assertEqual(detail['counts']['fixed_panel_eval_n'],4)
                    self.assertNotIn(task,detail['counts']['train_tasks'])
            # Cross-seed summary pin, no source mutation permitted.
            # Simulate second seed report without rerunning expensive training.
            b=home/'vq5_other';b.mkdir()
            rep=read_json(out/'report.json');rep['seed']=43;write_json(b/'report.json',rep)
            prov=read_json(out/'provenance.json');prov['seed']=43;write_json(b/'provenance.json',prov)
            summarize_vq5.run([out,b],home/'summary')
            self.assertEqual(read_json(home/'summary/report.json')['panel_n'],16)
            with self.assertRaises(FileExistsError):summarize_vq5.run([out,b],home/'summary')
            prov['dataset_paired_sha256']='different';write_json(b/'provenance.json',prov)
            with self.assertRaises(ValueError):summarize_vq5.run([out,b],home/'broken_summary')

if __name__=='__main__': unittest.main()
