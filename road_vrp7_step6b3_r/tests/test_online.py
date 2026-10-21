import sys,tempfile,unittest,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from online_compare_b3r import run


class TestOnline(unittest.TestCase):
    def make_run(self,folder,values,extra=0):
        folder=Path(folder);folder.mkdir()
        mf={k: 'same' for k in ('source_sha256','road_sha256','edge_sha256','solver_sha256',
            'section_sha256','split','seeds','task_ids','scenario_ids')}
        mf.update(wall_sec=30,max_iterations=20000,anneal_iterations=5000,top_k=12,
            action_count=8,sampler_seed=20261019,r_score_every=8,r_max_scoring_sec=2,
            r_cache_size=64,r_min_static_fraction=0,r_fallback='c3')
        mf['top_k']+=extra
        (folder/'manifest.json').write_text(json.dumps(mf))
        (folder/'summary.json').write_text(json.dumps({'status':'COMPLETE','records':2,'expected_records':2}))
        rows=[{'task_id':f'task_000{i}','scenario_id':'july25_0800','date':'2025-07-25',
               'seed':42,'iterations':19000,'ex_post_road_travel_sec':value}
               for i,value in enumerate(values)]
        with (folder/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def test_paired_and_strict_meta(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            self.make_run(root/'M0',[100.,200.]);self.make_run(root/'M3',[90.,210.])
            res=run({'M0':root/'M0','M3':root/'M3'},root/'out')
            self.assertEqual(res['comparisons']['M3']['wins'],1)
            self.assertEqual(res['comparisons']['M3']['losses'],1)
            self.assertEqual(res['comparisons']['M3']['mean_road_improvement_vs_M0_sec'],0.)
            self.make_run(root/'bad',[90.,200.],extra=1)
            with self.assertRaises(ValueError):run({'M0':root/'M0','M3':root/'bad'},root/'out2')
if __name__=='__main__':unittest.main()
