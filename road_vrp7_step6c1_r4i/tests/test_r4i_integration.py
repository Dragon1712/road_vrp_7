"""End-to-end synthetic check: source SHA and role gates, offline executable."""
from __future__ import annotations
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from r4i_core import save,sha,source_audit
import test_r4i_core as fixtures

class Integration(unittest.TestCase):
    def fixture(self,base):
        root=Path(base);h=root/'r4h';c=root/'capture';p=root/'prefix'
        for d in (h,c,p):d.mkdir()
        report,hm,labs,branches,plans,snaps=fixtures.R4ISynthetic().fixture()
        save(p/'report.json',{'stage':'R4G-TRAIN-PREFIX-EXACT-v1','status':'COMPLETE',
              'data_role':'TRAIN_ONLY','n_comparisons':298,'exact_matches':298})
        for fn in ('manifest.json','excluded_windows.json'):(p/fn).write_text('{}\n')
        (p/'paired_rows.jsonl').write_text('{}\n')
        save(c/'manifest.json',{'stage':'R4G-TRAIN-SECTION-SNAPSHOT-v1','split':'train','arm':'capture'})
        (c/'snapshots.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in snaps))
        (c/'metrics.jsonl').write_text('')
        (c/'provenance.json').write_text('{}\n')
        (h/'labels.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in labs))
        (h/'branch_summary.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in branches))
        (h/'plans.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in plans))
        save(h/'report.json',{'stage':'R4H-TRAIN-SECTION-COUNTERFACTUAL-v1','status':'COMPLETE',
          'data_role':'TRAIN_ONLY','n_labels':len(labs),'n_branches':len(branches),
          'n_sham_checks':len(plans)*3,'n_cross_k_prefix_checks':200,
          'n_scenario_snapshots':16,'n_distinct_physical_states':8,
          'source_prefix_exact_comparisons':298})
        hm.update({'stage':'R4H-TRAIN-SECTION-COUNTERFACTUAL-v1','status':'CONFIGURED',
          'data_role':'TRAIN_ONLY','horizons':[50,200,500],'require_all_k':True,
          'source_sha256':{'prefix_'+name:sha(p/f) for name,f in
                [('report','report.json'),('manifest','manifest.json'),
                 ('paired_rows','paired_rows.jsonl'),('excluded_windows','excluded_windows.json')]}})
        hm['source_sha256'].update({'capture_'+name:sha(c/f) for name,f in
                [('manifest','manifest.json'),('snapshots','snapshots.jsonl'),
                 ('metrics','metrics.jsonl'),('provenance','provenance.json')]})
        save(h/'manifest.json',hm)
        return h,c,p
    def test_complete_probe(self):
        with tempfile.TemporaryDirectory() as d:
            h,c,p=self.fixture(d)
            command=[sys.executable,str(P/'probe_r4i.py'),'--r4h-dir',str(h),
                  '--capture-dir',str(c),'--prefix-dir',str(p),'--output-dir',str(Path(d)/'out')]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads((Path(d)/'out'/'report.json').read_text())
            self.assertEqual(result['signal']['n_snapshots'],16)
            self.assertEqual(result['signal']['n_physical_states'],8)
            self.assertEqual(len(result['folds']),64)
            self.assertTrue((Path(d)/'out'/'fold_predictions.jsonl').exists())
    def test_wrong_role_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            h,c,p=self.fixture(d)
            f=c/'manifest.json';man=json.loads(f.read_text());man['split']='validation';save(f,man)
            with self.assertRaisesRegex(ValueError,'Nontrain R4G capture'):
                source_audit(h,c,p)
    def test_changed_prefix_proof_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            h,c,p=self.fixture(d)
            (p/'paired_rows.jsonl').write_text('tampered\n')
            with self.assertRaisesRegex(ValueError,'Prefix proof changed'):
                source_audit(h,c,p)
    def test_changed_capture_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            h,c,p=self.fixture(d)
            with (c/'snapshots.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaisesRegex(ValueError,'Capture provenance differs'):
                source_audit(h,c,p)
if __name__=='__main__':unittest.main()
