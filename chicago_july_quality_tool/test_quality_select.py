#!/usr/bin/env python3
"""Self-contained synthetic QA tests; no Chicago data needed."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import batch_quality_select as qc
from make_validation_tasks import main as make_tasks


def fixture(td):
    p=Path(td);data=p/'chicago_data';data.mkdir()
    pd.DataFrame({'column_index':[0,1,2], 'segment_id':[10,20,30]}).to_csv(p/'segment_index.csv',index=False)
    pd.DataFrame({'segment_id':[10,20]}).to_csv(p/'selected.csv',index=False)
    pd.DataFrame({'segment_id':[10,10,20],
                  'osm_u':['1','2','3'],'osm_v':['2','3','4'],'osm_key':['0','0','0']}).to_csv(p/'edges.csv',index=False)
    for date in ('2025-07-28','2025-07-29'):
        time = pd.date_range(date,periods=144,freq='10min')
        rows=[]
        for k,t in enumerate(time):
            for seg in [10,20,30]:
                if date.endswith('29') and k==48 and seg==20: continue  # missing slot-cell
                rows.append({'time':(t+pd.Timedelta(seconds=63)).strftime('%Y-%m-%dT%H:%M:%S'),
                             'segment_id':seg,'speed':20 if k%2==0 else 25,
                             'street':str(seg),'direction':'NB','length':.5,
                             'start_latitude':41.,'start_longitude':-87.,
                             'end_latitude':41.1,'end_longitude':-87.1})
        if date.endswith('29'):
            rows.append(dict(rows[0])) # duplicate
        pd.DataFrame(rows).to_csv(data/f'traffic_{date}.csv',index=False)
    return p,data


def test():
    with tempfile.TemporaryDirectory() as tmp:
        root,data=fixture(tmp)
        argv=[ '--input-dir',str(data),
               '--segment-index',str(root/'segment_index.csv'),
               '--mapped-segments',str(root/'selected.csv'),
               '--edge-mapping',str(root/'edges.csv'),
               '--start','2025-07-28','--end','2025-07-29',
               '--departures','08:00,17:00',
               '--output-dir',str(root/'result')]
        assert qc.main(argv)==0
        d=pd.read_csv(root/'result/daily_quality.csv')
        assert len(d)==2
        assert bool(d.loc[0,'structural_ok'])
        assert not bool(d.loc[1,'structural_ok'])
        assert d.loc[0,'mapped_edges']==3
        assert d.loc[0,'valid_cells']==432
        assert d.loc[1,'duplicate_segment_slots']==1
        assert d.loc[1,'missing_grid_cells']==1
        s=pd.read_csv(root/'result/selected_scenarios.csv')
        assert len(s)==2 and set(s.date)=={'2025-07-28'}
        assert set(s.departure)=={'08:00','17:00'}
        q=pd.read_csv(root/'result/rejected_scenarios.csv')
        assert len(q)==2 and q.rejection_reasons.str.contains('structural_quality_failure').all()
        manifest=json.loads((root/'result/audit_manifest.json').read_text())
        assert manifest['mapped_edge_count']==3
        # Missing input is reportable and strictly rejected if asked.
        argv2=[v for v in argv]
        argv2[argv2.index('--end')+1]='2025-07-30'
        argv2[argv2.index('--output-dir')+1]=str(root/'missing_result')
        assert qc.main(argv2+['--fail-on-missing'])==2
        m=pd.read_csv(root/'missing_result/daily_quality.csv')
        assert len(m)==3 and not bool(m.iloc[-1].exists)
        assert 'missing_file' in pd.read_csv(root/'missing_result/rejected_scenarios.csv').rejection_reasons.tolist()
        # Mapping collision should be refused.
        pd.DataFrame({'segment_id':[10,20], 'osm_u':['1','1'], 'osm_v':['2','2'],
                      'osm_key':['0','0']}).to_csv(root/'bad_edge_map.csv',index=False)
        try:
            qc.read_mapping(root/'bad_edge_map.csv',[10,20]);assert False
        except ValueError as e:assert 'multiple' in str(e)
        # Task generation preserves customer information & does not modify source.
        tasks=root/'old_tasks';tasks.mkdir()
        old={'task_id':'task_0000','vehicles':5,'customers':[{'id':i,'demand':1} for i in range(1,31)],
             'scenarios':[{'scenario_id':'old','departure_local':'2025-06-01 08:00:00'}]}
        (tasks/'task_0000.json').write_text(json.dumps(old))
        make_tasks(['--selected',str(root/'result/selected_scenarios.json'),
                    '--task-dir',str(tasks),'--output-dir',str(root/'new_tasks')])
        new=json.loads((root/'new_tasks/task_0000.json').read_text())
        assert len(new['scenarios'])==2 and new['customers']==old['customers']
        assert json.loads((tasks/'task_0000.json').read_text())==old
        print('PASS: clean day, duplicate, missing cell, missing file, edge collision, split, selection, task export')

if __name__=='__main__': test()
