"""Self-contained synthetic regression tests for July pipeline, no June files."""
from pathlib import Path
import json
import tempfile
import datetime as dt
import numpy as np
import pandas as pd
import networkx as nx
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_july_stage2 import main, sha256, day_array, arrival_linear


def fixture(root:Path):
    (root/'chicago_data').mkdir(parents=True)
    src=root/'datasets/processed/chicago'
    src.mkdir(parents=True)
    graphdir=root/'datasets/raw/chicago';graphdir.mkdir(parents=True)
    mappingdir=src/'osm_match_loop/safe_mapping';mappingdir.mkdir(parents=True)
    idx=pd.DataFrame({'column_index':[0,1,2],'segment_id':[11,12,13]})
    idx.to_csv(src/'segment_index.csv',index=False)
    static=pd.DataFrame({'segment_id':[11,12,13],'length_m':[200.,100.,300.],
                         'train_valid_speed_ratio':[.8,.8,.8],
                         'street':['A','B','C']})
    static.to_csv(src/'segments_static_july.csv',index=False)
    g=nx.MultiDiGraph()
    for u,v,k,l in [('a','b','0',100.),('b','c','0',100.),('c','d','0',100.),('d','a','0',100.),
                    ('b','a','0',100.),('c','b','0',100.),('d','c','0',100.),('a','d','0',100.)]:
        g.add_edge(u,v,key=k,length=l,highway='residential')
    graph=graphdir/'chicago_loop_drive.graphml';nx.write_graphml(g,graph)
    mapping=pd.DataFrame([
        {'segment_id':11,'osm_u':'a','osm_v':'b','osm_key':'0','osm_length_m':100.,'path_fraction':.5},
        {'segment_id':11,'osm_u':'b','osm_v':'c','osm_key':'0','osm_length_m':100.,'path_fraction':.5},
        {'segment_id':12,'osm_u':'c','osm_v':'d','osm_key':'0','osm_length_m':100.,'path_fraction':1.},
    ])
    mapping.to_csv(mappingdir/'safe_osm_edge_mapping.csv',index=False)
    pd.DataFrame({'segment_id':[11,12]}).to_csv(mappingdir/'selected_monitor_segments.csv',index=False)
    (mappingdir/'safe_mapping_summary.json').write_text(json.dumps({'graphml_sha256':sha256(graph),
       'segment_index_sha256':sha256(src/'segment_index.csv'),'safe_segments':2,'unique_directed_edges':3}))
    for day in ['2025-07-02','2025-07-03']:
        date=dt.date.fromisoformat(day)
        rows=[]
        for t in pd.date_range(day,periods=144,freq='10min'):
            for sid in [11,12,13]:
                value = 30 if sid==11 else (10 if sid==12 else 20)
                if t.hour in (8,17) and sid==11:value=8
                if sid==11 and t.hour==3:value=-1
                rows.append({'time':t.isoformat(),'segment_id':sid,'speed':value})
        pd.DataFrame(rows).to_csv(root/f'chicago_data/traffic_{day}.csv',index=False)
    return src,graph,mappingdir


def run_test():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        src,graph,mappingdir=fixture(root)
        out=root/'processed_july_v1'
        params=['--csv-dir',str(root/'chicago_data'),'--segment-index',str(src/'segment_index.csv'),
                '--static-segments',str(src/'segments_static_july.csv'),
                '--mapping',str(mappingdir/'safe_osm_edge_mapping.csv'),
                '--selected-segments',str(mappingdir/'selected_monitor_segments.csv'),
                '--mapping-summary',str(mappingdir/'safe_mapping_summary.json'),
                '--graphml',str(graph),'--output-dir',str(out),
                '--start','2025-07-02','--end','2025-07-03','--hash-files']
        assert main(['build']+params)==0
        assert main(['validate','--output-dir',str(out),
                     '--mapping',str(mappingdir/'safe_osm_edge_mapping.csv'),
                     '--static-segments',str(src/'segments_static_july.csv')])==0
        assert main(['tdsp-smoke','--output-dir',str(out),
                     '--departure','2025-07-02 08:00:00','--pairs','8'])==0
        assert main(['tdsp-smoke','--output-dir',str(out),
                     '--departure','2025-07-03 17:00:00','--pairs','8'])==0
        with np.load(out/'traffic_tensor.npz',allow_pickle=False) as z:
            assert z['speed_kph'].shape==(288,3)
            assert z['observed_mask'].shape==(288,3)
            assert z['observed_mask'].sum()==276*1+288*2
        with np.load(out/'dynamic_costs/edge_costs.npz',allow_pickle=False) as z:
            raw=z['travel_sec_raw'];fifo=z['travel_sec_fifo'];obs=z['chicago_observed_mask']
            assert raw.shape==(288,8)
            assert int(z['is_mapped_edge'].sum())==3
            assert obs.sum()==276*2+288
            assert np.all(fifo>0)
        # GraphML tampering must be caught by mapping version check before build.
        graph2=root/'mismatch.graphml'
        graph2.write_bytes(graph.read_bytes()+b'\n')
        tampered_params=params.copy()
        tampered_params[tampered_params.index('--graphml')+1]=str(graph2)
        tampered_params[tampered_params.index('--output-dir')+1]=str(root/'tampered_output')
        try:
            main(['build']+tampered_params)
        except ValueError as exc:
            assert 'graphml_sha256' in str(exc), f'Unexpected mismatch reason: {exc}'
        else:raise AssertionError('Mapping provenance hash mismatch was not detected')
        # Structural hole must be rejected instead of silently interpreted as zero speed.
        hole=root/'hole.csv'
        z=pd.read_csv(root/'chicago_data/traffic_2025-07-02.csv')
        z.iloc[:-1].to_csv(hole,index=False)
        try:
            day_array(hole,dt.date(2025,7,2),[11,12,13],100.,True)
        except ValueError:pass
        else:raise AssertionError('Missing structural record was silently accepted')
        print('PASS: build, conservation, masks, FIFO, 2 TDSP smoke tests, structural guard')


if __name__=='__main__':run_test()
