import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from bootstrap_july import run_index, build_mapping, direction_ok, normalize_name


def test_real_day():
    original=Path('/mnt/data/traffic_2025-07-02.csv')
    if not original.exists():
        print('SKIP: original July02 CSV not available in local test runtime')
        return
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)
        d=p/'raw';d.mkdir()
        (d/'traffic_2025-07-02.csv').symlink_to(original)
        args=argparse.Namespace(input_dir=str(d),out_dir=str(p/'out'),start='2025-07-02',end='2025-07-02',train_end='2025-07-02',max_mph=100.,hash_files=False,fail_on_metadata_conflict=False)
        run_index(args)
        idx=pd.read_csv(p/'out'/'segment_index.csv')
        assert len(idx)==1047
        assert idx.column_index.to_list()==list(range(1047))
        q=pd.read_csv(p/'out'/'july_daily_raw_quality.csv')
        assert int(q.iloc[0].rows)==150768
        assert int(q.iloc[0].valid_speed_rows)==82401
        assert json.loads((p/'out'/'july_index_manifest.json').read_text())['metadata_conflicts']==0


def test_direction_and_name():
    assert normalize_name('West Madison Street')=='madison'
    assert direction_ok(argparse.Namespace(start_latitude=41.88,end_latitude=41.88,start_longitude=-87.65,end_longitude=-87.648,direction='EB'))
    assert not direction_ok(argparse.Namespace(start_latitude=41.88,end_latitude=41.88,start_longitude=-87.65,end_longitude=-87.648,direction='WB'))


def test_map():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);out=p/'processed';out.mkdir()
        g=nx.MultiDiGraph()
        coords=[(41.88,-87.65),(41.88,-87.648),(41.88,-87.646)]
        for i,(la,lo) in enumerate(coords):g.add_node(str(i),x=lo,y=la)
        g.add_edge('0','1',key='0',length=165,name='East Lake Street')
        g.add_edge('1','2',key='0',length=165,name='Lake')
        g.add_edge('1','0',key='0',length=165,name='Lake')
        g.add_edge('2','1',key='0',length=165,name='Lake')
        graph=p/'fresh.graphml';nx.write_graphml(g,graph)
        # 1 is best candidate for 0->1, 2 collides with 1 at same 0->1, 3 mismatch direction
        entries=[]
        for sid,frac,direction in [(1,.95,'EB'),(2,.55,'EB'),(3,.80,'WB')]:
            entries.append({'segment_id':sid,'street':'Lake','direction':direction,'length':.1025,
                            'length_m':165,'start_latitude':41.88,'start_longitude':-87.65,
                            'end_latitude':41.88,'end_longitude':-87.648,
                            'observed_train_cells':144,'valid_train_cells':round(frac*144),
                            'train_valid_speed_ratio':frac,'all_valid_speed_ratio':frac})
        pd.DataFrame(entries).to_csv(out/'segments_static_july.csv',index=False)
        pd.DataFrame({'column_index':range(3),'segment_id':[1,2,3]}).to_csv(out/'segment_index.csv',index=False)
        args=argparse.Namespace(processed_dir=str(out),graphml=str(graph),min_train_coverage=.5,
                                snap_max_m=70.,k_nearest=3,length_ratio_min=.8,length_ratio_max=1.25,min_street_fraction=.75)
        summ=build_mapping(args)
        assert summ['safe_segments']==1,summ
        assert summ['unique_directed_edges']==1,summ
        edge=pd.read_csv(out/'osm_match_loop'/'safe_mapping'/'safe_osm_edge_mapping.csv')
        assert edge.segment_id.tolist()==[1]
        assert abs(edge.path_fraction.sum()-1)<1e-9
        exc=pd.read_csv(out/'osm_match_loop'/'safe_mapping'/'excluded_monitor_segments.csv')
        assert 2 in set(exc.segment_id)
        cand=pd.read_csv(out/'july_osm_match_candidates.csv')
        assert cand.loc[cand.segment_id.eq(3),'match_status'].iloc[0]=='direction_inconsistent'


def test_fails_without_osm():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)
        pd.DataFrame({'segment_id':[1]}).to_csv(p/'segment_index.csv',index=False)
        pd.DataFrame({'segment_id':[1]}).to_csv(p/'segments_static_july.csv',index=False)
        args=argparse.Namespace(processed_dir=str(p),graphml=str(p/'missing.graphml'))
        try:build_mapping(args)
        except FileNotFoundError:pass
        else:raise AssertionError('Must not fabricate OSM mapping without road graph')

if __name__=='__main__':
    test_direction_and_name();test_map();test_fails_without_osm();test_real_day();print('PASS: fresh directed OSM synthetic tests; July02 real CSV tested when locally present')
