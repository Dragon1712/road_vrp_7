#!/usr/bin/env python3
"""Fresh July-only Chicago Traffic Tracker indices and directed OSM mapping.

Does not read ANY files from the prior road_vrp repository.
Only source labels: July CSVs. OSM comes from a newly downloaded GraphML or a user-provided independent graph.
Mapping is geometric/heuristic, never certified ground truth.
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

META = ['street','direction','length','start_latitude','start_longitude','end_latitude','end_longitude']
USE = ['time','segment_id','speed'] + META
INDEX_FILENAME='segment_index.csv'


def checksum(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def json_write(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def bool_date(s):return dt.date.fromisoformat(s)


def run_index(args):
    a,b=bool_date(args.start),bool_date(args.end)
    if a>b:raise ValueError('start must <= end')
    train_end=bool_date(args.train_end)
    if train_end<a or train_end>b:raise ValueError('train-end must lie within date range')
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    all_days=pd.date_range(a,b,freq='D')
    stats=defaultdict(lambda:{'total_rows':0,'train_valid':0,'train_cells':0,'valid_all':0,'observed_all':0})
    attributes={}; conflict=[]; daily=[]; ids=set(); hashes=[]
    for k,date in enumerate(all_days):
        ds=date.strftime('%Y-%m-%d'); path=Path(args.input_dir)/f'traffic_{ds}.csv'
        if not path.is_file():raise FileNotFoundError(f'Missing July data file: {path}')
        cols=pd.read_csv(path,nrows=0).columns.tolist()
        missing=set(USE)-set(cols)
        if missing: raise ValueError(f'{path}: Missing columns {sorted(missing)}')
        df=pd.read_csv(path,usecols=USE,low_memory=False)
        ti=pd.to_datetime(df.time,errors='coerce')
        ix=pd.to_numeric(df.segment_id,errors='coerce')
        if ti.isna().any() or ix.isna().any() or (ix%1!=0).any():raise ValueError(f'Invalid time or ID: {path}')
        df['segment_id']=ix.astype(int); df['_slot']=ti.dt.floor('10min')
        wrong=(df['_slot'].dt.date!=date.date())
        if wrong.any():raise ValueError(f'{ds}: rows from other days: {int(wrong.sum())}')
        dupl=int(df.duplicated(['_slot','segment_id']).sum())
        if dupl:raise ValueError(f'{ds}: duplicate slot-ID records: {dupl}')
        speed=pd.to_numeric(df.speed,errors='coerce')
        valid=(speed>0)&(speed<=args.max_mph)&np.isfinite(speed)
        d_ids=set(map(int,df.segment_id.unique()))
        ids.update(d_ids)
        groups=df.groupby('segment_id',sort=False)
        day_has=groups.size(); day_valid=valid.groupby(df.segment_id).sum()
        for sid,num in day_has.items():
            sid=int(sid)
            stats[sid]['total_rows']+=int(num)
            stats[sid]['valid_all']+=int(day_valid.get(sid,0))
            stats[sid]['observed_all']+=int(num)
            if date.date()<=train_end:
                stats[sid]['train_cells']+=int(num)
                stats[sid]['train_valid']+=int(day_valid.get(sid,0))
        # Compare stable segment metadata within and between days.
        for sid,g in groups:
            sid=int(sid)
            row={m:g.iloc[0][m] for m in META}
            for m in META:
                unique=g[m].dropna().unique()
                if len(unique)>1:conflict.append({'date':ds,'segment_id':sid,'field':m,'kind':'within_day'})
            if sid not in attributes:attributes[sid]=row
            else:
                prev=attributes[sid]
                for m in META:
                    if m in ('length','start_latitude','start_longitude','end_latitude','end_longitude'):
                        x=pd.to_numeric(pd.Series([prev[m]]),errors='coerce').iloc[0]
                        y=pd.to_numeric(pd.Series([row[m]]),errors='coerce').iloc[0]
                        if not np.isclose(x,y,atol=1e-5,rtol=1e-6,equal_nan=True):
                            conflict.append({'date':ds,'segment_id':sid,'field':m,'kind':'cross_day'})
                    elif str(prev[m])!=str(row[m]):
                        conflict.append({'date':ds,'segment_id':sid,'field':m,'kind':'cross_day'})
        slots=int(df['_slot'].nunique()); unique_cells=int(len(df)); count=len(d_ids)
        daily.append({'date':ds,'rows':unique_cells,'segment_count':count,'slots':slots,
                      'valid_speed_rows':int(valid.sum()),'valid_ratio':float(valid.mean()),
                      'missing_slots':144-slots,'duplicate_cells':dupl})
        hashes.append({'path':str(path),'sha256':checksum(path) if args.hash_files else None})
        print(f'[{k+1:02d}/{len(all_days)}] {ds} IDs={count}, slots={slots}, valid={valid.mean():.2%}',flush=True)
    if args.fail_on_metadata_conflict and conflict: raise ValueError(f'{len(conflict)} static metadata conflicts; see report once --fail-on-metadata-conflict removed')
    ordered=sorted(ids)
    pd.DataFrame({'column_index':range(len(ordered)),'segment_id':ordered}).to_csv(output/INDEX_FILENAME,index=False)
    n_train=(train_end-a).days+1
    full_len=len(all_days)*144
    train_len=n_train*144
    details=[]
    for sid in ordered:
        st=stats[sid]; row=attributes[sid]
        details.append({'segment_id':sid,'street':row['street'],'direction':row['direction'],
                        'length':row['length'],'length_m':pd.to_numeric(pd.Series([row['length']]),errors='coerce').iloc[0]*1609.344,
                        'start_latitude':row['start_latitude'],'start_longitude':row['start_longitude'],
                        'end_latitude':row['end_latitude'],'end_longitude':row['end_longitude'],
                        'observed_train_cells':st['train_cells'],'valid_train_cells':st['train_valid'],
                        'train_valid_speed_ratio':st['train_valid']/max(train_len,1),
                        'all_valid_speed_ratio':st['valid_all']/max(full_len,1)})
    pd.DataFrame(details).to_csv(output/'segments_static_july.csv',index=False)
    pd.DataFrame(daily).to_csv(output/'july_daily_raw_quality.csv',index=False)
    pd.DataFrame(conflict,columns=['date','segment_id','field','kind']).to_csv(output/'july_metadata_conflicts.csv',index=False)
    report={'source':'independent July CSVs only','date_start':str(a),'date_end':str(b),'train_end':str(train_end),
            'file_count':len(all_days),'segment_count':len(ordered),'expected_cells':len(ordered)*full_len,
            'metadata_conflicts':len(conflict),'dates_missing_slots':[x['date'] for x in daily if x['missing_slots']],
            'index_sha256':checksum(output/INDEX_FILENAME), 'training_observation_only':True,
            'inputs':hashes}
    json_write(output/'july_index_manifest.json',report)
    print('INDEX SAVED:',output/INDEX_FILENAME,'segments:',len(ordered),'metadata conflicts:',len(conflict))
    return report


def run_download(args):
    try:import osmnx as ox
    except ImportError as e:raise SystemExit('Download requires `pip install "osmnx>=2,<3"` on an Internet-connected computer') from e
    bbox=(args.west,args.south,args.east,args.north)
    # OSMnx 2.x: (west,south,east,north)
    graph=ox.graph_from_bbox(bbox,network_type='drive',simplify=True,retain_all=False)
    if not graph.number_of_nodes() or not graph.number_of_edges():raise RuntimeError('OSM road download returned empty graph')
    path=Path(args.graphml);path.parent.mkdir(parents=True,exist_ok=True)
    ox.save_graphml(graph,filepath=path)
    print('New OSM:',path,'nodes:',len(graph),'directed edges:',graph.number_of_edges())
    json_write(path.with_suffix('.source.json'),{'source':'OSMnx fresh OpenStreetMap download',
        'bbox_west_south_east_north':bbox,'osmnx_version':ox.__version__,
        'nodes':len(graph),'directed_edges':graph.number_of_edges(),'graphml_sha256':checksum(path)})


_STOP={'n','s','e','w','north','south','east','west','st','street','ave','avenue','road','rd','drive','dr',
       'boulevard','blvd','highway','hwy','parkway','pkwy','place','pl','expressway','expy','the'}
def normalize_name(x):
    if isinstance(x,(list,tuple,set)):return ' '.join(normalize_name(y) for y in x)
    if x is None:return ''
    v=str(x)
    if v.startswith('['):
        try:
            y=ast.literal_eval(v)
            if isinstance(y,(list,tuple)):return normalize_name(y)
        except (ValueError,SyntaxError):pass
    return ' '.join(z for z in re.findall(r'[a-z0-9]+',v.lower()) if z not in _STOP)

def street_match(monitor,osm):
    m=normalize_name(monitor);o=normalize_name(osm)
    if not m or not o:return False
    return m==o or m in o.split(' / ') or o in m.split(' / ')


def direction_ok(row):
    try:
        dy=float(row.end_latitude)-float(row.start_latitude)
        dx=(float(row.end_longitude)-float(row.start_longitude))*math.cos(math.radians(float(row.start_latitude)))
        s=str(row.direction).upper().strip()
        if s in ('NB','N'):return dy>0 and abs(dy)>=.3*abs(dx)
        if s in ('SB','S'):return dy<0 and abs(dy)>=.3*abs(dx)
        if s in ('EB','E'):return dx>0 and abs(dx)>=.3*abs(dy)
        if s in ('WB','W'):return dx<0 and abs(dx)>=.3*abs(dy)
    except (TypeError,ValueError):return False
    return False


def closest(coords,names,lat,lon,k=4,limit=70.0):
    # local equirectangular approximation (meters); bbox a small Chicago region
    lat0=math.radians(float(lat));p=np.array([lat*111100,lon*111100*math.cos(lat0)])
    xy=np.column_stack([coords[:,0]*111100,coords[:,1]*111100*math.cos(lat0)])
    dis=np.linalg.norm(xy-p,axis=1)
    take=np.argsort(dis,kind='stable')[:min(k,len(dis))]
    return [(names[i],float(dis[i])) for i in take if dis[i]<=limit]


def best_arc(data,street):
    # networkx MultiDiGraph shortest_path weight callback receives a key->attr dict
    pairs=data.items() if (data and all(isinstance(k,dict) for k in data.values())) else [(0,data)]
    cand=[]
    for key,e in pairs:
        try:l=float(e.get('length',float('nan')))
        except (ValueError,TypeError):continue
        if not np.isfinite(l) or l<=0:continue
        match=street_match(street,e.get('name',''))
        cand.append((l*(1 if match else 5),l,str(key),match))
    return min(cand) if cand else (float('inf'),float('inf'),'',False)


def build_mapping(args):
    folder=Path(args.processed_dir)
    index_path=folder/INDEX_FILENAME
    seg_path=folder/'segments_static_july.csv'
    if not (index_path.is_file() and seg_path.is_file()):
        raise FileNotFoundError('First run: bootstrap_july.py index ...')
    if not Path(args.graphml).is_file():raise FileNotFoundError('Fresh OSM graph missing: run download-osm or supply an independent GraphML')
    ids=pd.read_csv(index_path).segment_id.astype(int).tolist()
    seg=pd.read_csv(seg_path)
    if set(seg.segment_id.astype(int))!=set(ids):raise ValueError('July static metadata IDs do not match July index')
    g=nx.read_graphml(args.graphml)
    if not g.is_directed():raise ValueError('OSM graph must be DIRECTED')
    nodes=[];coords=[]
    for u,e in g.nodes(data=True):
        try:ll=(float(e['y']),float(e['x']))
        except (KeyError,ValueError,TypeError):continue
        nodes.append(str(u));coords.append(ll)
    if not nodes:raise ValueError('Graph nodes missing latitude/longitude')
    coords=np.asarray(coords)
    candidates=[];results=[]
    for r in seg.itertuples(index=False):
        sid=int(r.segment_id)
        base={'segment_id':sid,'street':r.street,'valid_speed_ratio':float(r.train_valid_speed_ratio),
              'observed_length_m':float(r.length_m),'osm_route_length_m':np.nan,
              'length_ratio':np.nan,'snap_start_m':np.nan,'snap_end_m':np.nan,
              'street_fraction':np.nan,'osm_start_node':'','osm_end_node':'','match_status':'','path_edge_count':0}
        if float(r.train_valid_speed_ratio)<args.min_train_coverage:
            base['match_status']='low_train_coverage';results.append(base);continue
        nums=[r.start_latitude,r.start_longitude,r.end_latitude,r.end_longitude,r.length_m]
        if any(not np.isfinite(float(x)) for x in nums) or float(r.length_m)<=0:
            base['match_status']='invalid_coordinates_or_length';results.append(base);continue
        if not direction_ok(r):
            base['match_status']='direction_inconsistent';results.append(base);continue
        starts=closest(coords,nodes,r.start_latitude,r.start_longitude,args.k_nearest,args.snap_max_m)
        ends=closest(coords,nodes,r.end_latitude,r.end_longitude,args.k_nearest,args.snap_max_m)
        if not starts or not ends:
            base['match_status']='no_nearby_directed_osm_nodes';results.append(base);continue
        winner=None
        for st,sd in starts:
            for ed,edist in ends:
                if st==ed:continue
                def weight(u,v,edata):return best_arc(edata,r.street)[0]
                try:route=nx.shortest_path(g,st,ed,weight=weight)
                except (nx.NetworkXNoPath,nx.NodeNotFound):continue
                edges=[];dist=matched=0.
                for u,v in zip(route,route[1:]):
                    _,length,key,match=best_arc(g.get_edge_data(u,v),r.street)
                    if not np.isfinite(length):edges=[];break
                    dist+=length;matched+=length*match
                    edges.append((str(u),str(v),key,length,bool(match)))
                if not edges:continue
                ratio=dist/float(r.length_m)
                frac=matched/dist
                qualified=(args.length_ratio_min<=ratio<=args.length_ratio_max and frac>=args.min_street_fraction)
                score=abs(math.log(max(ratio,1e-6)))+2.0*(1-frac)+(sd+edist)*.003
                proposal=(not qualified,score,dist,ratio,frac,st,ed,sd,edist,edges)
                if winner is None or proposal[:2]<winner[:2]:winner=proposal
        if winner is None:
            base['match_status']='no_directed_path';results.append(base);continue
        bad,score,dist,ratio,frac,st,ed,sd,edist,edges=winner
        base.update(osm_route_length_m=round(dist,3),length_ratio=round(ratio,5),
                    snap_start_m=round(sd,2),snap_end_m=round(edist,2),street_fraction=round(frac,5),
                    osm_start_node=st,osm_end_node=ed,path_edge_count=len(edges),
                    match_status='accepted_candidate' if not bad else 'poor_geometry_or_street_match')
        results.append(base)
        if not bad:candidates.append((sid,base,edges,score))
    # Non-overlapping map; avoid leaking one monitored corridor onto several conflicting segments.
    # Prefer higher train coverage and better geometric score, with deterministic ties.
    candidates.sort(key=lambda x:(-x[1]['valid_speed_ratio'],x[3],x[0]))
    used=set(); accepted=[]; edge_rows=[]; excluded=[]
    for sid,base,edges,score in candidates:
        keys={(u,v,key) for u,v,key,_,_ in edges}
        if len(keys)!=len(edges):
            excluded.append({'segment_id':sid,'reason':'repeated_directed_arc_in_path'});continue
        if keys & used:
            excluded.append({'segment_id':sid,'reason':'osm_edge_collision'});continue
        used.update(keys)
        accepted.append(base.copy())
        total=float(sum(x[3] for x in edges))
        for u,v,key,length,nameok in edges:
            edge_rows.append({'segment_id':sid,'osm_u':u,'osm_v':v,'osm_key':key,
                              'osm_length_m':length,'osm_street_match':nameok,
                              'segment_valid_speed_ratio':base['valid_speed_ratio'],
                              'mapped_path_length_m':total,'valid_speed_ratio':base['valid_speed_ratio'],
                              'path_fraction':length/total})
    folder_out=folder/'osm_match_loop'/'safe_mapping'
    folder_out.mkdir(parents=True,exist_ok=True)
    results_df=pd.DataFrame(results)
    rejected_ids={r['segment_id'] for r in excluded}
    results_df.loc[results_df.segment_id.isin(rejected_ids),'match_status']='excluded_collision_or_loop'
    results_df.to_csv(folder/'july_osm_match_candidates.csv',index=False)
    pd.DataFrame(excluded,columns=['segment_id','reason']).to_csv(folder_out/'excluded_monitor_segments.csv',index=False)
    selected_cols=['segment_id','street','valid_speed_ratio','observed_length_m','osm_route_length_m',
                   'length_ratio','snap_start_m','snap_end_m','street_fraction','osm_start_node','osm_end_node',
                   'match_status','path_edge_count']
    selected=pd.DataFrame(accepted,columns=selected_cols).sort_values('segment_id')
    edge_cols=['segment_id','osm_u','osm_v','osm_key','osm_length_m','osm_street_match',
               'segment_valid_speed_ratio','mapped_path_length_m','valid_speed_ratio','path_fraction']
    edges=pd.DataFrame(edge_rows,columns=edge_cols).sort_values(['segment_id','osm_u','osm_v','osm_key'])
    if edges.duplicated(['osm_u','osm_v','osm_key']).any():raise AssertionError('OSM edge collision not resolved')
    if len(edges) and not np.allclose(edges.groupby('segment_id').path_fraction.sum(),1.,atol=1e-9):raise AssertionError('Path fractions not 1')
    selected.to_csv(folder_out/'selected_monitor_segments.csv',index=False)
    edges.to_csv(folder_out/'safe_osm_edge_mapping.csv',index=False)
    if not len(edges): raise RuntimeError('ZERO safely matched edges. Inspect july_osm_match_candidates.csv; do not fabricate mapping files')
    summary={'source':'fresh July CSV training partition + newly supplied OSM GraphML',
             'no_old_project_dependencies':True,'graphml_sha256':checksum(Path(args.graphml)),
             'segment_index_sha256':checksum(index_path),
             'monitor_candidates':len(candidates),'safe_segments':len(selected),'unique_directed_edges':len(edges),
             'excluded_due_to_conflicts':len(excluded),
             'min_train_coverage':args.min_train_coverage,'snap_max_m':args.snap_max_m,
             'length_ratio_interval':[args.length_ratio_min,args.length_ratio_max],
             'min_street_fraction':args.min_street_fraction,
             'warning':'Heuristic directed OSM mapping, NOT independent map-matching ground truth; manually inspect geometry/direction.'}
    json_write(folder_out/'safe_mapping_summary.json',summary)
    print('MAP SAVED:',folder_out,'selected corridors:',len(selected),'mapped directed edges:',len(edges))
    return summary


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='cmd',required=True)
    i=subs.add_parser('index',help='Independently rebuild monitor index and July segment metadata')
    i.add_argument('--input-dir',default='chicago_data');i.add_argument('--out-dir',default='datasets/processed/chicago')
    i.add_argument('--start',default='2025-07-02');i.add_argument('--end',default='2025-07-31')
    i.add_argument('--train-end',default='2025-07-21');i.add_argument('--max-mph',type=float,default=100.)
    i.add_argument('--hash-files',action='store_true');i.add_argument('--fail-on-metadata-conflict',action='store_true')
    dl=subs.add_parser('download-osm',help='Fresh OSMnx download (Internet required)')
    dl.add_argument('--graphml',default='datasets/raw/chicago/chicago_loop_drive_july.graphml')
    dl.add_argument('--west',type=float,default=-87.667);dl.add_argument('--south',type=float,default=41.851)
    dl.add_argument('--east',type=float,default=-87.592);dl.add_argument('--north',type=float,default=41.906)
    m=subs.add_parser('map',help='Independently map July monitor segments to fresh directed OSM')
    m.add_argument('--processed-dir',default='datasets/processed/chicago')
    m.add_argument('--graphml',default='datasets/raw/chicago/chicago_loop_drive_july.graphml')
    m.add_argument('--min-train-coverage',type=float,default=.50)
    m.add_argument('--snap-max-m',type=float,default=70.)
    m.add_argument('--k-nearest',type=int,default=4)
    m.add_argument('--length-ratio-min',type=float,default=.80)
    m.add_argument('--length-ratio-max',type=float,default=1.25)
    m.add_argument('--min-street-fraction',type=float,default=.75)
    a=p.parse_args(argv)
    if a.cmd=='index':return run_index(a)
    if a.cmd=='download-osm':return run_download(a)
    if a.cmd=='map':return build_mapping(a)
    raise ValueError(a.cmd)

if __name__=='__main__':cli()
