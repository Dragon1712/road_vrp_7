#!/usr/bin/env python3
"""road_vrp_7 stage 2: July speed tensor -> mapped directed OSM FIFO TD costs.

Self-contained offline preprocessing. No 2025-06 input or legacy repo dependencies.
Historical ex-post observations are NOT future traffic forecasts.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import heapq
import json
import math
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

EDGE = ['osm_u', 'osm_v', 'osm_key']
STEP_SECONDS = 600.0
KPH_PER_MPH = 1.609344
DEFAULT_KPH = {'motorway': 80, 'trunk': 65, 'primary': 50, 'secondary': 45,
               'tertiary': 40, 'residential': 30, 'living_street': 15,
               'service': 20, 'unclassified': 35, 'other': 35}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def norm_id(v) -> str:
    s = str(v).strip()
    if re.fullmatch(r'[+-]?\d+\.0+', s):
        return s.split('.')[0]
    return s


def highway_kind(v) -> str:
    raw = str(v).lower()
    for k in DEFAULT_KPH:
        if k in raw:
            return k
    return 'other'


def static_speed(attr: dict) -> tuple[float, str]:
    try:
        x = float(attr.get('speed_kph', 'nan'))
        if 5 <= x <= 140 and math.isfinite(x):
            return x, 'osm_speed_kph_estimate'
    except (ValueError, TypeError):
        pass
    raw = str(attr.get('maxspeed', '') or '')
    m = re.search(r'\d+(?:\.\d+)?', raw)
    if m:
        v = float(m.group())
        if 'km' not in raw.lower() and 'kph' not in raw.lower():
            v *= KPH_PER_MPH  # bare Chicago maxspeed interpreted as mph
        if 5 <= v <= 140:
            return v, 'osm_maxspeed_heuristic'
    kind = highway_kind(attr.get('highway', 'other'))
    return float(DEFAULT_KPH[kind]), 'highway_class_fallback'


def graph_edges(graphml: Path) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    g = nx.read_graphml(graphml)
    if not g.is_directed() or g.number_of_edges() == 0:
        raise ValueError('GraphML must be a nonempty directed driving graph')
    iterable = g.edges(keys=True, data=True) if g.is_multigraph() else ((u, v, '0', a) for u, v, a in g.edges(data=True))
    rows = []
    for u, v, key, attrs in iterable:
        length = float(attrs.get('length', 'nan'))
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f'Invalid length in OSM GraphML: {(u, v, key)}')
        speed, source = static_speed(attrs)
        rows.append({'osm_u': norm_id(u), 'osm_v': norm_id(v), 'osm_key': norm_id(key),
                     'length_m': length, 'static_speed_kph': speed,
                     'static_speed_source': source, 'static_travel_sec': 3.6 * length / speed,
                     'highway': highway_kind(attrs.get('highway', 'other'))})
    edges = pd.DataFrame(rows).sort_values(EDGE).reset_index(drop=True)
    if edges.duplicated(EDGE).any():
        raise ValueError('Duplicate directed OSM edge identifiers')
    edges.insert(0, 'edge_idx', np.arange(len(edges), dtype=np.int64))
    return g, edges


def load_references(args, edges: pd.DataFrame) -> tuple[list[int], np.ndarray, pd.DataFrame, pd.DataFrame]:
    idx = pd.read_csv(args.segment_index)
    if not {'column_index', 'segment_id'}.issubset(idx.columns):
        raise ValueError('segment_index.csv requires column_index, segment_id')
    index = pd.to_numeric(idx['segment_id'], errors='raise').astype('int64').tolist()
    if len(set(index)) != len(index) or idx['column_index'].tolist() != list(range(len(index))):
        raise ValueError('Nonunique or unaligned segment index')
    static = pd.read_csv(args.static_segments)
    if not {'segment_id', 'length_m', 'train_valid_speed_ratio'}.issubset(static.columns):
        raise ValueError('Missing July training metadata columns')
    static['segment_id'] = static['segment_id'].astype(int)
    if static.segment_id.duplicated().any() or set(static.segment_id) != set(index):
        raise ValueError('July static road segment IDs do not match segment_index')
    length_by_index = static.set_index('segment_id').loc[index, 'length_m'].to_numpy(float)
    if not np.isfinite(length_by_index).all() or (length_by_index <= 0).any():
        raise ValueError('Invalid monitoring segment lengths')
    mapping = pd.read_csv(args.mapping, dtype={k: 'string' for k in EDGE})
    required = set(EDGE + ['segment_id', 'osm_length_m', 'path_fraction'])
    if not required.issubset(mapping.columns):
        raise ValueError('Mapping missing '+str(required-set(mapping.columns)))
    mapping['segment_id'] = pd.to_numeric(mapping['segment_id'], errors='raise').astype(int)
    for col in EDGE:
        mapping[col] = mapping[col].map(norm_id)
    mapping['osm_length_m'] = pd.to_numeric(mapping['osm_length_m'], errors='raise')
    mapping['path_fraction'] = pd.to_numeric(mapping['path_fraction'], errors='raise')
    if mapping.empty or mapping.duplicated(EDGE).any():
        raise ValueError('Empty or conflicting OSM directed edge mapping')
    if not set(mapping.segment_id).issubset(set(index)):
        raise ValueError('Mapped monitored IDs not found in segment_index')
    selected = pd.read_csv(args.selected_segments).segment_id.astype(int).tolist()
    if set(selected) != set(mapping.segment_id):
        raise ValueError('Selected monitored segments and mapping refer to different IDs')
    sumfractions = mapping.groupby('segment_id').path_fraction.sum().to_numpy(float)
    if not np.isfinite(sumfractions).all() or not np.allclose(sumfractions, 1, atol=1e-5):
        raise ValueError('Each monitored corridor must have path_fraction total 1')
    if not np.isfinite(mapping.path_fraction).all() or (mapping.path_fraction <= 0).any():
        raise ValueError('All path_fraction must be positive and finite')
    joined = mapping.merge(edges[EDGE + ['edge_idx', 'length_m']], on=EDGE,
                           how='left', validate='one_to_one', indicator=True)
    missing = joined['_merge'] != 'both'
    if missing.any():
        raise ValueError(f'{int(missing.sum())} mapped OSM edges absent from GraphML: '
                         f'{joined.loc[missing, EDGE].head(4).to_dict("records")}')
    if not np.all(np.isclose(joined['osm_length_m'], joined['length_m'], rtol=.02, atol=2)):
        raise ValueError('OSM mapped edge lengths conflict with current GraphML')
    if args.mapping_summary.is_file():
        obj = json.loads(args.mapping_summary.read_text(encoding='utf-8'))
        for name, actual in [('graphml_sha256', sha256(args.graphml)),
                             ('segment_index_sha256', sha256(args.segment_index))]:
            if name in obj and obj[name] != actual:
                raise ValueError(f'{name} disagrees with July mapping summary: wrong reference version')
        for name, current in [('safe_segments', len(set(selected))),
                              ('unique_directed_edges', len(mapping))]:
            if name in obj and int(obj[name]) != current:
                raise ValueError(f'July mapping summary {name} mismatch')
    return index, length_by_index, joined.drop(columns=['_merge']), static


def day_array(path: Path, day: dt.date, index: list[int], max_mph: float, strict_grid: bool) -> tuple[np.ndarray, dict]:
    source = pd.read_csv(path, usecols=['time','segment_id','speed'], low_memory=False)
    times = pd.to_datetime(source['time'], errors='coerce')
    sid = pd.to_numeric(source['segment_id'], errors='coerce')
    if times.isna().any() or sid.isna().any() or (sid % 1 != 0).any():
        raise ValueError(f'Unparseable time or noninteger segment_id in {path}')
    source['segment_id'] = sid.astype(np.int64)
    source['slot'] = times.dt.floor('10min')
    if (source['slot'].dt.date != day).any():
        raise ValueError(f'Wrong CSV date or clock timezone in {path}')
    if not set(source.segment_id.unique()).issubset(set(index)):
        raise ValueError(f'Unknown monitored segments in {path}')
    if source.duplicated(['slot','segment_id']).any():
        raise ValueError(f'Duplicate segment x slot records in {path}')
    clock = pd.date_range(pd.Timestamp(day), periods=144, freq='10min')
    source['speed'] = pd.to_numeric(source['speed'], errors='coerce')
    speed = source.pivot(index='slot', columns='segment_id', values='speed').reindex(index=clock, columns=index)
    nrows = len(source)
    if strict_grid and (nrows != len(index)*144 or source['slot'].nunique() != 144 or speed.isna().all(axis=1).any()):
        raise ValueError(f'Incomplete day structure in {path}; do not fill missing records silently')
    vals = speed.to_numpy(dtype=np.float32)
    valid = np.isfinite(vals) & (vals > 0) & (vals <= max_mph)
    vals[~valid] = np.nan
    return vals, {'date':str(day),'rows':int(nrows),'valid_speed_cells':int(valid.sum()),
                  'valid_speed_fraction':float(valid.mean()), 'unknown_speed_cells':int(valid.size-valid.sum()),
                  'slots':int(source['slot'].nunique())}


def process_traffic(args, idx: list[int], len_m: np.ndarray, out: Path):
    days = pd.date_range(args.start, args.end, freq='D')
    if len(days) == 0:
        raise ValueError('No dates')
    speed_kph = np.full((len(days)*144, len(idx)), np.nan, dtype=np.float32)
    obs_mask = np.zeros_like(speed_kph, dtype=bool)
    clock = pd.date_range(pd.Timestamp(args.start), periods=len(days)*144, freq='10min')
    quality = []; sources = []
    # One calendar day (~150k rows) in memory at a time.
    with (out/'traffic_long.csv').open('w', newline='', encoding='utf-8') as stream:
        import csv
        writer = csv.writer(stream)
        writer.writerow(['slot','segment_id','speed_kph','segment_travel_sec','observed'])
        for d, time in enumerate(days):
            day = time.date()
            path = args.csv_dir/f'traffic_{day.isoformat()}.csv'
            if not path.is_file():
                raise FileNotFoundError(f'Missing day {path}')
            mph, metrics = day_array(path, day, idx, args.max_mph, strict_grid=True)
            finite = np.isfinite(mph)
            speed_kph[d*144:(d+1)*144] = mph*KPH_PER_MPH
            obs_mask[d*144:(d+1)*144] = finite
            # Write chunk-by-chunk, retain all cells and explicit masks; no NaN -> false speed.
            lmi = len_m/1609.344
            for i in range(144):
                slotstr = clock[d*144+i].strftime('%Y-%m-%d %H:%M:%S')
                for j, ident in enumerate(idx):
                    if finite[i,j]:
                        x = float(mph[i,j]); sec = 3600.0*float(lmi[j])/x
                        writer.writerow([slotstr, ident, f'{x*KPH_PER_MPH:.6f}', f'{sec:.6f}', True])
                    else:
                        writer.writerow([slotstr, ident, '', '', False])
            sources.append({'date':day.isoformat(),'path':str(path),'sha256':sha256(path) if args.hash_files else None})
            quality.append(metrics)
            print(f'[{d+1:02d}/{len(days)}] {day} cells={len(idx)*144} valid={metrics["valid_speed_fraction"]:.2%}', flush=True)
    if (np.isfinite(speed_kph) != obs_mask).any():
        raise AssertionError('Traffic speed array / observed mask mismatch')
    np.savez_compressed(out/'traffic_tensor.npz', speed_kph=speed_kph, observed_mask=obs_mask,
                        segment_ids=np.array(idx, dtype=np.int64),slot_local_naive_ns=clock.asi8)
    pd.DataFrame({'time_local':clock.strftime('%Y-%m-%d %H:%M:%S')}).to_csv(out/'time_index.csv',index=False)
    pd.DataFrame({'column_index':range(len(idx)), 'segment_id':idx}).to_csv(out/'segment_index.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'july_traffic_daily_quality.csv',index=False)
    return speed_kph,obs_mask,clock,quality,sources


def make_edge_costs(args, out: Path, g, edges: pd.DataFrame, mapping: pd.DataFrame,
                    idx: list[int], len_m: np.ndarray, speed_kph: np.ndarray, obs_mask: np.ndarray,
                    clock: pd.DatetimeIndex) -> dict:
    costdir = out/'dynamic_costs'
    costdir.mkdir(parents=True,exist_ok=True)
    E = len(edges); T = len(clock)
    edges = edges.copy()
    edges['mapped_segment_id'] = ''
    edges['mapped_path_fraction'] = np.nan
    static_tt = edges.static_travel_sec.to_numpy(np.float32)
    raw = np.broadcast_to(static_tt, (T,E)).copy()
    edge_mask = np.zeros((T,E),dtype=bool)
    pos = {int(k):i for i,k in enumerate(idx)}
    for r in mapping.itertuples(index=False):
        j=int(r.edge_idx); i=pos[int(r.segment_id)]
        valid=obs_mask[:,i]
        # 3.6*(meters)/kmh; distribute MONITORED CORRIDOR travel along mapped path.
        corridor_sec = 3.6*len_m[i]/speed_kph[valid,i]
        raw[valid,j] = corridor_sec*np.float32(r.path_fraction)
        edge_mask[valid,j] = True
        edges.at[j,'mapped_segment_id'] = str(int(r.segment_id))
        edges.at[j,'mapped_path_fraction'] = float(r.path_fraction)
    mapped = edges.mapped_segment_id.ne('').to_numpy(bool)
    if mapped.sum() != len(mapping):
        raise AssertionError('OSM mapped edge count mismatch')
    if edge_mask[:,~mapped].any() or not np.isfinite(raw).all() or (raw<=0).any():
        raise ValueError('Raw travel times invalid or observed_mask leaked into unmapped edges')
    # FIFO repair, blocked for memory safety. Grid arrivals must be nondecreasing.
    fifo = np.empty_like(raw)
    rel=np.arange(T,dtype=np.float64)*STEP_SECONDS
    adjusted=0; largest=0.0
    for a in range(0,E,256):
        b=min(a+256,E)
        arrivals=np.maximum.accumulate(raw[:,a:b].astype(np.float64)+rel[:,None],axis=0)
        fixed=(arrivals-rel[:,None]).astype(np.float32)
        if not np.isfinite(fixed).all() or (fixed<=0).any():
            raise ValueError('Nonfinite/nonpositive FIFO costs')
        fifo[:,a:b]=fixed
        delta=fixed.astype(np.float64)-raw[:,a:b].astype(np.float64)
        adjusted+=int((delta>0.001).sum())
        largest=max(largest,float(np.maximum(delta,0).max()))
    # Caution: fifo values may differ from original observations; retain raw separately.
    edges.to_csv(costdir/'edge_index.csv', index=False)
    pd.DataFrame({'time_local':clock.strftime('%Y-%m-%d %H:%M:%S')}).to_csv(costdir/'time_index.csv', index=False)
    np.savez_compressed(costdir/'edge_costs.npz',travel_sec_raw=raw,travel_sec_fifo=fifo,
                        chicago_observed_mask=edge_mask,is_mapped_edge=mapped,
                        slot_local_naive_ns=clock.asi8,slot_seconds=np.float64(STEP_SECONDS))
    counts=edge_mask.sum(axis=1).astype(int)
    byslot=pd.DataFrame({'time_local':clock.strftime('%Y-%m-%d %H:%M:%S'),
                         'mapped_observed_edges':counts,
                         'mapped_observation_ratio':counts/mapped.sum(),
                         'all_graph_measured_edge_ratio':counts/E})
    byslot.to_csv(costdir/'observed_edge_coverage_by_slot.csv',index=False)
    daily=(byslot.assign(date=clock.date.astype(str)).groupby('date',as_index=False)
           .agg(mean_mapped_observation_ratio=('mapped_observation_ratio','mean'),
                mean_graph_measured_edge_ratio=('all_graph_measured_edge_ratio','mean'),
                slots=('time_local','count')))
    daily.to_csv(costdir/'observed_edge_coverage_by_day.csv',index=False)
    report={'osm_nodes':int(g.number_of_nodes()),'osm_directed_edges':E,
            'monitored_corridors':int(mapping.segment_id.nunique()),
            'mapped_directed_edges':int(mapped.sum()),
            'fraction_graph_edges_mapped':float(mapped.mean()),
            'time_slots':T,'step_minutes':10,
            'observed_mapped_edge_cells':int(edge_mask.sum()),
            'fraction_observed_within_mapped_edge_grid':float(edge_mask[:,mapped].mean()),
            'fraction_observed_within_entire_graph_grid':float(edge_mask.mean()),
            'zero_mapped_observation_slots':int((counts==0).sum()),
            'fifo_values_adjusted_over_1ms':adjusted,
            'fifo_max_added_seconds':largest,
            'observed_mask_meaning':'Chicago corridor speed measured at slot; apportioned across OSM path; NOT individual edge measured speed',
            'unmapped_cost_meaning':'OSM static estimate; not observed truth',
            'planner_warning':'This full July history is ex-post oracle hindsight, NOT accessible at future departure time'}
    save_json(costdir/'dynamic_cost_summary.json',report)
    return report


def build(args):
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}; specify a new --output-dir')
    # Verify all inputs and mapping BEFORE writing any output files.
    _,edges=graph_edges(args.graphml)
    idx,length,mapping,static=load_references(args,edges)
    out.mkdir(parents=True,exist_ok=True)
    try:
        speeds,masks,clock,quality,sources=process_traffic(args,idx,length,out)
        g,edges=graph_edges(args.graphml)
        cost_summary=make_edge_costs(args,out,g,edges,mapping,idx,length,speeds,masks,clock)
        meta={'version':'2.0','script_sha256':sha256(Path(__file__)),'date_start':str(args.start),'date_end':str(args.end),
              'train_end':'2025-07-21','validation_end':'2025-07-25','test_start':'2025-07-26',
              'slots':len(clock),'segments':len(idx),'valid_monitor_cells':int(masks.sum()),
              'monitor_observation_fraction':float(masks.mean()),
              'input_files':sources,'max_valid_mph':args.max_mph,
              'references':{key:{'path':str(v),'sha256':sha256(v)} for key,v in
                            [('graphml',args.graphml),('mapping',args.mapping),
                             ('selected_segments',args.selected_segments),('segment_index',args.segment_index),
                             ('static_segments',args.static_segments)]},
              'edge_cost_summary':cost_summary,
              'note':'No imputation was performed; only observed speeds serve as labels. Static fallback is marked unobserved.'}
        save_json(out/'stage2_manifest.json',meta)
        print(json.dumps({'output':str(out),'slots':len(clock),'segments':len(idx),'edges':len(edges),
                          'observed_speed_cells':int(masks.sum()),'mapped_edge_count':cost_summary['mapped_directed_edges']},
                         indent=2,ensure_ascii=False))
    except BaseException:
        (out/'BUILD_FAILED.txt').write_text('Build interrupted; delete this INCOMPLETE output directory and rerun.\n',encoding='utf-8')
        raise


def validate(args):
    out=args.output_dir
    cost=out/'dynamic_costs'
    report=json.loads((out/'stage2_manifest.json').read_text(encoding='utf-8'))
    idx=pd.read_csv(out/'segment_index.csv')
    times=pd.to_datetime(pd.read_csv(out/'time_index.csv').time_local)
    with np.load(out/'traffic_tensor.npz',allow_pickle=False) as z:
        speed=z['speed_kph'];mon=z['observed_mask'];sids=z['segment_ids'];slot=z['slot_local_naive_ns']
    with np.load(cost/'edge_costs.npz',allow_pickle=False) as z:
        raw=z['travel_sec_raw'];fifo=z['travel_sec_fifo'];obs=z['chicago_observed_mask'];mapped=z['is_mapped_edge'];
        cslot=z['slot_local_naive_ns'];step=float(z['slot_seconds'])
    edges=pd.read_csv(cost/'edge_index.csv',dtype={k:str for k in EDGE},keep_default_na=False)
    assert len(idx)==speed.shape[1]==len(sids), 'Segment index shape mismatch'
    assert np.array_equal(idx.segment_id.to_numpy(np.int64),sids), 'Segment ID order mismatch'
    assert speed.shape==mon.shape==(len(times),len(idx)), 'Speed tensor shape mismatch'
    assert np.array_equal(times.astype('int64').to_numpy(),slot), 'Speed clock mismatch'
    assert np.array_equal(slot,cslot), 'Traffic/road clock mismatch'
    assert np.isclose(step,600), 'Expected 10-minute time step'
    assert np.all(np.diff(slot)==600*1e9), 'Nonuniform clock'
    assert raw.shape==fifo.shape==obs.shape==(len(times),len(edges)), 'OSM edge shape mismatch'
    assert np.array_equal(edges.edge_idx.to_numpy(),np.arange(len(edges))), 'Edge ID order mismatch'
    assert np.isfinite(raw).all() and np.isfinite(fifo).all() and (raw>0).all() and (fifo>0).all(), 'Bad travel time'
    assert np.array_equal(mon,np.isfinite(speed)), 'Speed missing mask not preserved'
    assert not np.any(obs[:,~mapped]), 'Unmapped edge falsely observed'
    assert int(mapped.sum())==report['edge_cost_summary']['mapped_directed_edges'], 'Mapped edges mismatch'
    # All unmapped costs must remain STATIC at every slot.
    if (~mapped).any():
        expected=edges.static_travel_sec.to_numpy(np.float32)[~mapped]
        assert np.array_equal(raw[:,~mapped],np.broadcast_to(expected,(len(times),len(expected)))), 'Unmapped costs changed'
    arrival=np.arange(len(times),dtype=np.float64)[:,None]*step+fifo.astype(np.float64)
    if (np.diff(arrival,axis=0)<-0.015).any():
        raise AssertionError('FIFO arrival monotonicity violated beyond float32 tolerance')
    # Match observed OSM edge data back to source monitoring cells.
    position={int(k):i for i,k in enumerate(sids)}
    for r in edges.loc[mapped].itertuples():
        mid=int(r.mapped_segment_id); ix=position[mid];f=float(r.mapped_path_fraction)
        assert np.array_equal(obs[:,int(r.edge_idx)],mon[:,ix]), 'Observed mask differs from monitored corridor'
    # Faster complete travel-time conservation check by grouping mapped edge arrays.
    mapping=pd.read_csv(args.mapping)
    static=pd.read_csv(args.static_segments).set_index('segment_id')
    for sid,group in mapping.groupby('segment_id'):
        sid=int(sid);valid=mon[:,position[sid]]
        # The parent edge_index maps monitor ID; all mapped directed edges for this sid form its path.
        cols=edges.loc[edges.mapped_segment_id==str(sid),'edge_idx'].to_numpy(int)
        assert len(cols)==len(group),'Mapping path length mismatch'
        seconds=3.6*float(static.loc[sid,'length_m'])/speed[valid,position[sid]]
        if len(seconds):
            max_err=float(np.max(np.abs(raw[np.ix_(np.flatnonzero(valid),cols)].sum(axis=1)-seconds)))
            if max_err > max(0.01,0.0001*float(np.max(seconds))):
                raise AssertionError(f'Travel-time conservation failed sid={sid} err={max_err}')
    outcome={'validation':'PASS','rows':len(times),'monitor_segments':len(idx),'osm_edges':len(edges),
             'mapped_edges':int(mapped.sum()),'valid_monitor_cells':int(mon.sum()),
             'valid_osm_edge_cells':int(obs.sum()),'fifo_min_arrival_increment_sec':float(np.diff(arrival,axis=0).min()),
             'missing_labeled_imputation':False}
    save_json(out/'stage2_validation.json',outcome)
    print(json.dumps(outcome,indent=2,ensure_ascii=False))


def trip(adj, src, dst, depart: float, edgecost, trace: bool=True):
    time={src:depart}; back={};heap=[(depart,src)]
    while heap:
        t,u=heapq.heappop(heap)
        if t>time[u]+1e-9: continue
        if u==dst:
            if not trace:return t-depart,[]
            seq=[];here=dst
            while here!=src:
                p,e=back[here];seq.append(e);here=p
            return t-depart,seq[::-1]
        for v,e in adj.get(u,[]):
            a=t+edgecost(e,t)
            if a+1e-9 < time.get(v,math.inf):
                time[v]=a;back[v]=(u,e);heapq.heappush(heap,(a,v))
    return math.inf,[]


def arrival_linear(array: np.ndarray, edge: int, time: float, step: float) -> float:
    if time<0 or time>=len(array)*step:
        raise ValueError('Vehicle time outside July road cost dataset')
    pos=time/step
    k=min(int(pos),len(array)-2)
    alpha=min(max(pos-k,0.),1.)
    # For last sample, the final slot is held constant (Road.travel legacy semantics).
    if pos>=len(array)-1:
        return float(array[-1,edge])
    a0=k*step+float(array[k,edge]);a1=(k+1)*step+float(array[k+1,edge])
    return max(0., (a0+(a1-a0)*alpha)-time)


def tdsp_smoke(args):
    cost=args.output_dir/'dynamic_costs'
    edges=pd.read_csv(cost/'edge_index.csv',dtype={k:str for k in EDGE})
    times=pd.to_datetime(pd.read_csv(cost/'time_index.csv').time_local)
    with np.load(cost/'edge_costs.npz',allow_pickle=False) as z:
        fifo=z['travel_sec_fifo'];obs=z['chicago_observed_mask'];step=float(z['slot_seconds'])
    adj={}; g=nx.DiGraph()
    for r in edges.itertuples():
        adj.setdefault(r.osm_u,[]).append((r.osm_v,int(r.edge_idx)))
        g.add_edge(r.osm_u,r.osm_v)
    largest=max(nx.strongly_connected_components(g),key=len)
    nodes=sorted(largest)
    if len(nodes)<2:raise ValueError('OSM graph has no nontrivial SCC')
    t0=float((pd.Timestamp(args.departure)-times.iloc[0]).total_seconds())
    if t0 < 0 or t0+(args.max_pair_seconds) >= len(times)*step:
        raise ValueError('Sampled departure outside valid cost horizon')
    base=edges.static_travel_sec.to_numpy(float)
    sample=np.random.default_rng(args.seed)
    lines=[]
    for _ in range(args.pairs):
        src,dst=sample.choice(nodes,size=2,replace=False)
        static,seq=trip(adj,src,dst,t0,lambda i,t:base[i])
        dyn,tdseq=trip(adj,src,dst,t0,lambda i,t:arrival_linear(fifo,i,t,step))
        t=t0
        try:
            for e in seq:t+=arrival_linear(fifo,e,t,step)
        except ValueError:
            raise ValueError('Static-route ex post evaluation exceeds data horizon')
        realized=t-t0
        if dyn>realized+0.05:
            raise AssertionError('Dynamic Dijkstra loses to fixed static path: TDSP/FIFO bug')
        traversals=0;time=t0
        for e in tdseq:
            traversals+=int(obs[min(int(time//step),len(times)-1),e]);time+=arrival_linear(fifo,e,time,step)
        lines.append({'origin':src,'destination':dst,'static_planned_sec':static,
                      'ex_post_static_path_sec':realized,'tdsp_oracle_sec':dyn,
                      'oracle_advantage_sec':realized-dyn,'oracle_path_edges':len(tdseq),
                      'observed_edges_on_oracle_path':traversals})
    result=pd.DataFrame(lines)
    result.to_csv(args.output_dir/'tdsp_smoke.csv',index=False)
    summary={'tdsp':'PASS','departure_local':args.departure,'tested_pairs':len(lines),
             'mean_oracle_advantage_sec':float(result.oracle_advantage_sec.mean()),
             'max_oracle_advantage_sec':float(result.oracle_advantage_sec.max()),
             'fraction_oracle_paths_with_monitored_edges':float((result.observed_edges_on_oracle_path>0).mean()),
             'scc_nodes':len(largest)}
    save_json(args.output_dir/'tdsp_smoke_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    common=sub.add_parser('build',help='Process July CSVs and create TD road costs')
    common.add_argument('--csv-dir',type=Path,default=Path('chicago_data'))
    common.add_argument('--segment-index',type=Path,default=Path('datasets/processed/chicago/segment_index.csv'))
    common.add_argument('--static-segments',type=Path,default=Path('datasets/processed/chicago/segments_static_july.csv'))
    common.add_argument('--mapping',type=Path,default=Path('datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv'))
    common.add_argument('--selected-segments',type=Path,default=Path('datasets/processed/chicago/osm_match_loop/safe_mapping/selected_monitor_segments.csv'))
    common.add_argument('--mapping-summary',type=Path,default=Path('datasets/processed/chicago/osm_match_loop/safe_mapping/safe_mapping_summary.json'))
    common.add_argument('--graphml',type=Path,default=Path('datasets/raw/chicago/chicago_loop_drive.graphml'))
    common.add_argument('--output-dir',type=Path,default=Path('datasets/processed/chicago/july_v1'))
    common.add_argument('--start',type=dt.date.fromisoformat,default=dt.date(2025,7,2))
    common.add_argument('--end',type=dt.date.fromisoformat,default=dt.date(2025,7,31))
    common.add_argument('--max-mph',type=float,default=100.)
    common.add_argument('--hash-files',action='store_true')
    v=sub.add_parser('validate',help='Validate mask, tensor alignment, FIFO, conservation and static fallback')
    v.add_argument('--output-dir',type=Path,default=Path('datasets/processed/chicago/july_v1'))
    v.add_argument('--mapping',type=Path,default=Path('datasets/processed/chicago/osm_match_loop/safe_mapping/safe_osm_edge_mapping.csv'))
    v.add_argument('--static-segments',type=Path,default=Path('datasets/processed/chicago/segments_static_july.csv'))
    smoke=sub.add_parser('tdsp-smoke',help='Compare static road route replay with FIFO oracle TDSP')
    smoke.add_argument('--output-dir',type=Path,default=Path('datasets/processed/chicago/july_v1'))
    smoke.add_argument('--departure',default='2025-07-28 08:00:00')
    smoke.add_argument('--pairs',type=int,default=30)
    smoke.add_argument('--seed',type=int,default=42)
    smoke.add_argument('--max-pair-seconds',type=float,default=3600*5)
    a=p.parse_args(argv)
    if a.command=='build':
        if a.start>a.end or not (0<a.max_mph<=150):raise ValueError('Invalid date interval or speed bound')
        build(a)
    elif a.command=='validate':validate(a)
    elif a.command=='tdsp-smoke':tdsp_smoke(a)
    return 0

if __name__=='__main__':
    sys.exit(main())
