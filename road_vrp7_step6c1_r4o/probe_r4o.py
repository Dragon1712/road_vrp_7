#!/usr/bin/env python3
"""Source-pinned R4-O decision-time route context benchmark, TRAIN only."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ('road_vrp7_step6c1_r4n','road_vrp7_step6c1_r4m','road_vrp7_step6c1_r4l','road_vrp7_step6c1_r4i','road_vrp7_step6c1_r4k'):
    sys.path.insert(0,str(ROOT/folder))
import r4n_core as n
import r4l_core as l
import r4m_core as m
import r4o_core as o


def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fp:
        for b in iter(lambda:fp.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))

def readl(path):
    with Path(path).open(encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def source_records(a):
    paths={name:getattr(a,name) for name in ('r4l_result','old_cf','old_capture','old_prefix','new_r4k','new_cf','new_capture','new_prefix')}
    records,pins=m.source_preflight(paths)
    old_r4n=Path(a.r4n_result)
    rep=load(old_r4n/'report.json');prov=load(old_r4n/'provenance.json')
    o.need(rep.get('stage')==n.STAGE and rep.get('status')=='COMPLETE' and rep.get('data_role')=='TRAIN_ONLY', 'R4-N source report incomplete/nonTrain')
    o.need(prov.get('report_sha256')==filehash(old_r4n/'report.json') and prov.get('fold_predictions_sha256')==filehash(old_r4n/'fold_predictions.jsonl'), 'R4-N source SHA mismatch')
    o.need(prov.get('source_sha256',{}).get('r4l_report_sha256')==pins['r4l_report_sha256'] and prov.get('source_sha256',{}).get('r4m_report_sha256')==filehash(Path(a.r4m_result)/'report.json'), 'R4-N source pin mismatch')
    o.need(all(filehash(ROOT/'road_vrp7_step6c1_r4n'/k)==v for k,v in prov.get('code_sha256',{}).items()),'R4-N source code SHA differs')
    o.need(len(records)==1572 and len(o.groups(records))==200,'Bad source decision grid')
    return records,pins,{'r4n_report':filehash(old_r4n/'report.json'),'r4n_provenance':filehash(old_r4n/'provenance.json')}


def action_maps(a,records):
    old_snaps=list(readl(Path(a.old_capture)/'snapshots.jsonl'))
    new_snaps=list(readl(Path(a.new_capture)/'snapshots.jsonl'))
    old_actions={}
    for row in readl(Path(a.old_cf)/'labels.jsonl'):
        z=row['unit_spec'];k=o.sid(z);name=str(z['arm'])
        if int(z['horizon'])!=50:continue
        o.need((k,name) not in old_actions,'Duplicate original K50 candidate identity')
        old_actions[k,name]=row['first_action_id']
    new_actions={}
    for r in records:
        if r['source']!='new_s62s63':continue
        k=tuple(r['snap']);name=r['arm'];path=l.unit_file(a.new_cf,k,50,name)
        o.need(path.is_file(),'New K50 source unit absent '+str(path))
        unit=load(path)
        o.need(unit['unit_spec']==l.unit_spec(k,50,name) and unit['result']['arm']==name and unit['result']['horizon']==50,'Bad new unit identity')
        o.need((k,name) not in new_actions,'Duplicate new K50 unit')
        new_actions[k,name]=unit['result']['first_action_id']
    o.need(len(old_actions)==565 and len(new_actions)==1007,'Source candidate identity count mismatch')
    return old_snaps,new_snaps,old_actions,new_actions


def main(a):
    out=Path(a.output_dir) if a.output_dir else None
    if a.mode=='probe':o.need(out is not None and not out.exists(),'Use NEW independent output directory')
    records,pins,extra=source_records(a)
    print('R4O upstream pinned: 1572 alternatives/200 TRAIN snapshots, R4-L/M/N SHAs match',flush=True)
    if a.mode=='preflight':return
    old_snaps,new_snaps,old_actions,new_actions=action_maps(a,records)
    enriched=o.attach_context(records,old_snaps,new_snaps,old_actions,new_actions)
    splits=l.make_splits(enriched)
    o.need(len(splits)==6,'Expected 4 held-out tasks + 2 seed transfer splits')
    folds,details=o.evaluate(enriched,splits)
    o.need(len(folds)==6*3*2*len(o.POLICIES),'Unexpected fold count')
    upstream={**pins,**extra,'old_capture_snapshot_bytes':filehash(Path(a.old_capture)/'snapshots.jsonl'),
       'new_capture_snapshot_bytes':filehash(Path(a.new_capture)/'snapshots.jsonl'),
       'old_candidate_labels_bytes':filehash(Path(a.old_cf)/'labels.jsonl'),
       'new_cf_manifest_bytes':filehash(Path(a.new_cf)/'manifest.json')}
    doc={'stage':o.STAGE,'status':'COMPLETE','data_role':'TRAIN_ONLY',
      'semantics':'OFFLINE_FIXED_K500_NOT_REAL_30_SECOND_ONLINE',
      'reviewed_git_commit':'98c16e14a28a4edb16ba6145d1af7d68aad2cbec',
      'n_actions':len(enriched),'n_scenario_snapshots':len(o.groups(enriched)),
      'n_coarse_search_starts':len({(r['task'],r['snap'][2],r['snap'][3]) for r in enriched}),
      'protocol':{'splits':[x[0] for x in splits],'feature_modes':list(o.FEATURE_MODES),
         'policies':list(o.POLICIES),'non_deployable_oracle_policies':list(o.ORACLE_POLICIES),
         'shuffle_whole_train_state_outcomes':True,'gate_probability_threshold':.5,
         'models':'weighted linear pairwise ranking; independent task-balanced binary gate; frozen R4-L Ridge reference',
         'traffic_data_used_as_feature':False,'road_replay_cost_as_feature':False},
      'source_sha256':upstream,'folds':folds,
      'limitations':['Train exploratory only: 4 original tasks','Actions and scenarios share search starts; grouped splitting',
         'Oracle strategies use heldout future labels; never deploy','No actual 30-second online gain',
         'Route topology is not a traffic modality; no claim of multimodal gain',
         'All feature variants are investigative, no heldout model selection']}
    out.mkdir(parents=True,exist_ok=False)
    n.save(out/'report.json',doc)
    (out/'report.md').write_text(render(doc),encoding='utf8')
    with (out/'fold_predictions.jsonl').open('w',encoding='utf8') as f:
        for row in details:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    n.save(out/'provenance.json',{'stage':o.STAGE,
       'report_sha256':filehash(out/'report.json'),'predictions_sha256':filehash(out/'fold_predictions.jsonl'),
       'source_sha256':upstream,
       'code_sha256':{'r4o_core.py':filehash(Path(__file__).with_name('r4o_core.py')),
                      'probe_r4o.py':filehash(__file__)}})
    print('R4O complete',out,'folds',len(folds),flush=True)


def render(doc):
    lines=['# R4-O Train-only Section路线结构排序与独立委托门控','',
        '结果仅为冻结Train反事实下的固定K=500离线收益；非30秒在线或独立Validation/Test。','',
        f"场景快照：{doc['n_scenario_snapshots']}；候选：{doc['n_actions']}；粗粒度搜索起点：{doc['n_coarse_search_starts']}。",'',
        '| 留组 | 特征 | 策略 | 委托 | 正/负 | 平均K500收益(s) | 正收益Top1/机会 | 同状态随机Top1期望 |',
        '|---|---|---|---:|---:|---:|---:|']
    for x in doc['folds']:
        if x['shuffle_train_labels'] or x['policy'] not in ('C3','ridge_ref','pairwise_ridge_gate','learned_gate_pairwise','oracle_gate_pairwise','sampled_oracle'):continue
        if x['feature_mode'] not in ('baseline15','route_context','route_context_interact'):continue
        if x['policy'] in ('C3','ridge_ref') and x['feature_mode']!='baseline15':continue
        lines.append(f"| {x['split']} | {x['feature_mode']} | {x['policy']}{' (*)' if x['oracle_conditioned'] else ''} | {x['delegations']} | {x['positive']}/{x['negative']} | {x['mean_gain_sec']:+.3f} | {x['top1_positive_hits']}/{x['positive_oracle_states']} | {x['random_expected_hits']:.2f} |")
    lines.extend(['','(*) 事后Oracle策略读取留出未来收益，**不可部署**。','',
     '没有使用事后Road.replay成本作为模型特征；Route Context由原始当前/历史最优路线和候选Section边界构成。',
     'Shuffle训练对照仅更改训练数据；不同状态/动作不是独立样本。','',
     '解释重点：真实状态门控是否比原Ridge门控提高正收益覆盖且不增加损失；Route Context与相同留组/相同策略的base15相比是否改善Top1正收益命中。',''])
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','probe'),default='probe')
    p.add_argument('--r4n-result',default='runs/chicago_july_v1/step6c1_r4n/r4n_rank_gate_train_v1')
    p.add_argument('--r4m-result',default='runs/chicago_july_v1/step6c1_r4m/r4m_nested_risk_train_v1')
    p.add_argument('--r4l-result',default='runs/chicago_july_v1/step6c1_r4l/r4l_seed_transfer_loto_train_v1')
    p.add_argument('--old-cf',default='runs/chicago_july_v1/step6c1_r4h/r4h_train_full_v1')
    p.add_argument('--old-capture',default='runs/chicago_july_v1/step6c1_r4g/r4g_train16_fix2_capture')
    p.add_argument('--old-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4g_prefix_all16_v1')
    p.add_argument('--new-r4k',default='runs/chicago_july_v1/step6c1_r4k/r4k_full_128audited_v1')
    p.add_argument('--new-cf',default='runs/chicago_july_v1/step6c1_r4h/r4j_train_early_s62s63_v1_cf_full')
    p.add_argument('--new-capture',default='runs/chicago_july_v1/step6c1_r4g/r4j_train_early_s62s63_v1_capture')
    p.add_argument('--new-prefix',default='runs/chicago_july_v1/step6c1_r4g_prefix/r4j_train_early_s62s63_v1_prefix_full')
    p.add_argument('--output-dir')
    a=p.parse_args();main(a)
