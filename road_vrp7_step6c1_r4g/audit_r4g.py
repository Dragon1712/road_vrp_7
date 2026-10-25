#!/usr/bin/env python3
"""Read-only paired original-C3 vs TRAIN snapshot capture; no training or search."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ('road_vrp7_step6c1_r4g','road_vrp7_step6c1_r4a','road_vrp7_step6c1_r4b'):
    sys.path.insert(0,str(ROOT/name))
from r4a_core import write_json,sha256
from r4g_core import load_capture,paired_audit

def run(a):
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError('New results directory required: '+str(out))
    report=paired_audit(load_capture(a.plain,'plain'),load_capture(a.capture,'capture'))
    # Persist the full diagnostic even on a strict failure. Never alter
    # Plain/Capture source results, relax TRAIN guards, or claim a proof.
    out.mkdir(parents=True)
    report['source_sha256']={str(p):sha256(p) for p in (Path(a.plain)/'manifest.json',Path(a.plain)/'metrics.jsonl',
          Path(a.capture)/'manifest.json',Path(a.capture)/'metrics.jsonl',Path(a.capture)/'snapshots.jsonl')}
    write_json(out/'report.json',report)
    text=['# R4-G TRAIN-only 非干预审计',
          '',f"严格结论：**{report['strict_verdict']}**；固定迭代通过：**{report['strict_pass']}/{report['pairs']}**；路线和道路成本一致：**{report['route_road_equal']}/{report['pairs']}**。",
          f"快照：**{report['n_snapshots']}**；去重物理搜索状态：**{report['n_distinct_physical_states']}**；跨场景重复状态：**{report['multi_scenario_physical_states']}**。",
          '', '## 失败原因统计',
          '',f"失败配对：{report['n_failed_pairs']}；原因计数：{json.dumps(report['failure_reason_counts'],ensure_ascii=False)}。",
          '', '| Task | Scenario | Seed | Plain/Capture迭代 | Plain/Capture停止原因 | 路线相同 | Road差(Capture−Plain) | 强一致 | 失败原因 |',
          '|---|---|---:|---|---|---|---:|---|---|']
    for r in report['details']:
        text.append(f"| {r['task_id']} | {r['scenario_id']} | {r['seed']} | {r['plain_iterations']}/{r['capture_iterations']} | {r['plain_termination_reason']}/{r['capture_termination_reason']} | {r['route_equal']} | {r['road_delta_capture_minus_plain_sec']:+.3f} | {r['strong_pass']} | {','.join(r['failure_reasons']) or '—'} |")
    text.extend(['','## 判读边界','',
        '- ITERATION_CAP_NOT_MET：至少一个分支未跑满固定迭代上限，不能证明固定迭代无干预，不自动等于算法逻辑已改变。',
        '- FINAL_ROUTE_MISMATCH：末端路线不同；如都达到迭代上限，需要进一步检查算子/Repair的时间相关分支。',
        '- ROAD_COST_MISMATCH：Road.replay成本不同，不能忽略。',
        '- 这份报告只诊断Train捕获；不是恢复证明，也不是30秒性能改善。',
        '- 失败配对的状态不得作为已严格验证的无干预捕获。'])
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print('R4G AUDIT',report['strict_verdict'],report['strict_pass'],'/',report['pairs'],
          'failures',report['failure_reason_counts'],'report:',out/'report.md',flush=True)
    if a.require_strict and not report['can_claim_fixed_iteration_noninterference']:
        raise ValueError('Fixed-iteration noninterference NOT fully verified; see '+str(out/'report.md'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plain',required=True);p.add_argument('--capture',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--require-strict',action='store_true');run(p.parse_args())
