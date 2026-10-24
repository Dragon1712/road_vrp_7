#!/usr/bin/env python3
"""Regenerate source hash manifest for VQ4 package without self-reference."""
from pathlib import Path
import json,hashlib
root=Path(__file__).parent
names=sorted([str(p.relative_to(root)) for p in root.glob('*.py') if p.name!='build_manifest.py']+[str(p.relative_to(root)) for p in root.glob('*.md')]+['run_vq4.sh']+[str(p.relative_to(root)) for p in root.glob('tests/test_*.py')])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta={'stage':'R3-VQ4-NESTED-TRAIN-SIZE-v1','predecessor':'R3-VQ3-DECISION-RISK-LOTO-v1',
      'predecessor_collector':'R3-VQ-train-only-restart-v1',
      'notice':'Only code and synthetic tests; real Chicago outcomes are not part of this package.',
      'files_sha256':{k:sha(root/k) for k in names}}
(root/'MANIFEST.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
