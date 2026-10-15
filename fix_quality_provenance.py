#!/usr/bin/env python3
"""One-time source metadata fix for the existing July QC script in road_vrp_7.
No old data is read. Idempotent and conservative.
"""
from pathlib import Path
import argparse

def fix(path:Path):
    s=path.read_text(encoding='utf-8')
    old='"input_git_repo":"Dragon1712/road_vrp","reference_git_commit":"e64d6f1118ba910d983ff07aa1bdc892ccdb409c"'
    new='"input_git_repo":"Dragon1712/road_vrp_7","reference_git_commit":None,"mapping_origin":"independent_July_CSVs_and_fresh_OSM"'
    if old in s:
        path.write_text(s.replace(old,new),encoding='utf-8')
        print('PATCHED new-project provenance:',path)
    elif new in s:
        print('Already updated:',path)
    else:
        raise RuntimeError(f'Unexpected manifest template in {path}; inspect manually, no changes made')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--file',type=Path,default=Path('chicago_july_quality_tool/batch_quality_select.py'))
    fix(p.parse_args().file)
