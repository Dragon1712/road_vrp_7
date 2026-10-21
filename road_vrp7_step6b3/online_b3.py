#!/usr/bin/env python3
"""Use original B2-R 30-second benchmark, only preloading B3 model classes.

Usage: python road_vrp7_step6b3/online_b3.py --control learned \
 --model-mode F4 --model-method P2 --model-path <B3 artifact> [B2-R args].
"""
import sys,runpy
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
# import model to enable trusted B3 pickle class deserialization.
import model  # noqa: F401
fn=BASE/'road_vrp7_step6b2_r/benchmark_30s_r.py'
if not fn.is_file():raise FileNotFoundError(fn)
sys.argv[0]=str(fn)
runpy.run_path(str(fn),run_name='__main__')
