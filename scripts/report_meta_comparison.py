#!/usr/bin/env python3
"""Report case-paired results from two independently adopted Meta benchmarks."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.contracts import read_json,atomic_json
from llmgo.rsi.meta_benchmark import compare

p=argparse.ArgumentParser(__doc__)
p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=compare(read_json(a.left),read_json(a.right));atomic_json(a.output,r)
print(r)
