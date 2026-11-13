#!/usr/bin/env python3
"""Isolated on-demand GPU benchmark, never initializes or resumes a study."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.contracts import atomic_json,load_site
from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.meta_jobs import GpuMetaBackend


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--site',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--long-context',action='store_true')
    parser.add_argument('--repetitions',type=int,default=3,choices=range(1,4))
    args=parser.parse_args();site=load_site(args.site)
    site.update(meta_benchmark=True,meta_benchmark_repetitions=args.repetitions,meta_long_context_check=args.long_context)
    root=Path(site['repo'])/'var/operator-checks'/args.name
    workspace=root/'workspace';workspace.mkdir(parents=True,exist_ok=True)
    atomic_json(workspace/'context.json',{'engineering_benchmark':True})
    result=GpuMetaBackend(site,MemoryStore(root)).turn(workspace,'Isolated Meta capability benchmark only; never start Student work')
    print(result)


if __name__=='__main__':main()
