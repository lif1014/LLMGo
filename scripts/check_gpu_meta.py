#!/usr/bin/env python3
"""Explicit one-GPU thinking/tool acceptance; no campaign baseline or Student training."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.contracts import atomic_json,load_site
from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.meta_jobs import GpuMetaBackend

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--site',required=True)
    parser.add_argument('--name',default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    args=parser.parse_args();site=load_site(args.site);site['meta_engineering_check']=True
    root=Path(site['repo'])/'var/operator-checks'/('qwen-meta-'+args.name)
    workspace=root/'workspace';workspace.mkdir(parents=True,exist_ok=True)
    atomic_json(workspace/'context.json',{'engineering_check':True})
    result=GpuMetaBackend(site,MemoryStore(root)).turn(workspace,'GPU Meta tool acceptance only; no research or training')
    print(result)
if __name__=='__main__':main()
