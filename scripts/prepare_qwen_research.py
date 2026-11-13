#!/usr/bin/env python3
"""Prepare a clean input tree; does not start research or allocate GPUs."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.clean_start import prepare
from llmgo.rsi.contracts import read_json,inside

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,default=Path(__file__).resolve().parents[1]/'configs/rsi/site.qwen38-meta.json')
    parser.add_argument('--target',type=Path)
    args=parser.parse_args();site=read_json(args.site)
    target=inside(site['user_root'],args.target or Path(site['repo'])/'var/clean-starts/qwen38-27b-meta-v1')
    print(prepare(site['repo'],target,args.site))
if __name__=='__main__':main()
