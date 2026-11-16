#!/usr/bin/env python3
"""Prepare a separate Phase 2 source; does not start a campaign or submit jobs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.contracts import inside, read_json
from llmgo.rsi.phase2 import prepare_source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,
        default=Path(__file__).resolve().parents[1]/'configs/rsi/site.phase2-deepseek.json')
    parser.add_argument('--target',type=Path,required=True)
    args=parser.parse_args();site=read_json(args.site)
    target=inside(site['user_root'], args.target)
    print(prepare_source(site['repo'],target,args.site))


if __name__=='__main__':
    main()
