#!/usr/bin/env python3
"""Check a proposal locally without snapshots, GPU allocation or task submission."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.meta_protocol import strict_object
from llmgo.rsi.proposal import validate_workspace_proposal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,default=Path('proposal.json'))
    parser.add_argument('--require-training',action='store_true')
    args=parser.parse_args()
    try:
        data=validate_workspace_proposal(args.proposal.resolve().parent,strict_object(args.proposal.read_text()),require_training=args.require_training)
        print(json.dumps({'ok':True,'jobs':len(data.get('jobs',[])),'proposal':str(args.proposal.resolve()),'submitted':False}))
    except (ValueError,OSError) as error:parser.exit(2,str(error)+'\n')


if __name__=='__main__':main()
