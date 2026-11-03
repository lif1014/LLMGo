#!/usr/bin/env python3
"""Record IDs, statistics and hypothesis for retaining existing training files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.data.lichess import report_training_data


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--train',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hypothesis',required=True)
    parser.add_argument('--evidence',action='append',default=[])
    parser.add_argument('--query',default='use existing filtered training data')
    args=parser.parse_args()
    print(json.dumps(report_training_data(args.train,args.output,hypothesis=args.hypothesis,
                                         evidence=args.evidence,query=args.query)),flush=True)


if __name__=='__main__':
    main()
