#!/usr/bin/env python3
"""Download and prepare the optional full Lichess training-only pool."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.data.lichess import download, prepare, verify, write_json


def main():
    root = Path('/user/lyh4170/LLMGo!')
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--raw', type=Path, help='Use an existing official CSV.zst instead of downloading')
    parser.add_argument('--proxy', help='Download proxy; defaults to the process http_proxy/https_proxy settings')
    parser.add_argument('--reference', type=Path, default=root/'assets/datasets/chess-r1/boxed-nonthinking-v2/train.parquet')
    parser.add_argument('--heldout', action='append', type=Path, help='Repeat for dev and test; defaults to the current frozen files')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--progress', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    heldouts = args.heldout or [args.reference.with_name('dev.parquet'), args.reference.with_name('test.parquet')]
    output = args.snapshot/'boxed-nonthinking-v2'
    if not args.verify_only:
        raw = args.raw or args.snapshot/'raw/lichess_db_puzzle.csv.zst'
        if args.raw is None:
            print(json.dumps(download(raw, proxy=args.proxy)), flush=True)
        result = prepare(raw, output, args.reference, heldouts, workers=args.workers, progress=args.progress)
        print(json.dumps({'stage':'prepared','counts':result['counts']}), flush=True)
    if args.progress:
        write_json(args.progress, {'status':'verifying','output':str(output)})
    proof = verify(output, args.reference, heldouts)
    if args.progress:
        write_json(args.progress, proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
