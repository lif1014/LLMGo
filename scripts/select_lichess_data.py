#!/usr/bin/env python3
"""Create a training subset and the Meta-Agent's selection evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.data.lichess import select


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pool', type=Path, default=Path('/user/lyh4170/LLMGo!/assets/datasets/lichess/20261103/boxed-nonthinking-v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hypothesis', required=True)
    parser.add_argument('--evidence', action='append', default=[])
    parser.add_argument('--rating-min', type=int)
    parser.add_argument('--rating-max', type=int)
    parser.add_argument('--theme', action='append', default=[])
    parser.add_argument('--opening', action='append', default=[])
    parser.add_argument('--puzzle-ids', type=Path)
    parser.add_argument('--sample-ids', type=Path)
    parser.add_argument('--max-positions', type=int)
    args = parser.parse_args()
    print(json.dumps(select(args.pool, args.output, hypothesis=args.hypothesis,
        rating_min=args.rating_min, rating_max=args.rating_max, themes=args.theme,
        openings=args.opening, puzzle_ids=args.puzzle_ids, sample_ids=args.sample_ids,
        max_positions=args.max_positions, evidence=args.evidence)), flush=True)


if __name__ == '__main__':
    main()
