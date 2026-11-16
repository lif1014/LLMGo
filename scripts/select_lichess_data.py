#!/usr/bin/env python3
"""Optional Lichess selector: filters, puzzle mixtures, tag catalog and preview."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.data.lichess import select
from llmgo.data.lichess_selection import groups_from_mix, inspect_pool, select_mix


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pool', type=Path, default=Path('/user/lyh4170/LLMGo!/assets/datasets/lichess/20261103/boxed-nonthinking-v2'))
    parser.add_argument('--output', type=Path, help='New experiment directory (required for export)')
    parser.add_argument('--hypothesis', help='Selection hypothesis (required for export)')
    parser.add_argument('--evidence', action='append', default=[])
    parser.add_argument('--rating-min', type=int)
    parser.add_argument('--rating-max', type=int)
    parser.add_argument('--theme', action='append', default=[], help='Global themes: require all repeated values')
    parser.add_argument('--opening', action='append', default=[], help='Global openings: match any repeated value')
    parser.add_argument('--puzzle-ids', type=Path)
    parser.add_argument('--sample-ids', type=Path)
    parser.add_argument('--max-positions', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--mix', nargs='+', metavar='THEME=WEIGHT', help='Puzzle ratios, e.g. fork=1 pin=2')
    mode.add_argument('--spec', type=Path, help='JSON with num_puzzles and groups; per-group filters supported')
    parser.add_argument('--num-puzzles', type=int, help='Total puzzle quota for a mixture')
    parser.add_argument('--list-tags', action='store_true', help='Read-only tag/rating catalog, in puzzle counts')
    parser.add_argument('--preview', action='store_true', help='Read-only quota feasibility; no export or training')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text()) if args.spec else {}
    groups = groups_from_mix(args.mix) if args.mix else spec.get('groups', [])
    total = args.num_puzzles if args.num_puzzles is not None else spec.get('num_puzzles')
    filters = dict(rating_min=args.rating_min if args.rating_min is not None else spec.get('rating_min'),
                   rating_max=args.rating_max if args.rating_max is not None else spec.get('rating_max'),
                   themes=args.theme, openings=args.opening)
    if (args.mix or args.spec) and (not groups or total is None):
        parser.error('Mixtures require groups and --num-puzzles (or num_puzzles in --spec)')
    if args.num_puzzles is not None and not groups:
        parser.error('--num-puzzles requires --mix or --spec')
    if (groups or args.list_tags or args.preview) and any(
            x is not None for x in (args.puzzle_ids, args.sample_ids, args.max_positions)):
        parser.error('ID/position-limit options are for legacy filter export only')
    if args.list_tags or args.preview:
        result, _ = inspect_pool(args.pool, groups=groups, num_puzzles=total, **filters)
    else:
        if args.output is None or not args.hypothesis:
            parser.error('Export requires --output and --hypothesis')
        if groups:
            result = select_mix(args.pool, args.output, groups=groups, num_puzzles=total,
                                hypothesis=args.hypothesis, evidence=args.evidence, **filters)
        else:
            result = select(args.pool, args.output, hypothesis=args.hypothesis, evidence=args.evidence,
                            puzzle_ids=args.puzzle_ids, sample_ids=args.sample_ids,
                            max_positions=args.max_positions, **filters)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 2 if result['status'] == 'insufficient' else 0


if __name__ == '__main__':
    raise SystemExit(main())
