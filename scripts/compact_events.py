#!/usr/bin/env python3
"""Backfill portable logs without deleting or rewriting raw events."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llmgo.store.compact import TokenWindows, backfill

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--tokenizer', type=Path, help='Existing model directory or tokenizer.json')
    args = parser.parse_args()
    windows = TokenWindows(args.tokenizer) if args.tokenizer else None
    if args.tokenizer and windows.tokenizer is None:
        parser.error(windows.error)
    sources = [args.runs / 'events.jsonl'] if (args.runs / 'events.jsonl').is_file() else sorted(args.runs.rglob('events.jsonl'))
    for source in sources:
        print(json.dumps({'run': str(source.parent), **backfill(source.parent, windows)}, ensure_ascii=False))
