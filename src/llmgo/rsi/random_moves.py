"""Derived counts of executed random fallback moves, independent of parsing policy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .contracts import atomic_json, utc


def count_game(path):
    """Count executed moves once; a fallback event is diagnostic, not a second move."""
    path = Path(path)
    size = path.stat().st_size
    data = path.read_bytes()[:size]
    end = data.rfind(b'\n') + 1
    student, opponent = Counter(), Counter()
    student_moves = opponent_moves = 0
    completed = aborted = False
    for line in data[:end].splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get('type')
        if kind == 'student_move':
            student_moves += 1
            if event.get('source') == 'random':
                student[event.get('fallback_reason') or 'unspecified'] += 1
        elif kind == 'opponent_move':
            opponent_moves += 1
            reason = event.get('diagnostics', {}).get('fallback_reason')
            if reason:
                opponent[reason] += 1
        elif kind == 'episode_finished':
            completed = True
        elif kind == 'episode_aborted':
            aborted = True
    return {'student_random_moves':sum(student.values()), 'opponent_random_moves':sum(opponent.values()),
            'student_fallback_reasons':dict(student), 'opponent_fallback_reasons':dict(opponent),
            'student_moves':student_moves, 'opponent_moves':opponent_moves,
            'completed':completed, 'aborted':aborted, 'cursor':end,
            'events_bytes':size, 'events_sha256':hashlib.sha256(data[:end]).hexdigest()}


def write_report(final_test):
    final_test = Path(final_test)
    games = {path.parent.name:count_game(path) for path in sorted((final_test/'games').glob('*/events.jsonl'))}
    report = {'schema_version':1, 'generated':utc(), 'definition':'executed_random_fallback_moves',
              'note':'Counts use recorded actions, not a reparse with the current answer parser.',
              'games':games, 'student_total':sum(g['student_random_moves'] for g in games.values()),
              'opponent_total':sum(g['opponent_random_moves'] for g in games.values())}
    atomic_json(final_test/'random-moves.json',report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs_root', type=Path)
    parser.add_argument('--run', help='One run ID; omitted scans all existing final-test directories')
    args = parser.parse_args(argv)
    folders = [args.runs_root/args.run/'final-test'] if args.run else sorted(args.runs_root.glob('*/final-test'))
    for folder in folders:
        if not (folder/'games').is_dir():
            continue
        report = write_report(folder)
        print(json.dumps({'run':folder.parent.name, 'games':len(report['games']),
                          'student_random_moves':report['student_total'], 'opponent_random_moves':report['opponent_total']}))
    return 0
