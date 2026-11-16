"""Quota correctness with overlapping tags and actual compatible Parquet exports."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard

from llmgo.data.lichess import prepare, verify
from llmgo.data.lichess_selection import inspect_pool, select_mix
from test_lichess_data import SCHEMA, sample


@pytest.fixture
def pool(tmp_path):
    # The first puzzle belongs to both groups; unequal trajectory lengths expose
    # an accidental position-level rather than puzzle-level quota implementation.
    specifications = [
        ('both', 1000, 'fork pin', 'a2a3 a7a6 b2b3 b7b6'),
        ('fork-only', 1100, 'fork', 'b2b3 a7a6'),
        ('pin-only', 1200, 'pin endgame', 'c2c3 a7a6 d2d3 b7b6'),
        ('hard', 2200, 'fork pin', 'd2d3 a7a6'),
        ('other', 900, 'mateIn1', 'e2e3 a7a6'),
    ]
    rows = [{'PuzzleId': pid, 'FEN': chess.STARTING_FEN, 'Moves': moves,
             'Rating': rating, 'RatingDeviation': 50, 'Popularity': 90,
             'NbPlays': 100, 'Themes': themes, 'OpeningTags': 'Test_Opening'}
            for pid, rating, themes, moves in specifications]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    raw = tmp_path/'raw.csv.zst'
    raw.write_bytes(zstandard.ZstdCompressor().compress(text.getvalue().encode()))
    reference = tmp_path/'reference.parquet'
    pq.write_table(pa.Table.from_pylist([], schema=SCHEMA), reference)
    heldout = tmp_path/'heldout.parquet'
    pq.write_table(pa.Table.from_pylist([sample('held', ['e2e4', 'e7e5'])], schema=SCHEMA), heldout)
    output = tmp_path/'pool'
    prepare(raw, output, reference, [heldout])
    verify(output, reference, [heldout])
    return output


def test_mixture_assigns_overlap_once_and_keeps_all_solver_positions(pool):
    groups = [{'name': 'fork', 'weight': 1}, {'name': 'pin', 'weight': 2}]
    output = pool.parent/'mix'
    result = select_mix(pool, output, groups=groups, num_puzzles=3,
                        rating_min=800, rating_max=1600,
                        hypothesis='Increase forks and pins after losses', evidence=['game/events.jsonl'])
    assert result['status'] == 'selected'
    assert result['puzzles'] == 3 and result['positions'] == 5
    assignments = json.loads((output/'selected_groups.json').read_text())
    assert assignments == {'fork-only': 'fork', 'both': 'pin', 'pin-only': 'pin'}
    assert result['groups']['fork']['assigned_puzzles'] == 1
    assert result['groups']['pin']['assigned_puzzles'] == 2
    assert result['groups']['pin']['positions'] == 4
    assert set((output/'selected_ids.txt').read_text().split()) == set(assignments)
    selected = (output/'selected_positions.txt').read_text().split()
    assert len(selected) == len(set(selected)) == 5
    assert 'both:1' in selected and 'both:3' in selected
    stats = json.loads((output/'stats.json').read_text())
    assert stats['puzzle_themes']['fork'] == 2
    assert stats['puzzle_themes']['pin'] == 2
    assert stats['themes']['fork'] == 3
    assert pq.read_schema(output/'train.parquet').equals(pq.read_schema(pool/'train.parquet'))
    source = [r for r in pq.read_table(pool/'train.parquet').to_pylist()
              if r['extra_info']['puzzle_id'] in assignments]
    assert pq.read_table(output/'train.parquet').to_pylist() == source
    assert result['evidence'] == ['game/events.jsonl']
    assert 'seed' not in result['query']


def test_preview_and_catalog_are_read_only_and_count_puzzles(pool):
    before = {p: p.stat().st_mtime_ns for p in pool.rglob('*')}
    preview, chosen = inspect_pool(pool, groups=[{'name': 'fork'}, {'name': 'pin'}],
                                   num_puzzles=3, rating_max=1600)
    assert chosen == {} and preview['status'] == 'feasible'
    assert sum(g['target_puzzles'] for g in preview['groups'].values()) == 3
    assert preview['catalog']['puzzles'] == 4
    assert preview['catalog']['themes']['fork'] == 2
    assert before == {p: p.stat().st_mtime_ns for p in pool.rglob('*')}


def test_shortage_reports_joint_overlap_without_export(pool):
    # Each group individually has two eligible puzzles, but the same two cannot
    # fill four slots. This must not silently duplicate puzzles or change ratios.
    groups = [{'name': 'fork', 'weight': 1}, {'name': 'pin', 'weight': 1}]
    output = pool.parent/'shortage'
    result = select_mix(pool, output, groups=groups, num_puzzles=4,
                        rating_max=1600, hypothesis='Check joint capacity')
    assert result['status'] == 'insufficient'
    assert result['distinct_eligible_puzzles'] == 3
    assert sum(g['shortfall'] for g in result['groups'].values()) == 1
    assert not (output/'train.parquet').exists()
    assert json.loads((output/'stats.json').read_text())['exported'] is False


def test_per_group_rating_and_any_all_excluded_tags(pool):
    groups = [
        {'name': 'easy-forks', 'themes_all': ['fork'], 'themes_exclude': ['pin'],
         'rating_min': 1000, 'rating_max': 1150},
        {'name': 'endgames', 'themes_any': ['endgame', 'mateIn1'],
         'openings_any': ['Test_Opening'], 'rating_min': 1100, 'rating_max': 1500},
    ]
    result = select_mix(pool, pool.parent/'ranges', groups=groups, num_puzzles=2,
                        hypothesis='Two independent difficulty bands')
    assert result['puzzles'] == 2
    assert (pool.parent/'ranges/selected_ids.txt').read_text().split() == ['fork-only', 'pin-only']


def test_cli_mix_json_and_preview(pool):
    script = Path(__file__).resolve().parents[1]/'scripts/select_lichess_data.py'
    common = [sys.executable, str(script), '--pool', str(pool)]
    preview = subprocess.run(common + ['--mix', 'fork=1', 'pin=2', '--num-puzzles', '3',
                                     '--rating-max', '1600', '--preview'], capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert json.loads(preview.stdout)['groups']['pin']['target_puzzles'] == 2
    spec = pool.parent/'spec.json'
    spec.write_text(json.dumps({'num_puzzles': 3, 'rating_max': 1600,
                               'groups': [{'name': 'fork', 'weight': 1}, {'name': 'pin', 'weight': 2}]}))
    exported = subprocess.run(common + ['--spec', str(spec), '--output', str(pool.parent/'cli'),
                                       '--hypothesis', 'CLI mixture'], capture_output=True, text=True)
    assert exported.returncode == 0, exported.stderr
    assert json.loads(exported.stdout)['positions'] == 5
    catalog = subprocess.run(common + ['--list-tags', '--rating-max', '1600'],
                             capture_output=True, text=True)
    assert catalog.returncode == 0, catalog.stderr
    assert json.loads(catalog.stdout)['catalog']['themes']['fork'] == 2
    shortage = subprocess.run(common + ['--mix', 'missingTag=1', '--num-puzzles', '1', '--preview'],
                              capture_output=True, text=True)
    assert shortage.returncode == 2
    assert json.loads(shortage.stdout)['status'] == 'insufficient'
