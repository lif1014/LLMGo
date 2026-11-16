"""Puzzle-level mixtures over the verified, training-only Lichess index.

Weights count puzzles, not expanded solver positions. A small flow network assigns
multi-label puzzles to exactly one quota, without starving a feasible mixture.
"""
from __future__ import annotations

import collections
from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile

import pyarrow.parquet as pq

from llmgo.data.lichess import select, sha, write_json


def groups_from_mix(values):
    groups = []
    for value in values:
        name, weight = value.rsplit('=', 1)
        groups.append({'name': name, 'weight': weight, 'themes_any': [name]})
    return groups


def normalize_groups(groups):
    result = []
    for original in groups:
        group = dict(original)
        name = group['name'].strip()
        weight = Fraction(str(group.get('weight', 1)))
        if not name or weight <= 0 or name in {g['name'] for g in result}:
            raise ValueError('Group names must be unique and weights positive')
        if not any(k in group for k in ('themes_all', 'themes_any', 'themes_exclude', 'openings_any')):
            group['themes_any'] = [name]
        for key in ('themes_all', 'themes_any', 'themes_exclude', 'openings_any'):
            if isinstance(group.get(key), str):
                raise ValueError(f'{key} must be a list of tag strings')
            group[key] = list(group.get(key, []))
            if any(not isinstance(tag, str) for tag in group[key]):
                raise ValueError(f'{key} must contain tag strings')
        if group.get('rating_min') is not None and group.get('rating_max') is not None:
            if group['rating_min'] > group['rating_max']:
                raise ValueError('rating_min exceeds rating_max')
        group.update(name=name, weight=str(weight))
        result.append(group)
    if not result:
        raise ValueError('Specify at least one mixture group')
    return result


def quotas(groups, total):
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError('num_puzzles must be positive')
    weights = [Fraction(g['weight']) for g in groups]
    exact = [total * w / sum(weights) for w in weights]
    counts = [int(x) for x in exact]
    order = sorted(range(len(groups)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def _matches(row, group):
    rating = row['rating']
    if group.get('rating_min') is not None and rating < group['rating_min']:
        return False
    if group.get('rating_max') is not None and rating > group['rating_max']:
        return False
    themes, openings = set(row['themes']), set(row['opening_tags'])
    return (set(group.get('themes_all', ())).issubset(themes)
            and (not group.get('themes_any') or themes.intersection(group['themes_any']))
            and not themes.intersection(group.get('themes_exclude', ()))
            and (not group.get('openings_any') or openings.intersection(group['openings_any'])))


def _scan(pool, groups, total, filters, collect):
    """Count unique puzzles; reservoir-sample each exact eligibility combination."""
    counts = collections.Counter()
    reservoirs = collections.defaultdict(list)
    themes, openings = collections.Counter(), collections.Counter()
    bands = collections.Counter()
    seen = set()
    positions = puzzles = 0
    rng = random.Random()  # No seed argument: actual IDs are the reproducibility record.
    for batch in pq.ParquetFile(Path(pool)/'index.parquet').iter_batches(
            batch_size=16384, columns=['puzzle_id', 'rating', 'themes', 'opening_tags']):
        for row in batch.to_pylist():
            if not _matches(row, filters):
                continue
            positions += 1
            pid = row['puzzle_id']
            if pid in seen:
                continue
            seen.add(pid)
            puzzles += 1
            themes.update(row['themes'])
            openings.update(row['opening_tags'])
            low = row['rating']//200*200
            bands[f'{low}-{low+199}'] += 1
            membership = tuple(i for i, g in enumerate(groups) if _matches(row, g))
            if not membership:
                continue
            counts[membership] += 1
            if collect:
                bucket = reservoirs[membership]
                if len(bucket) < total:
                    bucket.append(pid)
                else:
                    j = rng.randrange(counts[membership])
                    if j < total:
                        bucket[j] = pid
    catalog = {'puzzles': puzzles, 'positions': positions, 'themes': dict(themes),
               'opening_tags': dict(openings), 'rating_bands': dict(bands),
               'tag_count_unit': 'puzzles; multi-label counts can exceed puzzle count'}
    return counts, reservoirs, catalog


def _allocation(counts, targets):
    """Maximum flow over eligibility buckets -> quota groups (integer puzzles)."""
    residual = collections.defaultdict(dict)
    source, sink = ('source',), ('sink',)
    def edge(a, b, capacity):
        residual[a][b] = capacity
        residual[b][a] = 0
    for membership, count in counts.items():
        bucket = ('bucket', membership)
        edge(source, bucket, count)
        for i in membership:
            edge(bucket, ('group', i), count)
    for i, target in enumerate(targets):
        edge(('group', i), sink, target)
    while True:
        parent = {source: None}
        queue = collections.deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for other, capacity in residual[node].items():
                if capacity > 0 and other not in parent:
                    parent[other] = node
                    queue.append(other)
        if sink not in parent:
            break
        amount, node = sum(targets), sink
        while parent[node] is not None:
            previous = parent[node]
            amount = min(amount, residual[previous][node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= amount
            residual[node][previous] += amount
            node = previous
    allocation = {membership: {i: residual[('group', i)][('bucket', membership)]
                              for i in membership} for membership in counts}
    actual = [sum(a.get(i, 0) for a in allocation.values()) for i in range(len(targets))]
    return allocation, actual


def inspect_pool(pool, *, groups=(), num_puzzles=None, rating_min=None, rating_max=None,
                 themes=(), openings=(), collect=False):
    """Read metadata only; preview does not write data or launch training."""
    pool = Path(pool)
    proof = json.loads((pool/'verification.json').read_text())
    if proof.get('status') != 'passed':
        raise ValueError('Use an independently verified pool')
    if rating_min is not None and rating_max is not None and rating_min > rating_max:
        raise ValueError('rating_min exceeds rating_max')
    groups = normalize_groups(groups) if groups else []
    targets = quotas(groups, num_puzzles) if groups else []
    filters = {'rating_min': rating_min, 'rating_max': rating_max,
               'themes_all': list(themes), 'openings_any': list(openings)}
    counts, reservoirs, catalog = _scan(pool, groups, num_puzzles, filters, collect)
    allocation, actual = _allocation(counts, targets)
    report = {'status': 'feasible' if actual == targets else 'insufficient',
              'source_pool': str(pool.resolve()), 'catalog': catalog,
              'distinct_eligible_puzzles': sum(counts.values()) if groups else catalog['puzzles'],
              'query': {'unit': 'puzzles', 'num_puzzles': num_puzzles,
                        'filters': filters, 'groups': groups},
              'groups': {g['name']: {'target_puzzles': targets[i],
                         'eligible_puzzles': sum(c for m, c in counts.items() if i in m),
                         'assigned_puzzles': actual[i], 'shortfall': targets[i]-actual[i]}
                         for i, g in enumerate(groups)},
              'note': 'Preview reads index metadata; export additionally verifies full pool file hashes.'}
    chosen = {}
    if collect:
        rng = random.Random()
        for membership, allocations in allocation.items():
            candidates = reservoirs[membership]
            rng.shuffle(candidates)
            offset = 0
            for i, count in allocations.items():
                for pid in candidates[offset:offset+count]:
                    chosen[pid] = groups[i]['name']
                offset += count
    return report, chosen


def select_mix(pool, output, *, groups, num_puzzles, hypothesis, evidence=(), **filters):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not hypothesis.strip():
        raise ValueError('Describe the selection hypothesis')
    report, chosen = inspect_pool(pool, groups=groups, num_puzzles=num_puzzles,
                                  collect=True, **filters)
    if report['status'] == 'insufficient':
        output.mkdir(parents=True)
        report.update(hypothesis=hypothesis, evidence=list(evidence))
        write_json(output/'selection.json', report)
        write_json(output/'stats.json', {'groups': report['groups'], 'exported': False})
        return report
    # Reuse the existing verified exporter: exact training schema, all available
    # solver positions for each chosen puzzle, and unchanged prompt/reward data.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.lichess-mix-', dir=output.parent) as temporary:
        ids = Path(temporary)/'ids.txt'
        ids.write_text(''.join(pid+'\n' for pid in sorted(chosen)))
        result = select(pool, output, hypothesis=hypothesis, evidence=evidence, puzzle_ids=ids)
    stats = json.loads((output/'stats.json').read_text())
    groups_stats = report['groups']
    for g in groups_stats.values():
        g['positions'] = 0
    puzzle_themes, puzzle_openings = collections.Counter(), collections.Counter()
    seen = set()
    for batch in pq.ParquetFile(output/'index.parquet').iter_batches(
            columns=['puzzle_id', 'themes', 'opening_tags']):
        for row in batch.to_pylist():
            pid = row['puzzle_id']
            groups_stats[chosen[pid]]['positions'] += 1
            if pid not in seen:
                seen.add(pid)
                puzzle_themes.update(row['themes'])
                puzzle_openings.update(row['opening_tags'])
    stats.update(groups=groups_stats, puzzle_themes=dict(puzzle_themes),
                 puzzle_opening_tags=dict(puzzle_openings),
                 classification_note='Group quotas count puzzles; themes/opening_tags count positions. '
                 'puzzle_themes/puzzle_opening_tags count puzzles. Tags are multi-label.')
    write_json(output/'stats.json', stats)
    write_json(output/'selected_groups.json', chosen)
    result.update(query=report['query'], groups=groups_stats,
                  sampling='uniform without replacement within exact eligibility buckets; no seed option',
                  overlap_policy='Each puzzle assigned to one quota group; all original tags retained')
    result['files'].update({name: sha(output/name) for name in ('stats.json', 'selected_groups.json')})
    write_json(output/'selection.json', result)
    return result
