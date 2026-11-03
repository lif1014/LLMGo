"""Full Lichess puzzle pool with the existing non-thinking Chess-R1 schema."""
from __future__ import annotations

import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import time
import urllib.request

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from llmgo.training.chess_r1 import ANSWER_PROTOCOL, NONTHINKING_SYSTEM_PROMPT, messages

URL = 'https://database.lichess.org/lichess_db_puzzle.csv.zst'
COLUMNS = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
INDEX_SCHEMA = pa.schema([
    ('row_number', pa.int64()), ('sample_id', pa.string()), ('puzzle_id', pa.string()),
    ('solver_ply', pa.int64()), ('rating', pa.int64()), ('rating_deviation', pa.int64()),
    ('popularity', pa.int64()), ('nb_plays', pa.int64()), ('themes', pa.list_(pa.string())),
    ('opening_tags', pa.list_(pa.string())), ('game_url', pa.string()), ('daily_date', pa.string()),
    ('fen_key', pa.string()), ('side', pa.string()), ('legal_count', pa.int64()),
    ('solution_plies', pa.int64()),
])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def position_key(board):
    # Ignore move clocks; retain side, castling and only legally relevant en passant.
    return ' '.join(board.fen(en_passant='legal').split()[:4])


def fingerprint(key):
    return hashlib.sha256(key.encode()).digest()


def download(destination, url=URL):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_suffix(destination.suffix+'.json')
    if destination.exists():
        if not receipt.is_file() or json.loads(receipt.read_text())['sha256'] != sha(destination):
            raise ValueError('Existing raw data lacks a matching download receipt')
        return json.loads(receipt.read_text())
    temporary = destination.with_suffix(destination.suffix+'.partial')
    # A complete retry replaces only this tool's incomplete download.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=120) as response, temporary.open('wb') as stream:
        headers = dict(response.headers)
        total = 0
        for block in iter(lambda: response.read(4*1024*1024), b''):
            stream.write(block)
            total += len(block)
        advertised = response.headers.get('Content-Length')
        if advertised is not None and total != int(advertised):
            raise RuntimeError('Incomplete Lichess download')
    result = {'url':url, 'bytes':total, 'sha256':sha(temporary), 'headers':headers,
              'downloaded_at':time.time(), 'license':'CC0'}
    temporary.replace(destination)
    write_json(receipt, result)
    return result


def holdout_identity(paths):
    ids, positions, files = set(), set(), []
    for path in map(Path, paths):
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=['reward_model', 'extra_info']):
            for row in batch.to_pylist():
                gt = row['reward_model']['ground_truth']
                gt = json.loads(gt) if isinstance(gt, str) else gt
                ids.add(str(gt.get('id') or row['extra_info']['puzzle_id']))
                positions.add(fingerprint(position_key(chess.Board(gt['board_fen']))))
                count += 1
        files.append({'path':str(path.resolve()), 'sha256':sha(path), 'rows':count})
    if not ids or not positions:
        raise ValueError('Nonempty held-out puzzle IDs and positions are required')
    return ids, positions, files


def _puzzle(record):
    """Validate the whole trajectory before returning any solver examples."""
    try:
        for field in ('Rating', 'RatingDeviation', 'Popularity', 'NbPlays'):
            int(record.get(field) or 0)
        board = chess.Board(record['FEN'])
        moves = record['Moves'].split()
        if not board.is_valid() or len(moves) < 2:
            raise ValueError('invalid position or missing solution')
        points = []
        for ply, uci in enumerate(moves):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f'illegal trajectory move at ply {ply}')
            if ply % 2 == 1:
                prompt = messages(board.fen(), enable_thinking=False)
                legal = prompt[1]['content'].split('\nLegal moves: ', 1)[1]
                points.append({'fen':board.fen(), 'san':board.san(move), 'legal':legal,
                               'prompt':prompt, 'ply':ply, 'key':position_key(board)})
            board.push(move)
        return record, points, None
    except (ValueError, KeyError, IndexError) as error:
        return record, [], str(error)


def _chunk(records):
    return [_puzzle(record) for record in records]


def _converted(records, workers):
    if workers == 1:
        yield from map(_puzzle, records)
        return
    def chunks():
        buffer = []
        for record in records:
            buffer.append(record)
            if len(buffer) == 64:
                yield buffer
                buffer = []
        if buffer:
            yield buffer
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        pending, source = collections.deque(), iter(chunks())
        for _ in range(workers*2):
            chunk = next(source, None)
            if chunk is not None:
                pending.append(pool.submit(_chunk, chunk))
        while pending:
            yield from pending.popleft().result()
            chunk = next(source, None)
            if chunk is not None:
                pending.append(pool.submit(_chunk, chunk))


def _statistics():
    return {'positions':0, 'puzzles':0, 'themes':collections.Counter(),
            'opening_tags':collections.Counter(), 'rating_bands':collections.Counter(),
            'side':collections.Counter()}


def _count(stats, meta):
    stats['positions'] += 1
    stats['themes'].update(meta['themes'])
    stats['opening_tags'].update(meta['opening_tags'])
    band = meta['rating']//200*200
    stats['rating_bands'][f'{band}-{band+199}'] += 1
    stats['side'].update([meta['side']])


def prepare(raw, output, reference, heldouts, *, workers=1, progress=None):
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(f'Preserve the existing dataset: {output}')
    schema = pq.ParquetFile(reference).schema_arrow
    if schema.names != COLUMNS:
        raise ValueError('Reference is not the current Chess-R1 five-column schema')
    held_ids, held_fens, held_files = holdout_identity(heldouts)
    building = output.with_name(output.name+'.building')
    building.mkdir(parents=True)  # A failed build is preserved for diagnosis.
    counts, stats, seen, puzzle_ids = collections.Counter(), _statistics(), set(), set()
    train_buffer, index_buffer = [], []
    started, last_progress = time.time(), 0.
    def publish():
        nonlocal last_progress
        last_progress = time.time()
        value = {'status':'building', 'counts':dict(counts), 'seconds':time.time()-started,
                 'output':str(output), 'updated':time.time()}
        write_json(progress or building/'progress.json', value)
        print(json.dumps(value), flush=True)
    def records(reader):
        for record in reader:
            counts['raw_puzzles'] += 1
            if record['PuzzleId'] in held_ids:
                counts['excluded_holdout_id_puzzles'] += 1
                continue
            yield record
    with raw.open('rb') as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as decoded, \
            io.TextIOWrapper(decoded, encoding='utf-8', newline='') as text, \
            pq.ParquetWriter(building/'train.parquet', schema, compression='zstd') as train_writer, \
            pq.ParquetWriter(building/'index.parquet', INDEX_SCHEMA, compression='zstd') as index_writer, \
            (building/'invalid.jsonl').open('w') as invalid:
        reader = csv.DictReader(text)
        if not {'PuzzleId','FEN','Moves','Rating','Themes'}.issubset(reader.fieldnames or []):
            raise ValueError('Unexpected raw Lichess CSV header')
        raw_columns = reader.fieldnames
        def flush():
            if train_buffer:
                train_writer.write_table(pa.Table.from_pylist(train_buffer, schema=schema))
                index_writer.write_table(pa.Table.from_pylist(index_buffer, schema=INDEX_SCHEMA))
                train_buffer.clear()
                index_buffer.clear()
        for record, points, error in _converted(records(reader), workers):
            pid = record['PuzzleId']
            if error:
                counts['invalid_puzzles'] += 1
                invalid.write(json.dumps({'puzzle_id':pid,'error':error})+'\n')
                continue
            if any(fingerprint(point['key']) in held_fens for point in points):
                counts['excluded_holdout_position_puzzles'] += 1
                continue
            if pid in puzzle_ids:
                counts['duplicate_puzzle_ids'] += 1
                continue
            puzzle_ids.add(pid)
            emitted = False
            for point in points:
                key = fingerprint(point['key'])
                if key in seen:
                    counts['duplicate_positions'] += 1
                    continue
                seen.add(key)
                number = counts['positions']
                rating = int(record['Rating'])
                gt = {'board_fen':point['fen'], 'next_move_san':point['san'],
                      'legal_moves_san':point['legal'], 'id':pid, 'rating':rating}
                train_buffer.append({'data_source':'chess_r1_train', 'prompt':point['prompt'],
                    'ability':'chess', 'reward_model':{'style':'rule', 'ground_truth':json.dumps(gt,separators=(',',':'))},
                    'extra_info':{'split':'train', 'puzzle_id':pid, 'rating':rating,
                                  'solver_index':number, 'answer_protocol':ANSWER_PROTOCOL}})
                meta = {'row_number':number, 'sample_id':f'{pid}:{point["ply"]}', 'puzzle_id':pid,
                    'solver_ply':point['ply'], 'rating':rating,
                    'rating_deviation':int(record.get('RatingDeviation') or 0),
                    'popularity':int(record.get('Popularity') or 0), 'nb_plays':int(record.get('NbPlays') or 0),
                    'themes':record.get('Themes','').split(), 'opening_tags':record.get('OpeningTags','').split(),
                    'game_url':record.get('GameUrl',''), 'daily_date':record.get('DailyDate',''),
                    'fen_key':point['key'], 'side':'white' if point['fen'].split()[1]=='w' else 'black',
                    'legal_count':len(point['legal'].split()), 'solution_plies':len(record['Moves'].split())-1}
                index_buffer.append(meta)
                _count(stats, meta)
                counts['positions'] += 1
                emitted = True
                if len(train_buffer) >= 4096:
                    flush()
            counts['training_puzzles' if emitted else 'all_positions_duplicate_puzzles'] += 1
            if time.time()-last_progress >= 30:
                publish()
        flush()
    if not counts['positions']:
        raise ValueError('Preparation produced no training data')
    # Hold-out files must be the same evidence throughout the build.
    if any(sha(row['path']) != row['sha256'] for row in held_files):
        raise RuntimeError('Held-out data changed during preparation')
    stats['puzzles'] = counts['training_puzzles']
    write_json(building/'stats.json', stats)
    manifest = {'status':'prepared', 'source':{'path':str(raw.resolve()),'sha256':sha(raw),'columns':raw_columns},
        'reference':{'path':str(Path(reference).resolve()),'sha256':sha(reference)},
        'heldouts':held_files, 'counts':dict(counts), 'schema':str(schema),
        'answer_protocol':ANSWER_PROTOCOL, 'enable_thinking':False,
        'system_prompt_sha256':hashlib.sha256(NONTHINKING_SYSTEM_PROMPT.encode()).hexdigest(),
        'deduplication':'legal-en-passant FEN first four fields; move clocks ignored; SHA-256 identity',
        'selection':'all valid solver turns excluding held-out puzzle IDs/positions; first occurrence per position',
        'files':{name:{'sha256':sha(building/name),'bytes':(building/name).stat().st_size}
                 for name in ('train.parquet','index.parquet','stats.json','invalid.jsonl')},
        'seconds':time.time()-started}
    write_json(building/'manifest.json', manifest)
    building.rename(output)
    return manifest


def verify(pool, reference, heldouts):
    """Independently scan every emitted row; regenerate legal SAN on a sample."""
    pool = Path(pool)
    manifest = json.loads((pool/'manifest.json').read_text())
    held_ids, held_fens, held_files = holdout_identity(heldouts)
    if held_files != manifest['heldouts']:
        raise ValueError('Held-out identity differs from pool preparation')
    schema = pq.ParquetFile(reference).schema_arrow
    train, index = pq.ParquetFile(pool/'train.parquet'), pq.ParquetFile(pool/'index.parquet')
    if not train.schema_arrow.equals(schema) or train.metadata.num_rows != index.metadata.num_rows:
        raise ValueError('Pool schema or index row count differs')
    for name, identity in manifest['files'].items():
        if sha(pool/name) != identity['sha256']:
            raise ValueError(f'Pool file changed: {name}')
    seen, rows, canonical_checks = set(), 0, 0
    batches = zip(train.iter_batches(batch_size=4096), index.iter_batches(batch_size=4096), strict=True)
    for batch, index_batch in batches:
        if len(batch) != len(index_batch):
            raise ValueError('Index batch alignment differs')
        for row, meta in zip(batch.to_pylist(), index_batch.to_pylist(), strict=True):
            gt = json.loads(row['reward_model']['ground_truth'])
            key = fingerprint(' '.join(gt['board_fen'].split()[:4]))
            if gt['id'] in held_ids or key in held_fens or key in seen:
                raise ValueError('Held-out contamination or duplicate position')
            seen.add(key)
            expected_prompt = [{'role':'system','content':NONTHINKING_SYSTEM_PROMPT},
                {'role':'user','content':f'Current FEN string: {gt["board_fen"]}\nLegal moves: {gt["legal_moves_san"]}'}]
            if (row['data_source'] != 'chess_r1_train' or row['ability'] != 'chess'
                    or row['reward_model']['style'] != 'rule'
                    or row['prompt'] != expected_prompt or row['extra_info']['split'] != 'train'
                    or row['extra_info']['puzzle_id'] != gt['id']
                    or row['extra_info']['solver_index'] != rows
                    or row['extra_info']['answer_protocol'] != ANSWER_PROTOCOL):
                raise ValueError('Prompt/protocol/index mismatch')
            if (meta['row_number'] != rows or meta['puzzle_id'] != gt['id']
                    or meta['rating'] != gt['rating']
                    or meta['fen_key'] != ' '.join(gt['board_fen'].split()[:4])
                    or meta['sample_id'] != f'{gt["id"]}:{meta["solver_ply"]}'):
                raise ValueError('Selection index does not match the training row')
            if rows < 128 or rows % 10000 == 0:
                board = chess.Board(gt['board_fen'])
                if row['prompt'] != messages(gt['board_fen'], enable_thinking=False):
                    raise ValueError('Canonical legal SAN prompt differs')
                if gt['next_move_san'] not in [board.san(move) for move in board.legal_moves]:
                    raise ValueError('Illegal solution label')
                canonical_checks += 1
            rows += 1
    result = {'status':'passed','rows':rows,'schema_matches_reference':True,
              'holdout_id_overlap':0,'holdout_position_overlap':0,'duplicate_positions':0,
              'canonical_legal_checks':canonical_checks,'heldouts':held_files,'time':time.time()}
    write_json(pool/'verification.json', result)
    return result


def select(pool, output, *, hypothesis, rating_min=None, rating_max=None, themes=(),
           openings=(), puzzle_ids=None, sample_ids=None, max_positions=None, evidence=()):
    pool, output = Path(pool), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not hypothesis.strip():
        raise ValueError('Describe the selection hypothesis')
    if max_positions is not None and max_positions < 1:
        raise ValueError('max_positions must be positive')
    source = json.loads((pool/'manifest.json').read_text())
    proof = json.loads((pool/'verification.json').read_text())
    if proof.get('status') != 'passed':
        raise ValueError('Use an independently verified pool')
    for name in ('train.parquet','index.parquet'):
        if sha(pool/name) != source['files'][name]['sha256']:
            raise ValueError('Selection pool identity changed')
    output.mkdir(parents=True)
    allowed_puzzles = set(Path(puzzle_ids).read_text().split()) if puzzle_ids else None
    allowed_samples = set(Path(sample_ids).read_text().split()) if sample_ids else None
    selected, ids, stats = set(), set(), _statistics()
    index = pq.ParquetFile(pool/'index.parquet')
    with pq.ParquetWriter(output/'index.parquet', index.schema_arrow, compression='zstd') as writer, \
            (output/'selected_positions.txt').open('w') as sample_list:
        for batch in index.iter_batches(batch_size=4096):
            matched = []
            for row in batch.to_pylist():
                if rating_min is not None and row['rating'] < rating_min or rating_max is not None and row['rating'] > rating_max:
                    continue
                if themes and not set(themes).issubset(row['themes']):
                    continue
                if openings and not set(openings).intersection(row['opening_tags']):
                    continue
                if allowed_puzzles is not None and row['puzzle_id'] not in allowed_puzzles:
                    continue
                if allowed_samples is not None and row['sample_id'] not in allowed_samples:
                    continue
                selected.add(row['row_number'])
                ids.add(row['puzzle_id'])
                matched.append(row)
                sample_list.write(row['sample_id']+'\n')
                _count(stats, row)
                if max_positions is not None and len(selected) >= max_positions:
                    break
            if matched:
                writer.write_table(pa.Table.from_pylist(matched, schema=index.schema_arrow))
            if max_positions is not None and len(selected) >= max_positions:
                break
    if not selected:
        write_json(output/'selection.json', {'status':'empty','hypothesis':hypothesis})
        raise ValueError('Selection matched no positions; inspect the preserved selection directory')
    train = pq.ParquetFile(pool/'train.parquet')
    offset = 0
    last_selected = max(selected)
    with pq.ParquetWriter(output/'train.parquet', train.schema_arrow, compression='zstd') as writer:
        for batch in train.iter_batches(batch_size=4096):
            indices = [i for i in range(len(batch)) if offset+i in selected]
            if indices:
                writer.write_table(pa.Table.from_batches([batch]).take(pa.array(indices, type=pa.int64())))
            offset += len(batch)
            if offset > last_selected:
                break
    (output/'selected_ids.txt').write_text(''.join(pid+'\n' for pid in sorted(ids)))
    stats['puzzles'] = len(ids)
    write_json(output/'stats.json', stats)
    result = {'status':'selected','hypothesis':hypothesis,'evidence':list(evidence),
        'query':{'rating_min':rating_min,'rating_max':rating_max,'themes_all':list(themes),
                 'openings_any':list(openings),'max_positions':max_positions,
                 'puzzle_ids_sha256':sha(puzzle_ids) if puzzle_ids else None,
                 'sample_ids_sha256':sha(sample_ids) if sample_ids else None},
        'source_manifest_sha256':sha(pool/'manifest.json'),'source_pool':str(pool.resolve()),
        'positions':len(selected),'puzzles':len(ids),
        'files':{name:sha(output/name) for name in ('train.parquet','index.parquet',
                 'selected_ids.txt','selected_positions.txt','stats.json')}}
    write_json(output/'selection.json', result)
    return result


def report_training_data(paths, output, *, hypothesis, evidence=(), query='use existing filtered training data'):
    """Record the same delivery artifacts when retaining the default dataset."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not hypothesis.strip():
        raise ValueError('Describe the data-choice hypothesis')
    output.mkdir(parents=True)
    ids, stats, files = set(), _statistics(), []
    for path in map(Path, paths):
        before = sha(path)
        for batch in pq.ParquetFile(path).iter_batches(columns=['reward_model','extra_info']):
            for row in batch.to_pylist():
                gt = row['reward_model']['ground_truth']
                gt = json.loads(gt) if isinstance(gt,str) else gt
                ids.add(gt['id'])
                _count(stats, {'rating':gt['rating'],'themes':[],'opening_tags':[],
                              'side':'white' if gt['board_fen'].split()[1]=='w' else 'black'})
        if sha(path) != before:
            raise ValueError('Training file changed while generating the data report')
        files.append({'path':str(path.resolve()),'sha256':before})
    stats['puzzles'] = len(ids)
    stats['classification_note'] = 'Rating/side counted from training records; themes/opening tags require the separate full-pool index.'
    (output/'selected_ids.txt').write_text(''.join(pid+'\n' for pid in sorted(ids)))
    write_json(output/'stats.json',stats)
    result = {'status':'reported','query':query,'hypothesis':hypothesis,'evidence':list(evidence),
              'training_files':files,'positions':stats['positions'],'puzzles':len(ids)}
    write_json(output/'selection.json',result)
    return result
