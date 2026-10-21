"""Token boundaries, portable replay, raw detail and incremental cursors."""
import asyncio
import hashlib
import json
from unittest.mock import patch

import httpx
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from llmgo.store.compact import TokenWindows, backfill
from llmgo.store.jsonl import TrajectoryStore


@pytest.fixture
def model(tmp_path):
    root = tmp_path / 'model'
    root.mkdir()
    vocab = {'<unk>': 0, **{f't{i}': i + 1 for i in range(6000)}, '思考': 6001}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(root / 'tokenizer.json'))
    return root


@pytest.mark.parametrize('count', [0, 999, 1000, 1001, 1999, 2000, 2001, 5000])
def test_exact_windows_have_no_overlap_and_preserve_boundaries(model, count):
    words = [f't{i}' for i in range(count)]
    excerpt = TokenWindows(model).clip(' '.join(words))
    assert excerpt['exact'] and excerpt['unit'] == 'tokens' and excerpt['total'] == count
    assert excerpt['head'].split() == words[:1000]
    assert excerpt['tail'].split() == words[max(1000, count - 1000):]
    assert excerpt['omitted'] == max(0, count - 2000)
    assert excerpt['head_count'] + excerpt['tail_count'] + excerpt['omitted'] == count


def test_missing_tokenizer_is_explicit_not_a_false_token_claim(tmp_path):
    excerpt = TokenWindows(tmp_path / 'absent').clip('思考' * 3000)
    assert not excerpt['exact'] and excerpt['unit'] == 'characters' and excerpt['warning']


def test_generated_token_windows_use_original_ids_rather_than_retokenizing(model, tmp_path):
    ids = list(range(5000, 0, -1))
    excerpt = TokenWindows(model).clip('visible text is intentionally different', ids)
    assert excerpt['scope'] == 'generated tokens' and excerpt['exact'] and excerpt['total'] == 5000
    assert excerpt['head'].split() == [f't{i}' for i in range(4999, 3999, -1)]
    assert excerpt['tail'].split() == [f't{i}' for i in range(999, -1, -1)]
    assert excerpt['generated_tokens']['head_ids'] == ids[:1000]
    assert excerpt['generated_tokens']['tail_ids'] == ids[-1000:]
    offline = TokenWindows(tmp_path / 'missing').clip('字元預覽', ids)
    assert not offline['exact'] and offline['unit'] == 'characters'
    assert offline['generated_tokens'] == excerpt['generated_tokens']


def make_store(tmp_path, model):
    store = TrajectoryStore(tmp_path / 'harness_runs' / 'portable')
    store.write_config({'game': 'chess', 'model_path': str(model), 'model_seat': 'white'})
    store.append({'type': 'position', 'game': 'chess', 'pieces': [], 'side': 'white'})
    store.append({'type': 'model_request', 'status': 'started', 'request_id': '0:white:1',
                  'seat': 'white', 'prompt': 'one prompt only', 'request_payload': {'messages': [{'content': 'one prompt only'}], 'max_tokens': 32768}})
    store.append({'type': 'token', 'request_id': '0:white:1', 'seat': 'white',
                  'channel': 'reasoning', 'text': 'stream fragments stay raw'})
    output = ' '.join(f't{i}' for i in range(5000))
    store.append({'type': 'model_request', 'status': 'finished', 'request_id': '0:white:1', 'seat': 'white',
                  'output': output, 'reasoning': output, 'text': 'e2e4', 'prompt': 'one prompt only',
                  'raw_response': 'r' * 400000, 'request_payload': {'messages': ['duplicated']},
                  'completion_tokens': 5100, 'usage': {'completion_tokens': 5100},
                  'latency_ms': 1200, 'finish_reason': 'stop', 'accepted': True, 'uci': 'e2e4'})
    store.append({'type': 'attempt', 'parsed_by': 'harness', 'source': 'model', 'seat': 'white', 'ok': True,
                  'uci': 'e2e4', 'reasoning': output, 'prompt': 'one prompt only', 'llm_input': 'one prompt only'})
    store.write_summary({'game_id': 'portable', 'game': 'chess', 'result': 'draw'})
    return store


def test_storage_is_bounded_and_backfill_reproduces_it_without_changing_raw(tmp_path, model):
    store = make_store(tmp_path, model)
    raw = store.events_path.read_bytes()
    short = store.short_events_path.read_bytes()
    assert len(short) < len(raw) / 10
    events = [json.loads(line) for line in short.splitlines()]
    assert not any(e['type'] == 'token' for e in events)
    finished = next(e for e in events if e.get('status') == 'finished')
    assert finished['output_excerpt']['total'] == 5000 and finished['output_excerpt']['omitted'] == 3000
    assert finished['usage']['completion_tokens'] == 5100 and finished['text'] == 'e2e4'
    assert 'prompt' not in finished and 'raw_response' not in finished and 'reasoning' not in events[-1]
    assert b'one prompt only' in short and short.count(b'one prompt only') == 1
    ref = finished['raw_ref']
    assert json.loads(raw[ref['offset']:ref['offset'] + ref['length']])['raw_response'] == 'r' * 400000
    result = backfill(store.root, TokenWindows(model))
    assert result['tokenizer_exact'] and store.short_events_path.read_bytes() == short
    assert store.events_path.read_bytes() == raw


def test_backfill_keeps_partial_unicode_thinking_and_ignores_incomplete_json(tmp_path, model):
    store = TrajectoryStore(tmp_path / 'unfinished')
    store.append({'type': 'model_request', 'status': 'started', 'seat': 'white', 'request_id': 'p'})
    store.append({'type': 'token', 'seat': 'white', 'request_id': 'p', 'channel': 'reasoning', 'text': '思考'})
    store.append({'type': 'token', 'seat': 'white', 'request_id': 'p', 'channel': 'reasoning', 'text': ' 思考'})
    with store.events_path.open('ab') as handle:
        handle.write(b'{"type":')
    digest = hashlib.sha256(store.events_path.read_bytes()).hexdigest()
    backfill(store.root, TokenWindows(model))
    last = json.loads(store.short_events_path.read_text().splitlines()[-1])
    assert last['status'] == 'streaming' and last['output_excerpt']['head'] == '思考 思考'
    assert hashlib.sha256(store.events_path.read_bytes()).hexdigest() == digest
