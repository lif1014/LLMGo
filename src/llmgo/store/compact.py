"""Small, portable traces. Raw events remain the authoritative local record."""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

WINDOW = 1000


@lru_cache(maxsize=4)
def _tokenizer(path: str):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TokenWindows:
    def __init__(self, model_path=None):
        self.tokenizer = None
        self.source = None
        self.error = 'No readable tokenizer.json; character windows are not token windows'
        candidate = os.environ.get('LLMGO_TOKENIZER') or model_path
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_dir():
                path = path / 'tokenizer.json'
            try:
                self.tokenizer, digest = _tokenizer(str(path.resolve()))
                self.source = {'file': path.name, 'sha256': digest}
                self.error = None
            except Exception as exc:
                # A missing tokenizer must never abort an otherwise valid game.
                self.tokenizer = None
                self.error = f'{type(exc).__name__}: {exc}'

    def clip(self, text: str) -> dict:
        text = str(text or '')
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids if self.tokenizer else None
        total = len(ids) if ids is not None else len(text)
        head_count = min(total, WINDOW)
        tail_count = min(max(total - head_count, 0), WINDOW)
        omitted = max(0, total - head_count - tail_count)
        if ids is not None:
            head = self.tokenizer.decode(ids[:head_count], skip_special_tokens=False)
            tail = self.tokenizer.decode(ids[-tail_count:], skip_special_tokens=False) if tail_count else ''
        else:
            head, tail = text[:head_count], text[-tail_count:] if tail_count else ''
        return {'head': head, 'tail': tail, 'head_count': head_count, 'tail_count': tail_count,
                'omitted': omitted, 'total': total, 'window': WINDOW,
                'unit': 'tokens' if ids is not None else 'characters', 'exact': ids is not None,
                'scope': 'retokenized visible model output; generation usage is separate',
                'tokenizer': self.source, 'warning': self.error}


def excerpt_text(excerpt: dict) -> str:
    middle = f"\n\n[… {excerpt['omitted']} {excerpt['unit']} omitted …]\n\n" if excerpt['omitted'] else ''
    return excerpt['head'] + middle + excerpt['tail']


def from_config(config: dict) -> TokenWindows:
    return TokenWindows(config.get('model_path') or (config.get('server') or {}).get('model_path')
                        or (config.get('protocol') or {}).get('model_path'))


def compact_event(event: dict, windows: TokenWindows, raw_ref=None) -> dict | None:
    kind = event.get('type')
    if kind == 'token':
        return None  # One bounded excerpt per completed request, never duplicate stream chunks.
    shown = {key: value for key, value in event.items() if key not in {
        'prompt', 'llm_input', 'reasoning', 'output', 'raw_response', 'request_payload',
        'data', 'tool_library', 'allowed_choices', 'blackboard', 'skills', 'options'}}
    shown['trace_format'] = 'short-v1'
    for key in ('error', 'error_detail'):
        if isinstance(shown.get(key), str) and len(shown[key]) > 2000:
            shown[key] = shown[key][:1000] + '\n[… error truncated; see raw event …]\n' + shown[key][-1000:]
    if raw_ref is not None:
        shown['raw_ref'] = raw_ref
    if kind == 'model_request':
        if event.get('status') == 'started':
            shown['prompt'] = event.get('prompt', '')
            shown['request_payload'] = {k: v for k, v in event.get('request_payload', {}).items() if k != 'messages'}
        else:
            output = event.get('output') or '\n\n'.join(filter(None, (event.get('reasoning'), event.get('text'))))
            shown['output_excerpt'] = windows.clip(output)
            # Final answers are normally a few characters. Bound malformed prose too.
            final = event.get('text', '')
            shown['text'] = final if len(final) <= 2000 else final[:1000] + '\n[… final text truncated …]\n' + final[-1000:]
            shown['final_text_truncated'] = len(final) > 2000
    elif kind == 'tool_call':
        shown['data'] = {k: v for k, v in event.get('data', {}).items() if k != 'candidates'}
        if event.get('tool') == 'candidates' and event.get('status') == 'finished':
            shown['blackboard'] = event.get('result', [])
    elif kind == 'choice':
        shown['skills'] = [{'id': card['id'], 'title': card.get('title')} for card in event.get('skills', [])]
        if event.get('status') == 'asking':
            shown['options'] = event.get('options', [])
            if event.get('prompt') and not event.get('turn_id'):
                shown['prompt'] = event['prompt']  # Older traces have no model_request records.
        if event.get('output'):
            shown['output_excerpt'] = windows.clip(event['output'])
        if not event.get('turn_id') and event.get('blackboard'):
            shown['blackboard'] = event['blackboard']
    elif kind == 'attempt' and event.get('parsed_by') != 'harness':
        # Other games retain their existing replay contract, with bounded output.
        shown['prompt'] = event.get('prompt', '')
        shown['reasoning'] = excerpt_text(windows.clip(event.get('reasoning', '')))
    return shown


def backfill(folder: Path, windows: TokenWindows | None = None) -> dict:
    """Stream old logs into an atomic compact file; preserve unfinished responses."""
    if windows is None:
        path = folder / 'config.json'
        windows = from_config(json.loads(path.read_text()) if path.exists() else {})
    target = folder / 'short_events.jsonl'
    temporary = folder / f'.short_events.{os.getpid()}.tmp'
    pending = {}
    count = 0
    with (folder / 'events.jsonl').open('rb') as source, temporary.open('w', encoding='utf-8') as destination:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line.endswith(b'\n'):
                break
            if not line.strip():
                continue
            event = json.loads(line)
            key = event.get('request_id') or (event.get('seat'), event.get('stage'))
            if event.get('type') == 'token':
                record = pending.setdefault(key, {'type': 'model_request', 'status': 'streaming',
                    'request_id': event.get('request_id'), 'seat': event.get('seat'), 'stage': event.get('stage'),
                    'ply': event.get('ply'), 'reasoning': '', 'text': ''})
                channel = 'reasoning' if event.get('channel') == 'reasoning' else 'text'
                record[channel] = event.get('text', '') if event.get('replace') else record[channel] + event.get('text', '')
            elif event.get('type') == 'model_request' and event.get('status') == 'finished':
                pending.pop(key, None)
            shown = compact_event(event, windows, {'offset': offset, 'length': len(line)})
            if shown is not None:
                destination.write(json.dumps(shown, ensure_ascii=False) + '\n')
                count += 1
        for event in pending.values():
            destination.write(json.dumps(compact_event(event, windows), ensure_ascii=False) + '\n')
            count += 1
    temporary.replace(target)
    return {'events': count, 'bytes': target.stat().st_size, 'tokenizer_exact': windows.tokenizer is not None}
