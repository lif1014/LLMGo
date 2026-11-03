"""DeepSeek Flash non-thinking chess client shared by reference and final tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from llmgo.training.chess_r1 import Answer, extract_answer

DEFAULT_KEY = Path('/user/lyh4170/.config/llmgo/deepseek-api-key')
DEFAULT_PROXY = 'http://whitelist-proxy.cybertron.svc.cluster.local:7891'


def opponent_identity():
    return {'model':os.environ.get('DEEPSEEK_MODEL','deepseek-flash'),
            'base_url':os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/'),
            'thinking':'disabled'}


def request(messages, request_id, timeout=90, max_tokens=16384):
    key = Path(os.environ.get('DEEPSEEK_KEY_FILE', str(DEFAULT_KEY))).read_text().strip()
    identity = opponent_identity()
    body = {'model': identity['model'], 'thinking': {'type': 'disabled'},
            'messages': messages, 'temperature': 1, 'top_p': .95,
            'max_tokens': max_tokens, 'stream': False}
    req = urllib.request.Request(identity['base_url']+'/chat/completions',
        data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json', 'X-Client-Request-Id': request_id})
    proxy = os.environ.get('DEEPSEEK_PROXY_URL', DEFAULT_PROXY)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))
    for attempt in range(3):
        try:
            with opener.open(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(400).decode(errors='replace')
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f'DeepSeek HTTP {error.code}: {detail}') from error
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt == 2:
                raise RuntimeError('DeepSeek network ' + type(error).__name__) from error
        time.sleep(2 ** attempt)


def extract_loose_answer(response, fen, *, finish_reason=None):
    """Accept only unambiguous legal boxed SAN, allowing harmless surrounding prose."""
    if finish_reason in ('length', 'max_tokens'):
        return Answer(error='truncated')
    text = response.strip()
    for suffix in ('<|im_end|>', '<|endoftext|>'):
        text = text.removesuffix(suffix).strip()
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1].strip()
    if '<think>' in text or '</think>' in text:
        return Answer(error='incomplete_thinking')
    markers = list(re.finditer(r'\\boxed', text))
    starts = list(re.finditer(r'\\boxed\s*\{', text))
    if not starts:
        return Answer(error='missing_box')
    if len(markers) != len(starts):
        return Answer(error='malformed_box')
    answers = []
    for start in starts:
        depth, end = 1, start.end()
        while end < len(text) and depth:
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        if depth:
            return Answer(error='unclosed_box')
        candidate = extract_answer(text[start.start():end], fen, finish_reason='stop')
        if not candidate.legal:
            return Answer(san=candidate.san, error=candidate.error or 'invalid_boxed_san')
        answers.append(candidate)
    if len({item.uci for item in answers}) != 1:
        return Answer(error='conflicting_boxed_moves')
    return answers[0]
