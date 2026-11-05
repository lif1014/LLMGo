"""DeepSeek Flash non-thinking chess client shared by reference and final tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from llmgo.training.chess_r1 import extract_answer

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
    """Compatibility name; DeepSeek and Qwen share the same last-box parser."""
    return extract_answer(response, fen, finish_reason=finish_reason)
