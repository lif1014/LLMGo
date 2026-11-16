"""DeepSeek Flash non-thinking chess client shared by reference and final tests."""
from __future__ import annotations

import json
import logging
import os
import http.client
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from llmgo.training.chess_r1 import extract_answer

DEFAULT_KEY = Path('/user/lyh4170/.config/llmgo/deepseek-api-key')
DEFAULT_PROXY = 'http://whitelist-proxy.cybertron.svc.cluster.local:7891'
REQUEST_ATTEMPTS = 8
log = logging.getLogger(__name__)


class PostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """307/308 retain the completion POST, without forwarding keys to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if req.get_method() == 'POST' and code in (307, 308):
            def origin(url):
                p = urlsplit(url)
                return p.scheme, p.hostname, p.port or (443 if p.scheme == 'https' else 80)
            if origin(req.full_url) != origin(newurl):
                raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)
            return urllib.request.Request(newurl, data=req.data,
                headers=dict(req.header_items()), method='POST',
                origin_req_host=req.origin_req_host, unverifiable=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def opponent_identity():
    return {'model':os.environ.get('DEEPSEEK_MODEL','deepseek-flash'),
            'base_url':os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/'),
            'thinking':'disabled'}


def request(messages, request_id, timeout=None, max_tokens=16384):
    key = Path(os.environ.get('DEEPSEEK_KEY_FILE', str(DEFAULT_KEY))).read_text().strip()
    identity = opponent_identity()
    body = {'model': identity['model'], 'thinking': {'type': 'disabled'},
            'messages': messages, 'temperature': 1, 'top_p': .95,
            'max_tokens': max_tokens, 'stream': False}
    req = urllib.request.Request(identity['base_url']+'/chat/completions',
        data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json', 'X-Client-Request-Id': request_id})
    proxy = os.environ.get('DEEPSEEK_PROXY_URL', DEFAULT_PROXY)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({'http': proxy, 'https': proxy} if proxy else {}),
        PostRedirectHandler())
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            with opener.open(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(400).decode(errors='replace')
            if error.code not in (307, 308, 429, 500, 502, 503, 504) or attempt == REQUEST_ATTEMPTS - 1:
                raise RuntimeError(f'DeepSeek HTTP {error.code}: {detail}') from error
            log.warning('DeepSeek HTTP %s; retry %s/%s for %s',
                        error.code, attempt + 1, REQUEST_ATTEMPTS - 1, request_id)
        except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError) as error:
            if attempt == REQUEST_ATTEMPTS - 1:
                raise RuntimeError('DeepSeek network ' + type(error).__name__) from error
            log.warning('DeepSeek %s; retry %s/%s for %s',
                        type(error).__name__, attempt + 1, REQUEST_ATTEMPTS - 1, request_id)
        time.sleep(min(30, 2 ** (attempt + 1)))


def extract_loose_answer(response, fen, *, finish_reason=None):
    """Compatibility name; DeepSeek and Qwen share the same last-box parser."""
    return extract_answer(response, fen, finish_reason=finish_reason)
