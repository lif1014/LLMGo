import http.client
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llmgo.rsi import deepseek_client as client


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    key = tmp_path/'key'
    key.write_text('fixture-key')
    monkeypatch.setenv('DEEPSEEK_KEY_FILE', str(key))
    monkeypatch.setenv('DEEPSEEK_PROXY_URL', '')
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'http://127.0.0.1:12345/v1')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'fixture')


@pytest.mark.parametrize('status', [307, 308])
def test_real_post_redirect_preserves_body_and_request_identity(credentials, monkeypatch, status):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            received.append((self.path, body, self.headers['Authorization'],
                             self.headers['X-Client-Request-Id'], self.headers['Content-Type']))
            if self.path == '/v1/chat/completions':
                self.send_response(status)
                self.send_header('Location', '/v1/completion-backend')
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"choices":[]}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('DEEPSEEK_BASE_URL', f'http://127.0.0.1:{server.server_port}/v1')
    try:
        assert client.request([{'role':'user', 'content':'move'}], 'same-request') == {'choices':[]}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(received) == 2
    assert received[0][1:] == received[1][1:]
    body = json.loads(received[1][1])
    assert body['thinking'] == {'type':'disabled'} and body['max_tokens'] == 16384


def test_308_without_location_retries_same_post_without_a_deadline(credentials, monkeypatch):
    calls, sleeps = [], []
    class Opener:
        def open(self, request, *, timeout):
            calls.append((request, timeout))
            if len(calls) < 3:
                raise urllib.error.HTTPError(request.full_url, 308, 'redirect', {}, io.BytesIO())
            return io.BytesIO(b'{"choices":[]}')
    monkeypatch.setattr(client.urllib.request, 'build_opener', lambda *args:Opener())
    monkeypatch.setattr(client.time, 'sleep', sleeps.append)
    assert client.request([], 'request-308') == {'choices':[]}
    assert len({id(req) for req, _ in calls}) == 1
    assert all(timeout is None and req.get_method() == 'POST' for req, timeout in calls)
    assert sleeps == [2, 4]


@pytest.mark.parametrize('status,attempts', [(308, 8), (429, 8), (401, 1)])
def test_http_retry_exhaustion_is_explicit(credentials, monkeypatch, status, attempts):
    calls = []
    class Opener:
        def open(self, request, *, timeout):
            calls.append(timeout)
            raise urllib.error.HTTPError(request.full_url, status, 'fixture', {}, io.BytesIO())
    monkeypatch.setattr(client.urllib.request, 'build_opener', lambda *args:Opener())
    monkeypatch.setattr(client.time, 'sleep', lambda _:None)
    with pytest.raises(RuntimeError, match=f'DeepSeek HTTP {status}'):
        client.request([], 'fixture-request')
    assert len(calls) == attempts and all(timeout is None for timeout in calls)


def test_redirect_does_not_forward_authorization_to_another_origin():
    request = urllib.request.Request('https://provider.test/v1/chat/completions',
        data=b'{}', headers={'Authorization':'Bearer fixture'}, method='POST')
    with pytest.raises(urllib.error.HTTPError):
        client.PostRedirectHandler().redirect_request(request, None, 308, 'redirect', {},
                                                      'https://another.test/v1/chat/completions')


def test_disconnected_response_is_retried(credentials, monkeypatch):
    calls = []
    class Opener:
        def open(self, request, *, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise http.client.RemoteDisconnected()
            return io.BytesIO(b'{"choices":[]}')
    monkeypatch.setattr(client.urllib.request, 'build_opener', lambda *args:Opener())
    monkeypatch.setattr(client.time, 'sleep', lambda _:None)
    assert client.request([], 'fixture-request') == {'choices':[]}
    assert calls == [None, None]
