import json

import pytest
from fastapi.testclient import TestClient

from llmgo.rsi.codex_bridge import chat_base, create_app


def test_bridge_startup_error_reports_signal_and_redacts_log(tmp_path):
    from llmgo.rsi.codex_bridge import bridge_exit_error
    log=tmp_path/'bridge.log';log.write_text('import failed opaque-upstream-key')
    detail=bridge_exit_error(-11,log,('opaque-upstream-key',))
    assert 'SIGSEGV' in detail and 'opaque-upstream-key' not in detail
    log.write_text('')
    assert 'no service log output' in bridge_exit_error(1,log)


def test_runtime_overlay_is_used_only_for_matching_verified_host(tmp_path):
    from llmgo.rsi.codex_bridge import compatible_bridge_python,runtime_identity
    prefix=tmp_path/'venvs/llmgo-meta-bridge-devspace-20261023'
    (prefix/'bin').mkdir(parents=True);(prefix/'bin/python').touch()
    receipt=prefix/'installation.json'
    receipt.write_text(json.dumps({'status':'installed','runtime_identity':runtime_identity()}))
    assert compatible_bridge_python('original-python',root=tmp_path)==str(prefix/'bin/python')
    receipt.write_text(json.dumps({'status':'installed','runtime_identity':{'python':[3,12,99]}}))
    assert compatible_bridge_python('original-python',root=tmp_path)=='original-python'


@pytest.mark.parametrize('url',[
    'https://yeysai.com','https://yeysai.com/', 'https://yeysai.com/v1',
    'https://yeysai.com/v1/', 'https://yeysai.com/v1/chat/completions',
])
def test_base_url_never_duplicates_v1_or_chat_path(url):
    assert chat_base(url)=='https://yeysai.com/v1'


def test_bridge_authenticates_locally_and_forces_chat_upstream(tmp_path):
    seen=[]
    async def responder(**kwargs):
        seen.append(kwargs)
        return {'id':'response-one','object':'response','output':[],'status':'completed'}
    app=create_app('deepseek-v4.1-flash','https://yeysai.com/v1/chat/completions',
                   'opaque-upstream-key','local-test-token',tmp_path/'events.jsonl',responder=responder)
    with TestClient(app) as client:
        assert client.post('/v1/responses',json={'input':'hello'}).status_code==401
        response=client.post('/v1/responses',headers={'Authorization':'Bearer local-test-token'},json={'model':'deepseek-v4.1-flash','input':'hello'})
    assert response.status_code==200 and response.json()['status']=='completed'
    assert seen[0]['model']=='openai/deepseek-v4.1-flash'
    assert seen[0]['api_base']=='https://yeysai.com/v1'
    assert seen[0]['use_chat_completions_api'] is True and seen[0]['api_key']=='opaque-upstream-key'


def test_stream_events_keep_types_and_redact_saved_tokens(tmp_path):
    async def responder(**kwargs):
        async def events():
            yield {'type':'response.output_text.delta','delta':'opaque-upstream-key'}
            yield {'type':'response.completed','response':{'id':'one','status':'completed'}}
        return events()
    app=create_app('test','https://yeysai.com/v1','opaque-upstream-key','local-test-token',
                   tmp_path/'events.jsonl',responder=responder)
    with TestClient(app) as client:
        response=client.post('/v1/responses',headers={'Authorization':'Bearer local-test-token'},json={'stream':True,'input':'hello'})
    assert response.status_code==200 and 'event: response.completed' in response.text
    text=(tmp_path/'events.jsonl').read_text()
    assert 'opaque-upstream-key' not in text and 'local-test-token' not in text


def test_upstream_errors_are_redacted_and_keep_status(tmp_path):
    class Failure(Exception):
        status_code=401
    async def responder(**kwargs):
        raise Failure('Unauthorized opaque-upstream-key')
    app=create_app('test','https://yeysai.com/v1','opaque-upstream-key','local-test-token',
                   tmp_path/'events.jsonl',responder=responder)
    with TestClient(app) as client:
        response=client.post('/v1/responses',headers={'Authorization':'Bearer local-test-token'},json={'input':'hello'})
    assert response.status_code==401 and 'opaque-upstream-key' not in response.text
    assert 'opaque-upstream-key' not in (tmp_path/'events.jsonl').read_text()
