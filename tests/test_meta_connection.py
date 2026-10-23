"""Exercise connection checks with a real subprocess and no external API calls."""
import json
import os
import sys

import pytest

from llmgo.rsi.connection_check import check_connection


SERVER = r'''
import json, os, sys, time
if '--version' in sys.argv:
    print('codex-cli connection-test')
    raise SystemExit(0)
def emit(obj):
    print(json.dumps(obj), flush=True)
print('diagnostic ' + os.environ.get('OPENAI_API_KEY',''), file=sys.stderr, flush=True)
print(os.environ.get('FAKE_OLD_KEY',''), file=sys.stderr, flush=True)
for line in sys.stdin:
    request=json.loads(line)
    method=request.get('method')
    if 'id' not in request:continue
    rid=request['id']
    if method=='initialize':emit({'id':rid,'result':{}})
    elif method=='command/exec':
        assert request['params']['sandboxPolicy']=={'type':'externalSandbox','networkAccess':'enabled'}
        assert request['params']['command']==['/bin/sh','-c','printf READY']
        failed=os.environ.get('FAKE_MODE')=='sandbox_failure'
        emit({'id':rid,'result':{'exitCode':1 if failed else 0,'stdout':'' if failed else 'READY',
            'stderr':'bwrap: Failed to make / slave: Permission denied '+os.environ.get('OPENAI_API_KEY','') if failed else ''}})
    elif method=='thread/start':
        assert request['params']['sandbox']=='danger-full-access'
        emit({'id':rid,'result':{'thread':{'id':'thread-test'}}})
    elif method=='turn/start':
        assert request['params']['outputSchema']['required']==['status','summary','proposal_path']
        assert request['params']['sandboxPolicy']=={'type':'externalSandbox','networkAccess':'enabled'}
        if os.environ.get('FAKE_MODE')=='timeout':
            time.sleep(30)
            continue
        answer={'status':'needs_input','summary':'WRONG' if os.environ.get('FAKE_MODE')=='wrong' else 'READY','proposal_path':None}
        # A fast completion may arrive before the turn/start acknowledgement.
        emit({'method':'item/completed','params':{'item':{'type':'agentMessage','text':json.dumps(answer)}}})
        emit({'id':rid,'result':{'turn':{'id':'turn-test'}}})
        turn={'id':'turn-test','status':'completed'}
        if os.environ.get('FAKE_MODE')=='auth':
            turn.update(status='failed',error={'message':'HTTP 401 invalid API key '+os.environ['OPENAI_API_KEY']})
        emit({'method':'turn/completed','params':{'turn':turn}})
'''


@pytest.fixture
def connection_site(tmp_path):
    home=tmp_path/'codex-home'
    home.mkdir()
    (home/'config.toml').write_text('model="deepseek-v4-pro"\nmodel_provider="custom"\n'
        '[model_providers.custom]\nbase_url="https://yeysai.com/v1"\nenv_key="OPENAI_API_KEY"\nwire_api="responses"\n')
    program=tmp_path/'codex'
    program.write_text('#!'+sys.executable+'\n'+SERVER)
    program.chmod(0o755)
    return {'user_root':str(tmp_path),'codex_home':str(home),'codex_bin':str(program)}


def environment(mode=''):
    return {**os.environ,'OPENAI_API_KEY':'opaque-connection-test-secret','FAKE_MODE':mode}


def test_success_uses_actual_ready_and_redacts_trace(connection_site,tmp_path):
    output=tmp_path/'success'
    result=check_connection(connection_site,output,env=environment())
    assert result['ok'] and result['inference_tested'] and result['reply']=='READY'
    assert result['base_url']=='https://yeysai.com/v1' and result['model']=='deepseek-v4-pro'
    assert 'opaque-connection-test-secret' not in ''.join(p.read_text() for p in output.iterdir())
    assert '[REDACTED]' in (output/'trace.jsonl').read_text()
    assert json.loads((output/'result.json').read_text())['ok']


def test_missing_key_does_not_launch(connection_site,tmp_path):
    connection_site['codex_bin']='/does/not/exist'
    result=check_connection(connection_site,tmp_path/'missing',env={})
    assert not result['ok'] and 'Missing environment variable' in result['detail']
    assert result['error']=='ValueError' and not result['inference_tested']


def test_offline_never_calls_model(connection_site,tmp_path):
    result=check_connection(connection_site,tmp_path/'offline',offline=True,env={})
    assert result['ok'] and result['status']=='offline_passed' and not result['inference_tested']
    assert 'turn/start' not in (tmp_path/'offline/trace.jsonl').read_text()


def test_sandbox_preflight_and_inference_are_separate(connection_site,tmp_path):
    result=check_connection(connection_site,tmp_path/'sandbox-success',sandbox_check=True,env=environment())
    assert result['ok'] and result['shell_execution_tested'] and result['inference_tested']
    assert not result['model_tool_calls_tested']


def test_blocked_sandbox_stops_before_model_request_and_redacts(connection_site,tmp_path):
    output=tmp_path/'sandbox-failure'
    result=check_connection(connection_site,output,sandbox_check=True,env=environment('sandbox_failure'))
    assert not result['ok'] and not result['shell_execution_tested'] and not result['inference_tested']
    assert '挂载权限' in result['hint'] and result['sandbox_returncode']==1
    trace=(output/'trace.jsonl').read_text()
    assert 'turn/start' not in trace and 'opaque-connection-test-secret' not in trace


def test_sandbox_can_be_checked_offline_without_key(connection_site,tmp_path):
    result=check_connection(connection_site,tmp_path/'sandbox-offline',sandbox_check=True,offline=True,env={})
    assert result['ok'] and result['shell_execution_tested'] and not result['inference_tested']


def test_auth_failure_is_actionable_and_secret_safe(connection_site,tmp_path):
    output=tmp_path/'auth'
    result=check_connection(connection_site,output,env=environment('auth'))
    assert not result['ok'] and not result['inference_tested']
    assert 'API key' in result['hint'] and '401' in result['detail']
    assert 'opaque-connection-test-secret' not in ''.join(p.read_text() for p in output.iterdir())


def test_completed_without_ready_fails(connection_site,tmp_path):
    result=check_connection(connection_site,tmp_path/'wrong',env=environment('wrong'))
    assert not result['ok'] and 'required READY' in result['detail']


def test_timeout_stops_only_its_child(connection_site,tmp_path):
    result=check_connection(connection_site,tmp_path/'timeout',timeout=1,env=environment('timeout'))
    assert not result['ok'] and result['error']=='TimeoutError'
    assert result['seconds']<12


@pytest.mark.parametrize('keep_env_key',[False,True])
def test_inline_token_works_without_export(connection_site,tmp_path,keep_env_key):
    config=tmp_path/'codex-home/config.toml'
    text=config.read_text()
    if not keep_env_key:text=text.replace('env_key="OPENAI_API_KEY"\n','')
    config.write_text(text+'experimental_bearer_token="opaque-inline-secret"\nrequires_openai_auth=true\n')
    result=check_connection(connection_site,tmp_path/'inline',env={})
    assert result['ok'] and result['auth_source']=='config.experimental_bearer_token'
    assert 'opaque-inline-secret' not in ''.join(p.read_text() for p in (tmp_path/'inline').iterdir())


def test_inline_auth_failure_redacts_inline_and_stale_environment(connection_site,tmp_path):
    config=tmp_path/'codex-home/config.toml'
    config.write_text(config.read_text()+'experimental_bearer_token="opaque-inline-secret"\n')
    env=environment('auth');env['FAKE_OLD_KEY']=env['OPENAI_API_KEY']
    output=tmp_path/'inline-auth'
    result=check_connection(connection_site,output,env=env)
    assert not result['ok'] and '401' in result['detail']
    text=''.join(p.read_text() for p in output.iterdir())
    assert 'opaque-inline-secret' not in text and 'opaque-connection-test-secret' not in text


def test_doctor_accepts_config_token_without_cli_oauth_login(connection_site,tmp_path,monkeypatch):
    from llmgo.rsi import cli
    config=tmp_path/'codex-home/config.toml'
    config.write_text(config.read_text()+'experimental_bearer_token="opaque-doctor-secret"\n')
    site={**connection_site,'repo':str(tmp_path),'base_model':str(tmp_path),
          'engine':str(tmp_path/'missing-engine'),'cctl_bin':'missing-cctl'}
    def run(argv,**kwargs):
        raise AssertionError('Inline provider credentials must not require CLI OAuth login')
    monkeypatch.setattr(cli.subprocess,'run',run)
    result=cli.doctor(site,codex_auth=True)
    assert next(c for c in result['checks'] if c['name']=='codex_provider_credentials')['ok']
    assert result['inference_tested'] is False
    assert 'opaque-doctor-secret' not in json.dumps(result)
