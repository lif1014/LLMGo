import json
import sys
from types import SimpleNamespace

import pytest

from llmgo.remote.cctl_env import cctl_environment, cctl_failure_detail


def test_missing_scheme_is_normalized_without_changing_profile_or_credentials(tmp_path):
    (tmp_path/'current-profile').write_text('prod')
    server=tmp_path/'profiles/prod/server';server.parent.mkdir(parents=True)
    server.write_text('cluster.example\n')
    env={'HOME':'unchanged','CYBERTRON_TOKEN':'private-test-token'}
    result=cctl_environment(env,config_root=tmp_path)
    assert result['CCTL_SERVER']=='https://cluster.example'
    assert result['CYBERTRON_TOKEN']=='private-test-token' and result['HOME']=='unchanged'
    assert 'CCTL_SERVER' not in env
    assert server.read_text()=='cluster.example\n'


def test_bare_legacy_alias_uses_verified_canonical_endpoint(tmp_path):
    assert cctl_environment({'CCTL_SERVER':'cybertron-hcloud.modelbest.co'},config_root=tmp_path)['CCTL_SERVER']=='https://cybertron.modelbest.co'


@pytest.mark.parametrize('url',['http://internal:8080','https://cybertron-hcloud.modelbest.co'])
def test_explicit_server_keeps_host_and_protocol(tmp_path,url):
    result=cctl_environment({'CCTL_SERVER':url},config_root=tmp_path)
    assert result['CCTL_SERVER']==url


def test_profile_precedence_and_legacy_fallback(tmp_path):
    server=tmp_path/'profiles/selected/server';server.parent.mkdir(parents=True)
    server.write_text('selected.example')
    (tmp_path/'server').write_text('legacy.example')
    assert cctl_environment({'CCTL_PROFILE':'selected'},config_root=tmp_path)['CCTL_SERVER']=='https://selected.example'
    assert cctl_environment({},config_root=tmp_path)['CCTL_SERVER']=='https://legacy.example'


def test_error_keeps_actionable_reason_and_redacts_credentials():
    result=SimpleNamespace(returncode=1,stderr='HTTP 401 invalid token=opaque-token Bearer abcdef',stdout='')
    detail=cctl_failure_detail(result,{'CYBERTRON_TOKEN':'opaque-token'})
    assert 'HTTP 401' in detail and 'opaque-token' not in detail and 'abcdef' not in detail


@pytest.mark.parametrize('pools,success', [([{'name':'model-autosft'}],True),([],False),
    ({'data':[{'name':'model-autosft'}],'success':True},True),
    ({'pools':[{'name':'resource-pools/525','alias':'model-autosft'}]},True),
    ({'data':[{'name':'other-pool'}],'success':True},False)])
def test_doctor_normalizes_server_and_distinguishes_missing_pool(tmp_path,monkeypatch,pools,success):
    from llmgo.rsi.cli import doctor
    import llmgo.rsi.cli as cli
    monkeypatch.setenv('CCTL_SERVER','cybertron-hcloud.modelbest.co')
    site={k:str(tmp_path) for k in ['user_root','repo','base_model','engine','codex_home']}
    site.update(codex_bin=sys.executable,cctl_bin=sys.executable)
    def run(argv,**kwargs):
        assert kwargs['env']['CCTL_SERVER']=='https://cybertron.modelbest.co'
        return SimpleNamespace(returncode=0,stdout=json.dumps(pools),stderr='')
    monkeypatch.setattr(cli.subprocess,'run',run)
    check=next(c for c in doctor(site,platform=True)['checks'] if c['name']=='cctl_auth_and_pool')
    assert check['ok']==success
    if not success:assert 'model-autosft' in check['detail']
    if pools==[]:assert 'profile current/list' in check['detail']


def test_doctor_surfaces_safe_failure_details(tmp_path,monkeypatch):
    import llmgo.rsi.cli as cli
    monkeypatch.setenv('CYBERTRON_TOKEN','opaque-token')
    site={k:str(tmp_path) for k in ['user_root','repo','base_model','engine','codex_home']}
    site.update(codex_bin=sys.executable,cctl_bin=sys.executable)
    monkeypatch.setattr(cli.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout='',stderr='HTTP 401 token=opaque-token'))
    result=cli.doctor(site,platform=True)
    check=next(c for c in result['checks'] if c['name']=='cctl_auth_and_pool')
    assert not check['ok'] and 'HTTP 401' in check['detail']
    assert 'opaque-token' not in json.dumps(result)
