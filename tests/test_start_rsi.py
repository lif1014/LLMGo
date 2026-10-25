"""Exercise the launcher with owned fake services, without API/GPU jobs."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest

from llmgo.rsi.contracts import REPO


@pytest.fixture
def launcher(tmp_path):
    repo=tmp_path/'repo';scripts=repo/'scripts';scripts.mkdir(parents=True)
    shutil.copy(REPO/'scripts/start_rsi.sh',scripts/'start_rsi.sh')
    remote=repo/'src/llmgo/remote';remote.mkdir(parents=True)
    (remote.parent/'__init__.py').touch();(remote/'__init__.py').touch()
    shutil.copy(REPO/'src/llmgo/remote/cctl_env.py',remote/'cctl_env.py')
    root=tmp_path/'owned';root.mkdir()
    site={'user_root':str(root),'campaign_root':str(root/'campaigns'),
          'codex_home':str(root/'codex'),'runtime_root':str(root/'runtime'),'executor_mode':'embedded'}
    config=repo/'configs/rsi/site.beijing.json';config.parent.mkdir(parents=True)
    config.write_text(json.dumps(site))
    (scripts/'rsi.py').write_text('''import fcntl,json,os,sys,time
from pathlib import Path
site=json.loads(Path(sys.argv[2]).read_text());root=Path(site['user_root'])
command=sys.argv[3];campaign=sys.argv[-1]
with (root/'calls.jsonl').open('a') as f:f.write(json.dumps({'command':command,'argv':sys.argv[3:]})+'\\n')
if command=='doctor': print('{"ok":true}')
elif command=='init':
 p=Path(site['campaign_root'])/campaign;p.mkdir(parents=True)
 (p/'campaign.json').write_text('{"keep":"existing evidence"}')
elif command in ('run','executor'):
 p=Path(site['campaign_root'])/campaign
 lock=p/('executor/service.lock' if command=='executor' else 'controller.lock')
 with lock.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  (root/(command+'.pid')).write_text(str(os.getpid()))
  while not (root/('stop-'+command)).exists(): time.sleep(.05)
elif command=='status':print('{"phase":"BASELINE"}')
''')
    (scripts/'check_meta_connection.py').write_text("import os\nraise SystemExit(3 if os.environ.get('FAKE_BAD_CHECK') else 0)\n")
    (scripts/'run_web.sh').write_text('exit 97\n')
    env={**os.environ,'RSI_PYTHON':sys.executable}
    def run(*arguments,extra=None):
        return subprocess.run(['bash',str(scripts/'start_rsi.sh'),*arguments],env={**env,**(extra or {})},
                              capture_output=True,text=True,timeout=20)
    yield root,site,config,run
    # Only the fake services created by this fixture observe these stop files.
    for name in ('run','executor'):(root/('stop-'+name)).touch()


def calls(root):
    return [json.loads(line)['command'] for line in (root/'calls.jsonl').read_text().splitlines()]


def wait_unlocked(path):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        with path.open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(.05);continue
            return
    raise AssertionError('Owned controller did not release its launch lock')


@pytest.mark.parametrize('mode',['embedded','external'])
def test_repeat_start_and_resume_preserve_campaign_and_do_not_duplicate_services(launcher,mode):
    root,site,config,run=launcher
    site['executor_mode']=mode;config.write_text(json.dumps(site))
    first=run();assert first.returncode==0,first.stderr
    assert '后台启动' in first.stdout
    state=root/'campaigns/chess-autonomous-v2/campaign.json';before=state.read_bytes()
    repeated=run();assert repeated.returncode==0 and '同名loop' in repeated.stdout
    assert calls(root).count('init')==calls(root).count('run')==1
    if mode=='external':assert calls(root).count('executor')==1
    (root/'stop-run').touch()
    wait_unlocked(root/'runtime/launcher/chess-autonomous-v2.lock')
    (root/'stop-run').unlink()
    resumed=run();assert resumed.returncode==0,resumed.stderr
    assert calls(root).count('init')==1 and calls(root).count('run')==2
    if mode=='external':assert calls(root).count('executor')==1
    assert state.read_bytes()==before


def test_failed_connection_check_never_initializes_or_runs(launcher):
    root,site,config,run=launcher
    result=run(extra={'FAKE_BAD_CHECK':'1'})
    assert result.returncode==3 and calls(root)==['doctor']
    assert not (root/'campaigns').exists()


def test_occupied_web_port_is_left_alone(launcher):
    root,site,config,run=launcher
    with socket.socket() as service:
        service.bind(('0.0.0.0',0));service.listen()
        port=service.getsockname()[1]
        result=run('--web',extra={'LLMGO_WEB_PORT':str(port)})
        assert result.returncode==0,result.stderr
        assert '已占用' in result.stdout
        assert service.getsockname()[1]==port


def test_failed_web_start_is_reported_without_stopping_controller(launcher):
    root,site,config,run=launcher
    with socket.socket() as free:
        free.bind(('127.0.0.1',0));port=free.getsockname()[1]
    result=run('--web',extra={'LLMGO_WEB_PORT':str(port)})
    assert result.returncode==1 and '网页启动失败' in result.stderr
    assert calls(root).count('run')==1 and (root/'run.pid').exists()


def test_explicit_resume_runs_before_controller_without_upgrading(launcher):
    root,site,config,run=launcher
    campaign=root/'campaigns/chess-autonomous-v2'
    campaign.mkdir(parents=True)
    (campaign/'campaign.json').write_text('{"phase":"NEEDS_INPUT"}')
    result=run('--resume')
    assert result.returncode==0,result.stderr
    assert calls(root)==['doctor','resume','run']
    assert '已恢复暂停' in result.stdout


def test_upgrade_and_resume_are_mutually_exclusive(launcher):
    root,site,config,run=launcher
    result=run('--upgrade','--resume')
    assert result.returncode==2
    assert not (root/'calls.jsonl').exists()


def test_recover_meta_runs_before_controller_without_resuming_or_upgrading(launcher):
    root,site,config,run=launcher
    campaign=root/'campaigns/chess-autonomous-v2'
    campaign.mkdir(parents=True);(campaign/'campaign.json').write_text('{"phase":"PLANNING"}')
    result=run('--recover-meta')
    assert result.returncode==0,result.stderr
    assert calls(root)==['doctor','recover-meta','run']
    assert '无需重做研究' in result.stdout


@pytest.mark.parametrize('option',['--resume','--upgrade'])
def test_recover_meta_conflicts_with_other_migrations(launcher,option):
    root,site,config,run=launcher
    result=run('--recover-meta',option)
    assert result.returncode==2 and not (root/'calls.jsonl').exists()
