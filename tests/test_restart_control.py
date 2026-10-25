"""Signal only an owned waiting Controller; its independent worker survives."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from llmgo.rsi.contracts import atomic_json
from llmgo.rsi.memory import CampaignLocked, MemoryStore
from llmgo.rsi.restart_control import stop_waiting_controller


@pytest.fixture
def waiting_controller(tmp_path):
    repo=tmp_path/'repo';(repo/'scripts').mkdir(parents=True)
    root=tmp_path/'campaigns/test';root.mkdir(parents=True)
    memory=MemoryStore(root)
    manifest=root/'jobs/j000001/manifest.json'
    atomic_json(manifest.with_name('submission.json'),{'task_id':123,'description':'llmgo-rsi:test:j000001:hash'})
    memory.save_state({'phase':'WAITING','active_job':'j000001','jobs':{
        'j000001':{'manifest':str(manifest),'sha256':'hash'}}})
    atomic_json(root/'meta-session.json',{'process':None,'status':'completed'})
    script=repo/'scripts/rsi.py'
    script.write_text('''import fcntl,json,os,socket,sys,time
from pathlib import Path
root=Path(sys.argv[3])
with (root/'controller.lock').open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 f.seek(0);f.truncate();f.write(json.dumps({'pid':os.getpid(),'host':socket.gethostname()}));f.flush()
 (root/'ready').touch()
 while True:time.sleep(.1)
''')
    proc=subprocess.Popen([sys.executable,'scripts/rsi.py','run','test',str(root)],cwd=repo)
    # This process models a cluster worker outside the Controller lifecycle.
    worker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
    deadline=time.monotonic()+5
    while not (root/'ready').exists():
        assert time.monotonic()<deadline
        time.sleep(.02)
    site={'user_root':str(tmp_path),'repo':str(repo),'campaign_root':str(root.parent)}
    yield site,root,proc,worker,memory
    for child in (proc,worker):
        if child.poll() is None:child.terminate()
        child.wait(timeout=5)


def test_owned_waiting_controller_stops_but_worker_and_intent_survive(waiting_controller):
    site,root,proc,worker,memory=waiting_controller
    before=(root/'jobs/j000001/submission.json').read_bytes()
    result=stop_waiting_controller(site,'test')
    proc.wait(timeout=5)
    assert result['stopped'] and result['task_id']==123 and worker.poll() is None
    assert (root/'jobs/j000001/submission.json').read_bytes()==before
    assert (Path(result['backup'])/'campaign.json').is_file()
    assert memory.recent(1,'operator_controller_stopped')[0]['body']['cluster_task_stopped'] is False
    assert not stop_waiting_controller(site,'test')['stopped']


@pytest.mark.parametrize('mode',['active_meta','planning','uncertain_intent','other_host','busy_executor'])
def test_restart_refuses_unsafe_boundary_without_signalling(waiting_controller,mode):
    site,root,proc,worker,memory=waiting_controller
    lock=None
    if mode=='active_meta':atomic_json(root/'meta-session.json',{'process':{'pid':999}})
    elif mode=='planning':
        state=memory.state();state['phase']='PLANNING';memory.save_state(state)
    elif mode=='uncertain_intent':atomic_json(root/'jobs/j000001/submission.json',{'description':'llmgo-rsi:test:j000001:hash'})
    elif mode=='other_host':
        # Keep the locked inode intact while changing its owner fixture metadata.
        with (root/'controller.lock').open('w') as stream:json.dump({'pid':proc.pid,'host':'some-other-host'},stream)
    else:
        path=root/'executor/lifecycle.lock';path.parent.mkdir(exist_ok=True)
        lock=path.open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        with pytest.raises((ValueError,CampaignLocked)):
            stop_waiting_controller(site,'test')
        assert proc.poll() is None and worker.poll() is None
    finally:
        if lock:lock.close()
