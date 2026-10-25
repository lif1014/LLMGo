"""Retries persist across restarts; integrity and submission boundaries stay strict."""
from pathlib import Path

import pytest

from test_rsi import controller
from llmgo.rsi.controller import Controller
from llmgo.rsi.contracts import atomic_json
from llmgo.rsi.meta_protocol import MetaFailure, service_failure


def planning(c):
    c.state['phase']='PLANNING';c.save()
    w=c.workspace()
    atomic_json(w/'proposal.json',{'hypothesis':'A finished test experiment','jobs':[],'evaluation':False})
    return w


def test_transient_meta_failure_survives_restart_then_resumes_same_workspace(controller,monkeypatch):
    c=controller;workspace=planning(c);clock=[1000.]
    monkeypatch.setattr('llmgo.rsi.controller.time.time',lambda:clock[0])
    monkeypatch.setattr('llmgo.rsi.controller.time.sleep',lambda n:clock.__setitem__(0,clock[0]+n))
    class Backend:
        calls=0
        def turn(self,w,p):
            self.calls+=1;assert w==workspace
            if self.calls==1:raise MetaFailure('temporary stream disconnect')
            return {'status':'ready','summary':'Done','proposal_path':str(w/'proposal.json')}
    b=Backend()
    c.run(backend=b,max_steps=1)
    assert c.state['controller_recovery']['next_retry_at']==1060
    from llmgo.rsi.contracts import read_json
    status=read_json(c.root/'controller-status.json')
    assert status['controller_recovery']['next_retry_at']==1060
    assert (c.root/'controller-status.json').stat().st_size<16384
    assert not c.state['jobs']
    restored=Controller(c.root)
    restored.run(backend=b,max_steps=1)  # waits; does not ask the model early
    assert b.calls==1 and clock[0]==1060
    restored.run(backend=b,max_steps=1)
    assert b.calls==2 and restored.state['phase']=='EXPERIMENT'
    assert restored.state['workspace']==str(workspace) and restored.state['round']==1
    assert not restored.state.get('controller_recovery')
    assert len(restored.memory.recent(10,'meta_result'))==1
    assert not restored.state['jobs']  # prepare alone cannot submit GPU work


def test_retry_budget_is_persisted_and_does_not_busy_loop(controller,monkeypatch):
    c=controller;planning(c);clock=[1000.]
    monkeypatch.setattr('llmgo.rsi.controller.time.time',lambda:clock[0])
    class Broken:
        calls=0
        def turn(self,*args):self.calls+=1;raise MetaFailure('upstream unavailable')
    b=Broken();delays=[]
    for _ in range(5):
        c=Controller(c.root);c.run(backend=b,max_steps=1)
        r=c.state['controller_recovery'];delays.append(r['next_retry_at']-clock[0]);clock[0]=r['next_retry_at']
    assert delays==[60,120,240,480,600]
    c=Controller(c.root);result=c.run(backend=b,max_steps=1)
    assert result['status']=='needs_input' and b.calls==6
    assert c.state['phase']=='NEEDS_INPUT' and c.state['controller_recovery']['status']=='blocked'
    assert not c.state['jobs']


@pytest.mark.parametrize('error',[service_failure('HTTP 401 Unauthorized'),
                                MetaFailure('Invalid sandbox mode',code='configuration',retryable=False)])
def test_permanent_errors_pause_with_actionable_state_without_retry(controller,error):
    c=controller;planning(c)
    class Broken:
        def turn(self,*args):raise error
    result=c.run(backend=Broken(),max_steps=1)
    assert result['status']=='needs_input'
    assert c.state['controller_recovery']['next_retry_at'] is None
    assert c.state['controller_recovery']['attempt']==1
    assert 'start_rsi.sh --resume' in c.state['needs_input']


def test_integrity_failure_retains_active_task_on_explicit_resume(controller,monkeypatch):
    c=controller;c.state.update(phase='WAITING',active_job='existing-job');c.save()
    def damaged():raise ValueError('Worker artifact identity/hash mismatch')
    monkeypatch.setattr(c,'step',damaged)
    assert c.run(max_steps=1)['status']=='needs_input'
    assert c.state['active_job']=='existing-job'
    c.resume()
    assert c.state['phase']=='WAITING' and c.state['active_job']=='existing-job'
    assert not c.state.get('controller_recovery')


def test_operator_interrupt_is_not_converted_into_an_automatic_retry(controller,monkeypatch):
    c=controller
    def interrupted():raise KeyboardInterrupt()
    monkeypatch.setattr(c,'step',interrupted)
    with pytest.raises(KeyboardInterrupt):c.run(max_steps=1)
    assert not c.state.get('controller_recovery')


def test_prepare_recovers_atomic_snapshot_after_crash_before_state_save(controller,monkeypatch):
    c=controller;workspace=planning(c)
    def crash():raise OSError('Shared disk unavailable during state save')
    monkeypatch.setattr(c,'save',crash)
    with pytest.raises(OSError):c.prepare(workspace/'proposal.json')
    folder=c.root/'candidates/r000001'
    assert (folder/'source/source-manifest.json').is_file()
    restored=Controller(c.root)
    assert restored.state['phase']=='PLANNING'
    candidate=restored.prepare(workspace/'proposal.json')
    assert candidate['id']=='r000001' and restored.state['phase']=='EXPERIMENT'
    assert len(restored.memory.recent(10,'hypothesis'))==1
    assert not restored.state['jobs']


def test_prepared_snapshot_is_never_overwritten_by_a_changed_draft(controller,monkeypatch):
    c=controller;workspace=planning(c)
    def crash():raise OSError('state save interrupted')
    monkeypatch.setattr(c,'save',crash)
    with pytest.raises(OSError):c.prepare(workspace/'proposal.json')
    (workspace/'new_code.py').write_text('different = True\n')
    restored=Controller(c.root)
    with pytest.raises(ValueError,match='differs'):restored.prepare(workspace/'proposal.json')
    assert not (c.root/'candidates/r000001/source/new_code.py').exists()
