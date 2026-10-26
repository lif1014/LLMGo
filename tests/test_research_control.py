"""A persistent operator hold must never trigger another model call/submission."""
import json
from pathlib import Path

import pytest

from test_rsi import controller, Platform, skill_proposal, consume_evaluation
from llmgo.rsi.contracts import atomic_json, read_json
from llmgo.rsi.controller import Controller
from llmgo.rsi.research_control import (ResearchPaused, launcher_campaign, read_control,
                                       require_research, set_control)


def test_hold_survives_reload_and_requires_explicit_resume(controller):
    c = controller
    set_control(c.root, True, "PPO maintenance")
    c.state['phase'] = 'PLANNING'; c.save()
    class Forbidden:
        def turn(self, *args): raise AssertionError('Paused research called the model')
    Controller(c.root).run(backend=Forbidden(), max_steps=2)
    assert read_control(c.root)['paused']
    assert not c.state['jobs']
    assert read_json(c.root/'controller-status.json')['research_control']['paused']
    assert set_control(c.root, True, 'PPO maintenance')['revision'] == 1
    assert set_control(c.root, False, 'Operator resumes')['revision'] == 2
    assert not require_research(c.root)['paused']


def test_paused_controller_collects_existing_gpu_task_without_submitting_next(controller):
    c = controller; platform = Platform(); c.executor = platform
    c.step()  # prepare baseline task
    c.step()  # submit once
    task = c.state['active_job']; assert c.state['jobs'][task].get('task_id')
    job = c.state['jobs'][task]
    atomic_json(Path(job['manifest']).with_name('submission.json'), {
        'task_id':job['task_id'], 'description':f"llmgo-rsi:test:{task}:{job['sha256']}"})
    consume_evaluation(c)
    c.lifecycle.clock = lambda: 10**15  # advance past the required GPU polling interval
    set_control(c.root, True, 'Keep the worker, hold research')
    c.step()  # collect original task
    assert c.state['active_job'] is None
    assert c.step()['status'] == 'research_paused'
    assert platform.calls == 1 and len(c.state['jobs']) == 1


def test_hold_before_submit_preserves_prepared_task_and_intent(controller):
    c = controller; platform = Platform(); c.executor = platform
    c.step(); task = c.state['active_job']
    set_control(c.root, True, 'Hold new submissions')
    result = c.step()
    assert result['status'] == 'research_paused' and result['submission_held']
    assert platform.calls == 0 and c.state['active_job'] == task
    set_control(c.root, False, 'Explicit resume')
    c.step()
    assert platform.calls == 1 and c.state['active_job'] == task


def test_hold_with_missing_intent_never_recreates_a_recorded_task(controller):
    c = controller; platform = Platform(); c.executor = platform
    c.step(); c.step()
    assert c.state['jobs'][c.state['active_job']].get('task_id')
    set_control(c.root, True, 'Missing intent must be reconciled')
    assert c.step()['submission_held']
    assert platform.calls == 1


def test_pause_race_after_meta_result_never_prepares_a_candidate(controller):
    c = controller; c.state['phase'] = 'PLANNING'; c.save()
    class Backend:
        def turn(self, workspace, prompt):
            atomic_json(workspace/'proposal.json', {'hypothesis':'Ready', 'jobs':[], 'evaluation':False})
            set_control(c.root, True, 'Pause arriving with a result')
            return {'status':'ready', 'summary':'Ready', 'proposal_path':str(workspace/'proposal.json')}
    c.run(backend=Backend(), max_steps=1)
    assert c.state['phase'] == 'PLANNING' and not c.state['jobs'] and c.state.get('candidate') is None
    assert not c.state.get('controller_recovery')


def test_native_launcher_hold_happens_before_credentials_or_subprocess(controller, monkeypatch):
    from llmgo.rsi import codex_bridge
    import llmgo.rsi.research_control as control
    set_control(controller.root, True, 'Maintenance')
    monkeypatch.setattr(control, 'launcher_campaign', lambda cwd: controller.root)
    monkeypatch.setattr(codex_bridge.subprocess, 'run', lambda *a, **k: pytest.fail('Native binary started'))
    assert codex_bridge.launch(['app-server']) == 78


def test_backend_hold_is_not_classified_as_service_failure(controller):
    from llmgo.rsi.meta import CodexBackend
    set_control(controller.root, True, 'Maintenance')
    backend = CodexBackend(controller.site, controller.memory, popen=lambda *a, **k: pytest.fail('Spawned child'))
    with pytest.raises(ResearchPaused): backend.turn(controller.root, 'Start')
    assert not controller.memory.recent(1, 'meta_failure')


def test_workspace_guard_is_scoped_to_campaign_and_site(controller):
    c = controller; root = Path(c.site['user_root'])
    work = root/'workspaces/test/round-000001'; work.mkdir(parents=True)
    assert launcher_campaign(work, c.site) == c.root
    assert launcher_campaign(root/'other-project', c.site) is None


def test_invalid_control_fails_closed(controller):
    atomic_json(controller.root/'research-control.json', {'schema_version':1, 'paused':'yes', 'revision':1})
    with pytest.raises(ValueError): require_research(controller.root)


def test_active_turn_is_interrupted_as_operator_pause(controller, monkeypatch):
    from llmgo.rsi.meta import CodexBackend
    backend = CodexBackend(controller.site, controller.memory)
    backend.session['thread_id'] = 'owned-thread'
    calls = []
    monkeypatch.setattr(backend, '_rpc', lambda method, params, **kw: calls.append((method, params)))
    set_control(controller.root, True, 'Operator repair')
    with pytest.raises(ResearchPaused): backend._wait_turn('owned-turn')
    assert calls == [('turn/interrupt', {'threadId':'owned-thread','turnId':'owned-turn'})]


def test_short_pause_checks_do_not_emit_one_heartbeat_per_second(controller, monkeypatch):
    from llmgo.rsi.meta import CodexBackend
    backend = CodexBackend(controller.site, controller.memory)
    backend.proc = type('Owned', (), {'poll':lambda self:None})()
    now = [100.]
    monkeypatch.setattr('llmgo.rsi.meta.time.monotonic', lambda: now[0])
    for _ in range(10):
        backend._pump(timeout=.001); now[0] += 1
    assert len(controller.memory.recent(20, 'meta_heartbeat')) == 1
