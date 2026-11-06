"""Real queue/lifecycle with fake inference: all forty games must overlap."""
import asyncio
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmgo.rsi import evaluator
from llmgo.rsi.artifacts import source_snapshot
from llmgo.rsi.contracts import atomic_json, digest, file_hash, protocol, read_json
from llmgo.rsi.controller import Controller
from llmgo.rsi.ladder import opponent


def fixture(tmp_path,monkeypatch,gpu_count=8,fail=False,games=20):
    repo=tmp_path/'repo';repo.mkdir();model=tmp_path/'model';model.mkdir()
    atomic_json(model/'config.json',{});(model/'model.safetensors').write_bytes(b'fixture weights')
    site={'user_root':str(tmp_path),'repo':str(repo),'base_model':str(model),'engine':'fixture',
          'campaign_root':str(tmp_path/'campaigns'),'worktree_root':str(tmp_path/'workspaces'),
          'gpu_count':gpu_count,'eval_games_per_gpu':40,'eval_vllm_max_num_seqs':40,
          'eval_max_parallel_games':40,'eval_shared_levels':True,'image':'fixture'}
    source=source_snapshot(repo,tmp_path/'source')
    stats={'active':0,'peak':0,'active_gpu':Counter(),'peak_gpu':Counter(),'levels_gpu':defaultdict(set),
           'calls':0,'servers':[],'ready':set(),'barrier':None}
    class Server:
        def __init__(self,model,protocol,log,port,gpu,**kwargs):
            assert kwargs['max_num_seqs']==20
            self.gpu=gpu;self.base_url='fixture-'+gpu
            stats['servers'].append(self)
            self.starts=self.stops=0
        def start(self):self.starts+=1
        def stop(self):
            assert stats['active_gpu'][self.gpu]==0
            self.stops+=1
    def ready(servers,timeout):
        assert timeout is None
        stats['ready'].update(s.gpu for s in servers)
        return None
    helpers=SimpleNamespace(VllmServer=Server,_wait_servers=ready,
        _pick_ports=lambda first,count:list(range(first,first+count)),
        visible_gpu_ids=lambda:[str(i) for i in range(gpu_count)])
    async def play(p,url,cards,out,*,indices):
        index=next(iter(indices));level=p['protocol']['opponent']['level'];gpu=url.split('-')[-1]
        if fail and level==3 and index==0:raise RuntimeError('fixture engine failure')
        assert len(stats['ready'])==gpu_count  # no game before every replica is ready
        stats['calls']+=1;stats['active']+=1;stats['active_gpu'][gpu]+=1
        stats['peak']=max(stats['peak'],stats['active'])
        stats['peak_gpu'][gpu]=max(stats['peak_gpu'][gpu],stats['active_gpu'][gpu])
        stats['levels_gpu'][gpu].add(level)
        if not fail:
            if stats['barrier'] is None:stats['barrier']=asyncio.Event()
            if stats['active']==min(2*games,20*gpu_count):stats['barrier'].set()
            await asyncio.wait_for(stats['barrier'].wait(),timeout=5)
        else:await asyncio.sleep(.001)
        gid=f"{p.get('game_id_prefix',p['job_id'])}-{index:02d}"
        events=Path(p.get('games_output',out))/'games'/gid/'events.jsonl'
        events.parent.mkdir(parents=True,exist_ok=True);events.write_text('{"type":"fixture"}\n')
        atomic_json(events.with_name('summary.json'),{'completed':True,'student_color':'black' if index%2 else 'white',
            'result':0,'reason':'draw','policy_version':p['candidate_id']})
        row={'game_id':gid,'pair_id':index//2,'model_seat':'black' if index%2 else 'white',
             'result':'draw','index':index,'seed':p['seed_schedule'][index//2],
             'evaluation_key':evaluator.evaluation_key(p),'raw_events':str(events),
             'event_hashes':{'raw_events':file_hash(events)}}
        row['integrity']=digest(row);atomic_json(out/'game_results'/f'{index:02d}.json',row)
        stats['active']-=1;stats['active_gpu'][gpu]-=1
        return [row]
    monkeypatch.setattr(evaluator,'runtime_helpers',lambda _:helpers)
    monkeypatch.setattr(evaluator,'engine_identity',lambda _,spec:{'engine':'fixture','spec':spec})
    monkeypatch.setattr(evaluator,'play_batch',play)
    root=repo/'var/runs/shared'
    payloads=[]
    for level in (2,3):
        spec=opponent(level)
        payloads.append({'site':site,'model_path':str(model),'source':source,'candidate_source':source,
            'reference_source':source,'candidate_id':'initial','job_id':'j000001','role':'baseline',
            'skills_sha256':None,'track':'player','output':str(root/'evaluations'/f'L{level}'),
            'panel_output':str(root),'games_output':str(root),'game_id_prefix':f'j000001-L{level}',
            'protocol':protocol(spec,'fixture',games=games),'seed_schedule':list(range(games//2)),
            'comparison_id':f'validation{games}:L{level}:b0','engine_identity':{'engine':'fixture','spec':spec}})
    return site,payloads,stats


@pytest.mark.parametrize('gpu_count',[1,8])
@pytest.mark.parametrize('games',[20,40])
def test_shared_queue_respects_twenty_lanes_and_batch_size(tmp_path,monkeypatch,gpu_count,games):
    site,payloads,stats=fixture(tmp_path,monkeypatch,gpu_count,games=games)
    peak=min(2*games,20*gpu_count)
    paths=evaluator.evaluate_many(payloads)
    assert stats['peak']==peak and stats['calls']==2*games
    assert len(stats['servers'])==gpu_count
    assert all(s.starts==s.stops==1 for s in stats['servers'])
    assert all(value==peak//gpu_count for value in stats['peak_gpu'].values())
    assert all(levels=={2,3} for levels in stats['levels_gpu'].values())
    assert [read_json(path)['statistics']['games'] for path in paths]==[games,games]
    assert [read_json(path)['protocol']['opponent']['level'] for path in paths]==[2,3]
    progress=read_json(Path(payloads[0]['panel_output'])/'progress.json')
    assert progress['parallelism']==peak and progress['completed_games']==2*games
    assert progress['games_per_gpu']==peak//gpu_count
    # A completed retry reuses verified games and starts no extra vLLM replica.
    evaluator.evaluate_many(payloads)
    assert stats['calls']==2*games and len(stats['servers'])==gpu_count


def test_failed_level_keeps_games_but_publishes_no_official_panel(tmp_path,monkeypatch):
    site,payloads,stats=fixture(tmp_path,monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='Parallel evaluation incomplete'):
        evaluator.evaluate_many(payloads)
    assert len(list(Path(payloads[0]['output']).glob('game_results/*.json')))==20
    assert len(list(Path(payloads[1]['output']).glob('game_results/*.json')))==19
    assert all(not (Path(p['output'])/'metrics.json').exists() for p in payloads)
    assert all(server.stops==1 for server in stats['servers'])


def test_shared_pool_rejects_different_models_before_serving(tmp_path,monkeypatch):
    site,payloads,stats=fixture(tmp_path,monkeypatch)
    payloads[1]['model_path']=str(tmp_path/'another-model')
    with pytest.raises(ValueError,match='same model'):
        evaluator.evaluate_many(payloads)
    assert not stats['servers']


def test_shared_pool_rejects_overlapping_result_directories(tmp_path,monkeypatch):
    site,payloads,stats=fixture(tmp_path,monkeypatch)
    payloads[1]['output']=payloads[0]['output']
    with pytest.raises(ValueError,match='distinct levels and output'):
        evaluator.evaluate_many(payloads)
    assert not stats['servers']


def test_baseline_is_one_job_with_two_independent_forty_game_metrics(tmp_path,monkeypatch):
    site,_,stats=fixture(tmp_path,monkeypatch,games=40)
    c=Controller.create(site,'shared-baseline')
    c.engine_probe=evaluator.engine_identity
    c.step();job=c.state['jobs'][c.state['active_job']]
    payload=read_json(job['manifest'])['payload']
    assert job['kind']=='evaluate_panel' and job['levels']==[2,3]
    assert payload['protocol']['games']==80
    from llmgo.rsi import worker
    monkeypatch.setenv('LLMGO_RSI_WORKER','0')
    monkeypatch.setattr(worker,'worker_environment',lambda _: {})
    result=worker.execute(job['manifest'],job['sha256'])
    path=Path(result['artifact'])
    assert read_json(Path(payload['output'])/'progress.json')['completed_games']==80
    c._consume(job,{'path':str(path),'sha256':file_hash(path)},payload)
    assert c.state['evaluation_cursor']==2
    assert [row['level'] for row in c.state['pending_evaluations']]==[2,3]
    c.state['active_job']=None;c.save();c.step()
    assert c.state['phase']=='PLANNING' and len(c.state['jobs'])==1
    assert c.state['baseline_panel']['player']['games']==80
    assert stats['peak']==80
    from fastapi.testclient import TestClient
    from llmgo.server.app import app
    monkeypatch.setenv('LLMGO_RSI_ROOT',site['campaign_root'])
    monkeypatch.setenv('LLMGO_RUN_ROOT',str(Path(site['repo'])/'var/runs'))
    client=TestClient(app)
    assert len(client.get('/api/rsi/campaigns/shared-baseline/games').json())==80
    listing=client.get('/api/rsi/campaigns/shared-baseline/job-files',params={'job_id':job['id']}).json()
    assert listing['game_results']==listing['formal_games']==80


def test_validation_upgrade_preserves_paused_workspace_and_legacy_identity(tmp_path,monkeypatch):
    site,_,_=fixture(tmp_path,monkeypatch)
    c=Controller.create(site,'migration');c.state.update(phase='PLANNING');c.save()
    ws=c.workspace();(ws/'research.py').write_text('# keep research')
    initial=c.state['initial'].copy();canonical=dict(c.site)
    with pytest.raises(ValueError,match='paused idle'):c.upgrade_validation()
    atomic_json(c.root/'research-control.json',{'schema_version':1,'paused':True,'revision':1})
    atomic_json(c.root/'meta-session.json',{'status':'paused','process':None})
    with c.memory.lock():receipt=c.upgrade_validation()
    assert receipt['policy']['games_per_level']==40
    assert c.state['workspace']==str(ws) and (ws/'research.py').read_text()=='# keep research'
    assert c.state['initial']==initial and c.site==canonical and c.state['active_job'] is None
    assert read_json(ws/'context.json')['state']['validation_policy']['vllm_max_num_seqs']==20
    assert all(receipt['policy'][key] is None for key in (
        'request_timeout_seconds','service_ready_timeout_seconds','harness_timeout_seconds','uci_timeout_seconds'))
