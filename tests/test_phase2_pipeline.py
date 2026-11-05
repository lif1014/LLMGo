"""Local Phase 2 workflow; fake inference/platform, real harness subprocesses."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys

import chess
import pytest

from llmgo.rsi import evaluator, phase2
from llmgo.rsi.contracts import REPO, atomic_json, digest, read_json
from llmgo.rsi.controller import Controller
from llmgo.training.chess_r1 import messages
from test_shared_evaluation import fixture as evaluation_fixture
from test_rsi_executor import Platform


def setup_site(tmp_path,monkeypatch):
    site,_,stats=evaluation_fixture(tmp_path,monkeypatch,games=40)
    repo=Path(site['repo'])
    shutil.copytree(REPO/'examples/phase2',repo/'examples/phase2')
    home=tmp_path/'phase1-meta';home.mkdir()
    (home/'config.toml').write_text('model="fixture"\n')
    (home/'AGENTS.md').write_text('Phase 1: no harness, empty skills')
    site['codex_home']=str(home)
    return site,stats


@pytest.mark.parametrize('use_checkpoint',[False,True])
def test_phase2_initial_model_and_meta_home_are_independent(tmp_path,monkeypatch,use_checkpoint):
    site,_=setup_site(tmp_path,monkeypatch)
    checkpoint=tmp_path/'phase1-checkpoint';checkpoint.mkdir()
    atomic_json(checkpoint/'config.json',{});(checkpoint/'model.safetensors').write_bytes(b'trained fixture')
    c=Controller.create(site,'phase2',scope=phase2.SCOPE,checkpoint=checkpoint if use_checkpoint else None)
    expected=str(checkpoint) if use_checkpoint else site['base_model']
    assert c.state['initial']['model_path']==expected
    assert c.state['model_origin']['model_path']==expected
    home=Path(c.site['codex_home'])
    assert home!=Path(site['codex_home']) and (home/'config.toml').read_text()=='model="fixture"\n'
    assert (Path(site['codex_home'])/'AGENTS.md').read_text()=='Phase 1: no harness, empty skills'
    prompt=phase2.meta_instructions(c.state)
    assert '本阶段成果判为 0 分' in prompt and expected in prompt
    assert (home/'AGENTS.md').read_text()==(REPO/'docs/phase2-meta-prompt.md').read_text()
    assert not (Path(c.state['initial']['source']['root'])/'harness.py').exists()


def test_phase2_accepts_freeform_skills_and_generic_jobs_without_training_detection(tmp_path,monkeypatch):
    site,_=setup_site(tmp_path,monkeypatch)
    c=Controller.create(site,'open-phase2',scope=phase2.SCOPE);c.state['phase']='PLANNING';c.save();ws=c.workspace()
    atomic_json(ws/'skills.json',{'arbitrary':['a skill','another skill']})
    # This is admission only: no command is executed, no training takes place.
    job={'argv':['{python}','my_train.py']}
    atomic_json(ws/'proposal.json',{'hypothesis':'fixture','jobs':[job],'evaluation':False})
    candidate=c.prepare(ws/'proposal.json')
    assert candidate['jobs']==[job] and candidate['skills_sha256']


def test_harness_can_import_resources_and_receive_complete_history(tmp_path):
    (tmp_path/'helper.py').write_text('def guide(): return "protect your king"\n')
    (tmp_path/'harness.py').write_text('''from helper import guide

def build_messages_with_history(request):
    return request['default_messages']+[{'role':'user','content':guide()+" | "+" ".join(request['moves_san'])+" | "+" ".join(request['moves_uci'])}]
''')
    board=chess.Board();board.push_uci('e2e4');board.push_uci('e7e5')
    default=messages(board.fen(),enable_thinking=False)
    p={'candidate_source':{'root':str(tmp_path)},'research_scope':phase2.SCOPE}
    rendered=evaluator.candidate_messages(p,board.fen(),board,default)
    assert rendered[:2]==default
    assert rendered[-1]['content']=='protect your king | e4 e5 | e2e4 e7e5'
    assert evaluator.candidate_messages({**p,'track':'model'},board.fen(),board,default)==default
    assert evaluator.candidate_messages({**p,'role':'baseline'},board.fen(),board,default)==default


def test_phase2_bridge_has_no_arbitrary_card_or_message_limits(tmp_path):
    (tmp_path/'harness.py').write_text("def build_messages(*args): return [{'role':'user','content':'x'*70000}]*21\n")
    board=chess.Board();p={'candidate_source':{'root':str(tmp_path)},'research_scope':phase2.SCOPE}
    rendered=evaluator.candidate_messages(p,board.fen(),board,messages(board.fen(),enable_thinking=False))
    assert len(rendered)==21 and len(rendered[0]['content'])==70000


def test_phase2_full_local_loop_delivers_best_harness_and_preserves_phase1(tmp_path,monkeypatch):
    site,stats=setup_site(tmp_path,monkeypatch)
    p1=Controller.create(site,'phase1')
    original=(p1.root/'campaign.json').read_bytes()
    c=Controller.create(site,'phase2-loop',scope=phase2.SCOPE,max_model_len=49152)
    from llmgo.rsi import worker
    monkeypatch.setattr(worker,'worker_environment',lambda _:dict(os.environ))
    monkeypatch.setenv('LLMGO_RSI_WORKER','0')
    fake_play=evaluator.play_batch;rendered=[]
    async def play(payload,url,cards,output,*,indices):
        index=next(iter(indices))
        if index==0:
            board=chess.Board();base=messages(board.fen(),enable_thinking=False)
            prompt=await asyncio.to_thread(evaluator.candidate_messages,payload,board.fen(),board,base)
            rendered.append((payload['candidate_id'],prompt))
        rows=await fake_play(payload,url,cards,output,indices=indices)
        if payload['candidate_id']!='initial':
            row=rows[0];side=row['model_seat']
            winner=side if payload['candidate_id']=='r000001' else ('black' if side=='white' else 'white')
            row['result']=winner+'_win';row.pop('integrity');row['integrity']=digest(row)
            atomic_json(output/'game_results'/f'{index:02d}.json',row)
        return rows
    monkeypatch.setattr(evaluator,'play_batch',play)
    class LocalPlatform(Platform):
        status='Succeeded'
        def __init__(self):super().__init__();self.status='Succeeded'
        def submit(self,manifest):
            worker.execute(manifest)
            return super().submit(manifest)
    platform=LocalPlatform();c.executor=platform;c.engine_probe=evaluator.engine_identity
    class Meta:
        turns=0
        def turn(self,workspace,prompt):
            self.turns+=1
            assert 'Phase 2' in prompt and '禁止任何训练' in prompt
            assert read_json(workspace/'context.json')['state']['research_scope']==phase2.SCOPE
            text=(workspace/'harness.py').read_text()
            (workspace/'harness.py').write_text(text+f'\n# candidate {self.turns}\n')
            atomic_json(workspace/'skills.json',{'freeform':f'candidate {self.turns}'})
            atomic_json(workspace/'proposal.json',{'hypothesis':f'harness {self.turns}',
                'jobs':[],'evaluation':{'levels':[2,3]},'decision':'continue' if self.turns==1 else 'finish',
                'final_candidate':'best'})
            return {'status':'ready','summary':'fixture','proposal_path':str(workspace/'proposal.json')}
    meta=Meta();result=c.run(backend=meta,max_steps=30,poll_seconds=.001)
    assert result['status']=='completed' and meta.turns==2 and platform.creates==3
    bundle=read_json(c.root/'final-candidate.json')
    assert bundle['candidate_id']=='r000001' and bundle['model_path']==site['base_model']
    assert bundle['max_model_len']==49152 and len(bundle['evaluations'])==2
    assert '# candidate 1' in (Path(bundle['source']['root'])/'harness.py').read_text()
    assert '# candidate 2' not in (Path(bundle['source']['root'])/'harness.py').read_text()
    curve=read_json(c.root/'prompt-curve.json')['points']
    assert [p['score'] for p in curve]==[.5,1.,0.]
    assert [p['best_observed'] for p in curve]==[.5,1.,1.]
    assert all(sum(s['games'] for s in point['levels'].values())==80 for point in curve)
    assert len(stats['servers'])==24 and stats['calls']==240
    assert (p1.root/'campaign.json').read_bytes()==original
    assert any('Additional chess guidance' in row[1][-1]['content'] for row in rendered if row[0]=='r000001')
    assert all('Additional chess guidance' not in row[1][-1]['content'] for row in rendered if row[0]=='initial')
    assert not list((Path(site['repo'])/'var/runs').glob('*/grpo'))
    from fastapi.testclient import TestClient
    from llmgo.server.app import app
    monkeypatch.setenv('LLMGO_RSI_ROOT',site['campaign_root'])
    client=TestClient(app)
    overview=client.get('/api/rsi/campaigns/phase2-loop/overview').json()
    assert overview['research_scope']==phase2.SCOPE and overview['final_candidate']['candidate_id']=='r000001'
    response=client.get('/api/rsi/campaigns/phase2-loop/artifact',params={'path':'prompt-curve.json'})
    assert len(json.loads(response.json()['text'])['points'])==3


def test_phase2_context_length_reaches_vllm_command(tmp_path,monkeypatch):
    from llmgo.rsi.service import VllmServer
    from llmgo.rsi.deepseek_final import PROTOCOL
    calls=[]
    monkeypatch.setattr('llmgo.rsi.service.subprocess.Popen',lambda argv,**kwargs:calls.append(argv))
    server=VllmServer(tmp_path/'model',PROTOCOL,tmp_path/'log',8000,'0',max_model_len=49152)
    server.start();server.stop()
    assert calls[0][calls[0].index('--max-model-len')+1]=='49152'


def test_phase2_cli_reuses_its_isolated_site_for_status(tmp_path,monkeypatch,capsys):
    from llmgo.rsi import cli
    site,_=setup_site(tmp_path,monkeypatch)
    monkeypatch.setattr(cli,'load_site',lambda _:dict(site))
    assert cli.main(['init','cli-p2','--scope',phase2.SCOPE,'--max-model-len','49152'])==0
    created=json.loads(capsys.readouterr().out)['state']
    assert created['inference_max_model_len']==49152
    assert cli.main(['status','cli-p2'])==0
    assert json.loads(capsys.readouterr().out)['state']['research_scope']==phase2.SCOPE


def test_real_meta_backend_gets_phase2_instructions_and_home(tmp_path,monkeypatch):
    from llmgo.rsi import meta
    site,_=setup_site(tmp_path,monkeypatch)
    program=tmp_path/'fake-codex'
    program.write_text('#!'+sys.executable+'\n'+r'''
import json, os, sys
from pathlib import Path

def emit(value): print(json.dumps(value),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if 'id' not in request: continue
    rid=request['id']; method=request['method'];p=request['params']
    if method=='initialize':emit({'id':rid,'result':{}})
    elif method=='thread/start':
        Path('received-instructions.txt').write_text(p['developerInstructions'])
        Path('received-home.txt').write_text(os.environ['CODEX_HOME'])
        emit({'id':rid,'result':{'thread':{'id':'phase2-thread','turns':[]}}})
    elif method=='turn/start':
        emit({'id':rid,'result':{'turn':{'id':'phase2-turn'}}})
        answer={'status':'ready','summary':'local fixture','proposal_path':str(Path('proposal.json').resolve())}
        emit({'method':'item/completed','params':{'turnId':'phase2-turn','item':{'type':'agentMessage','text':json.dumps(answer)}}})
        emit({'method':'turn/completed','params':{'turn':{'id':'phase2-turn','status':'completed'}}})
''')
    program.chmod(0o755);site['codex_bin']=str(program)
    c=Controller.create(site,'meta-p2',scope=phase2.SCOPE);c.state['phase']='PLANNING';c.save();ws=c.workspace()
    backend=meta.CodexBackend(c.site,c.memory)
    answer=backend.turn(ws,phase2.turn_prompt(c.root,ws,0))
    assert answer['status']=='ready'
    instructions=(ws/'received-instructions.txt').read_text()
    assert '禁止任何模型训练或权重更新' in instructions
    assert '本阶段成果判为 0 分' in instructions
    assert '当前是Phase 1' not in instructions and 'skills.json必须保持空数组' not in instructions
    assert (ws/'received-home.txt').read_text()==c.site['codex_home']
    assert read_json(c.root/'meta-session.json')['thread_id']=='phase2-thread'


def test_phase2_final_game_renders_frozen_harness(tmp_path,monkeypatch):
    from llmgo.rsi import deepseek_final as module
    candidate=tmp_path/'candidate';candidate.mkdir()
    (candidate/'harness.py').write_text("def build_messages(*args): return args[-1]+[{'role':'user','content':'Frozen Phase 2 skill'}]\n")
    calls=[]
    class Episode:
        def __init__(self,*args,**kwargs):
            self.board=chess.Board();self.done=False;self.output=args[5];self.output.mkdir(parents=True)
            (self.output/'events.jsonl').write_text('')
        def advance(self):pass
        def event(self,*args,**kwargs):pass
        def act(self,uci,*args,**kwargs):
            assert uci=='e2e4';self.done=True
            atomic_json(self.output/'summary.json',{'completed':True,'aborted':False,'result':0,
                        'plies':1,'reason':'fixture','random_moves':0})
        def close(self):pass
    def completion(url,protocol,prompt,**kwargs):
        calls.append(prompt)
        return {'choices':[{'message':{'content':r'\boxed{e4}'},'finish_reason':'stop'}]}
    monkeypatch.setattr(module,'ChessEpisode',Episode)
    monkeypatch.setattr(module,'request_completion',completion)
    module.play_game(0,tmp_path/'output','fixture',{'candidate_id':'chosen',
        'candidate_source':{'root':str(candidate)},'research_scope':phase2.SCOPE})
    assert calls[0][-1]['content']=='Frozen Phase 2 skill'


def test_phase2_final_cli_loads_complete_bundle(tmp_path,monkeypatch):
    from llmgo.rsi import deepseek_final as module
    root=Path('/user/lyh4170/LLMGo!')
    bundle={'candidate_id':'r000004','model_path':str(root/'assets/models/Qwen3.5-4B'),
        'source':{'root':str(root/'var/campaigns/p2/candidates/r000004/source')},
        'research_scope':phase2.SCOPE,'max_model_len':49152}
    path=tmp_path/'final-candidate.json';atomic_json(path,bundle)
    seen={}
    def run(model,candidate,output,**kwargs):
        seen.update(model=model,candidate=candidate,output=output,**kwargs)
        return {'status':'completed'}
    monkeypatch.setattr(module,'run',run)
    monkeypatch.setenv('DEEPSEEK_MODEL','fixture')
    monkeypatch.setenv('DEEPSEEK_BASE_URL','https://fixture.invalid')
    assert module.main(['--final-candidate',str(path),'--output',str(root/'var/runs/p2-final/final-test'),
                        '--games','20','--parallel-games','20'])==0
    assert seen['model']==Path(bundle['model_path']) and seen['candidate']==Path(bundle['source']['root'])
    assert seen['research_scope']==phase2.SCOPE and seen['candidate_id']=='r000004'
    assert seen['max_model_len']==49152 and seen['games']==20 and seen['parallel_games']==20


def test_agent_can_finish_after_results_without_another_evaluation(tmp_path,monkeypatch):
    site,_=setup_site(tmp_path,monkeypatch)
    c=Controller.create(site,'finish-baseline',scope=phase2.SCOPE)
    baseline=[{'level':level,'track':'player','statistics':{'games':40,'wins':0,'draws':40,'losses':0,'score':.5}}
              for level in (2,3)]
    from llmgo.rsi.ladder import panel_summary
    c.state.update(phase='PLANNING',baseline=baseline,baseline_panel=panel_summary(baseline),
                   best={'candidate':c.state['initial'].copy(),'score':.5,'panel':panel_summary(baseline)})
    c.save();ws=c.workspace()
    atomic_json(ws/'proposal.json',{'hypothesis':'Baseline remains the preferred delivery',
        'jobs':[],'evaluation':False,'decision':'finish','final_candidate':'best'})
    c.prepare(ws/'proposal.json')
    c.step();c.step()
    assert c.step()['status']=='completed' and not c.state['jobs']
    final=read_json(c.root/'final-candidate.json')
    assert final['candidate_id']=='initial' and final['evaluations']==baseline
    assert not (Path(final['source']['root'])/'harness.py').exists()


def test_phase2_does_not_recreate_a_harness_removed_by_agent(tmp_path,monkeypatch):
    site,_=setup_site(tmp_path,monkeypatch)
    c=Controller.create(site,'no-harness',scope=phase2.SCOPE)
    c.state['phase']='PLANNING';c.save();ws=c.workspace()
    (ws/'harness.py').unlink()
    atomic_json(ws/'proposal.json',{'hypothesis':'Use original prompt','jobs':[],'evaluation':False})
    c.prepare(ws/'proposal.json');c.step();c.step()
    assert not (c.workspace()/'harness.py').exists()
