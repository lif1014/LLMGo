"""GRPO run boundaries, reward semantics and full-game replay."""
import importlib.util
import json
import os
from pathlib import Path

import chess
from fastapi.testclient import TestClient

from llmgo.rsi.contracts import atomic_json
from llmgo.rsi.evaluator import candidate_messages
from llmgo.server.app import app
from llmgo.training.chess_r1 import messages, extract_answer
from llmgo.training.chess_r1_oracle import score_answer
from llmgo.training.chess_r1_resume import resume_plan


def test_nonthinking_prompt_and_reward_are_canonical():
    fen = chess.STARTING_FEN
    prompt = messages(fen, enable_thinking=False)
    assert len(prompt) == 2 and '<think>' not in prompt[0]['content']
    assert 'e4' in prompt[1]['content']
    answer = extract_answer(r'\boxed{e4}', fen)
    assert answer.legal and answer.uci == 'e2e4'
    ground = {'board_fen':fen, 'id':'fixture', 'next_move_san':'e4'}
    row = score_answer(r'\boxed{e4}', ground, {'e2e4':.7}, lambda _:True)
    assert abs(row['score']-.9)<1e-9 and row['solution_correct'] == 1
    bad = score_answer(r'\boxed{e4} extra', ground, {}, lambda _:True)
    assert bad['score'] == 0 and bad['format'] == 0
    assert extract_answer(r'\boxed{e4}', fen, finish_reason='length').error == 'truncated'


def test_frozen_recipe_and_worker_result_have_separate_paths(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('grpo_launcher',Path(__file__).resolve().parents[1]/'scripts/run_grpo.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT = tmp_path
    run = tmp_path/'var/runs/example'
    model = tmp_path/'initial';model.mkdir()
    template = tmp_path/'template.json';template.write_text(json.dumps({'framework':'verl','algorithm':'grpo','steps':10,'model_path':'old'}))
    calls=[]
    initial=[]
    def fake(argv, **kwargs):
        calls.append(argv)
        if any(str(part).endswith('chess_r1_gpu_entry.py') for part in argv):
            initial.append(resume_plan(run/'grpo',json.loads((run/'grpo/recipe.json').read_text()),module.SOURCE)['step'])
        if any(str(part).endswith('export_grpo.py') for part in argv):
            atomic_json(run/'grpo/submission.json',{'model_path':str(run/'grpo/model'),'status':'ready'})
    monkeypatch.setattr(module.subprocess,'run',fake)
    monkeypatch.setattr(module.sys,'argv',['run_grpo.py','--output',str(run),'--model',str(model),'--template',str(template)])
    module.main()
    assert (run/'grpo/recipe.json').is_file()
    assert json.loads((run/'submission.json').read_text())['model_path']==str(run/'grpo/model')
    assert not (run/'result.json').exists()
    assert calls[0][-1]==str(run/'grpo')
    assert initial==[0]


def test_candidate_harness_and_skills_are_optional(tmp_path):
    source=tmp_path/'candidate';source.mkdir()
    default=messages(chess.STARTING_FEN,enable_thinking=False)
    payload={'candidate_source':{'root':str(source)},'skills_path':None}
    assert candidate_messages(payload,chess.STARTING_FEN,chess.Board(),default)==default
    (source/'harness.py').write_text("def build_messages(fen, legal_san, skills, default_messages):\n return [{'role':'system','content':skills[0]['text']}, default_messages[1]]\n")
    skills=tmp_path/'skills.json';skills.write_text('[{"text":"Find a tactic"}]')
    payload['skills_path']=str(skills)
    adapted=candidate_messages(payload,chess.STARTING_FEN,chess.Board(),default)
    assert adapted[0]['content']=='Find a tactic' and 'e4' in adapted[1]['content']


def test_run_api_reads_remote_layout_and_every_game_step(tmp_path,monkeypatch):
    project=tmp_path/'LLMGo!';runs=project/'var/runs';run=runs/'demo-j000001';run.mkdir(parents=True)
    campaigns=project/'var/campaigns';campaign=campaigns/'demo'
    atomic_json(campaign/'campaign.json',{'campaign_id':'demo','phase':'WAITING','jobs':{},'attempts':[]})
    atomic_json(campaign/'jobs/j000001/manifest.json',{'payload':{'output':str(run),'protocol':{'games':10}}})
    atomic_json(run/'progress.json',{'stage':'evaluating'})
    atomic_json(run/'grpo/progress.json',{'stage':'update_completed','step':1})
    (run/'research.log').write_text('line one\nline two\n')
    game=run/'games/g0';game.mkdir(parents=True)
    board=chess.Board();board.push_uci('e2e4')
    rows=[{'type':'episode_started','student_color':'white','fen':chess.STARTING_FEN},
          {'type':'model_request_started','messages':messages(chess.STARTING_FEN,enable_thinking=False),'request_id':'one'},
          {'type':'model_response','response':r'\boxed{e4}','response_tokens':8},
          {'type':'student_move','uci':'e2e4','fen':board.fen()},
          {'type':'episode_finished','result':1,'reason':'checkmate'}]
    (game/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    atomic_json(game/'summary.json',{'completed':True,'student_color':'white','result':1,'reason':'checkmate',
                                   'decisions':1,'requests':1,'response_tokens':8,'random_moves':0})
    monkeypatch.setenv('LLMGO_RUN_ROOT',str(runs));monkeypatch.setenv('LLMGO_RSI_ROOT',str(campaigns))
    client=TestClient(app)
    assert client.get('/api/grpo/runs').json()[0]['id']=='demo-j000001'
    files=client.get('/api/rsi/campaigns/demo/job-files',params={'job_id':'j000001'}).json()
    assert files['formal_games']==10 and any(x['path'].endswith('research.log') for x in files['files'])
    listing=client.get('/api/rsi/campaigns/demo/games').json()
    assert listing[0]['summary']['result']=='white_win'
    replay=client.get('/api/rsi/campaigns/demo/game-events',params={'folder':listing[0]['folder']}).json()
    assert [x['event']['type'] for x in replay['events']].count('position')==2
    assert replay['events'][-1]['event']['summary']['result']=='white_win'
    artifact=client.get('/api/rsi/campaigns/demo/artifact',params={'path':'jobs/j000001/games/g0/events.jsonl','length':64})
    assert artifact.status_code==200
