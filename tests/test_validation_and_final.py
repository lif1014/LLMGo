import chess
import json
from pathlib import Path
import threading

import pytest

from llmgo.rsi.deepseek_client import extract_loose_answer
from llmgo.rsi.deepseek_final import DeepSeekPlayer, OPPONENT
from llmgo.rsi.ladder import evaluation_plan, opponent, panel_summary


def test_validation_is_l2_to_l3_and_requires_both_anchors():
    assert evaluation_plan()['levels'] == [2, 3]
    assert evaluation_plan({'levels': [2.5]})['levels'] == [2.5]
    assert opponent(2.5)['skill'] == -3
    for level in (1, 4):
        try:
            opponent(level)
        except ValueError:
            pass
        else:
            raise AssertionError('Non-validation level admitted')
    rows = [{'level': level, 'statistics': {'score': score, 'games': 20}}
            for level, score in ((2, .4), (2.5, .9), (3, .6))]
    result = panel_summary(rows)['player']
    assert result['complete'] and result['score'] == .5
    assert panel_summary(rows[:1])['player']['score'] is None


def test_deepseek_opponent_uses_loose_box_and_records_fallback(monkeypatch):
    board = chess.Board()
    assert OPPONENT['thinking'] == 'disabled'
    assert extract_loose_answer('I choose \\boxed{e4}.', board.fen()).uci == 'e2e4'
    monkeypatch.setattr('llmgo.rsi.deepseek_final.deepseek_request',
        lambda messages, request_id, **kwargs: {'choices': [{'message': {'content': '\\boxed{e4}'},
            'finish_reason': 'stop'}], 'usage': {'completion_tokens': 12}})
    player = DeepSeekPlayer('white', 'example', 7)
    move, detail = player._play_board(board)
    assert move == 'e2e4' and detail['fallback_reason'] is None
    assert detail['raw_output'] == '\\boxed{e4}'
    monkeypatch.setattr('llmgo.rsi.deepseek_final.deepseek_request',
        lambda messages, request_id, **kwargs: {'choices': [{'message': {'content': 'invalid'},
            'finish_reason': 'stop'}]})
    move, detail = player._play_board(board)
    assert chess.Move.from_uci(move) in board.legal_moves
    assert detail['fallback_reason'] == 'missing_box'


@pytest.fixture
def final_fixture(tmp_path, monkeypatch):
    from llmgo.rsi import deepseek_final as module
    model=tmp_path/'model';model.mkdir()
    (model/'model.safetensors').write_bytes(b'fixture weights')
    (model/'config.json').write_text('{}')
    candidate=tmp_path/'candidate';candidate.mkdir()
    (candidate/'skills.json').write_text('[]')
    servers=[]
    class Server:
        base_url='http://fixture/v1'
        def __init__(self,*args,**kwargs):
            self.stopped=False;self.options=kwargs;servers.append(self)
        def start(self):pass
        def stop(self):self.stopped=True
    monkeypatch.setattr(module,'VllmServer',Server)
    monkeypatch.setattr(module,'visible_gpu_ids',lambda:['0'])
    monkeypatch.setattr(module,'_pick_ports',lambda *args:[8000])
    monkeypatch.setattr(module,'_wait_servers',lambda *args:None)
    return module,model,candidate,tmp_path/'run/final-test',servers


@pytest.mark.parametrize('games,parallel',[(10,1),(20,4)])
def test_final_test_balances_twenty_games_and_uses_actual_denominator(final_fixture,monkeypatch,games,parallel):
    module,model,candidate,output,servers=final_fixture
    lock=threading.Lock();seen=[]
    def unexpected_probe(*args,**kwargs):
        pytest.fail('No unscored inference request before formal games')
    monkeypatch.setattr(module,'request_completion',unexpected_probe)
    def play(index,folder,url,payload):
        assert payload['research_scope']=='phase1_rl_only'
        with lock:seen.append(index)
        return {'index':index,'student_color':'white' if index%2==0 else 'black',
                'pair_id':index//2,'seed':417000+index//2,'result':[1,0,-1][index%3]}
    monkeypatch.setattr(module,'play_game',play)
    result=module.run(model,candidate,output,games=games,parallel_games=parallel)
    protocol=json.loads((output/'protocol.json').read_text())
    assert sorted(seen)==list(range(games))
    assert protocol['games']==games and protocol['white']==protocol['black']==games//2
    assert protocol['seed_pairs']==[417000+i for i in range(games//2)]
    assert protocol['student_request_timeout_seconds'] is None
    assert protocol['opponent_request_timeout_seconds'] is None
    assert protocol['parallel_games']==parallel
    assert [row['index'] for row in result['rows']]==list(range(games))
    assert sum(row['student_color']=='white' for row in result['rows'])==games//2
    assert sum(row['student_color']=='black' for row in result['rows'])==games//2
    assert result['score_rate']==(result['wins']+.5*result['draws'])/games
    assert result['win_rate']==result['wins']/games
    assert len(list((output/'game_results').glob('*.json')))==games
    assert json.loads((output/'progress.json').read_text())['completed']==games
    assert servers[0].stopped and servers[0].options['max_num_seqs']>=parallel
    assert servers[0].options['async_scheduling'] is False
    assert servers[0].options['compilation_config']=={'cudagraph_mode':'PIECEWISE'}


def test_final_games_run_concurrently_and_keep_ordered_receipts(final_fixture,monkeypatch):
    module,model,candidate,output,servers=final_fixture
    barrier=threading.Barrier(4)
    def play(index,*args):
        barrier.wait(timeout=5)
        return {'index':index,'result':0}
    monkeypatch.setattr(module,'play_game',play)
    result=module.run(model,candidate,output,games=4,parallel_games=4)
    assert [row['index'] for row in result['rows']]==[0,1,2,3]
    assert result['draws']==4 and result['score_rate']==.5


@pytest.mark.parametrize('games,parallel',[(1,1),(3,1),(0,1),(True,1),(20,0),(20,41)])
def test_invalid_final_schedule_is_rejected_before_creating_output(final_fixture,games,parallel):
    module,model,candidate,output,servers=final_fixture
    with pytest.raises(ValueError):module.run(model,candidate,output,games=games,parallel_games=parallel)
    assert not output.exists() and not servers


def test_failed_game_does_not_publish_a_complete_score(final_fixture,monkeypatch):
    module,model,candidate,output,servers=final_fixture
    def play(index,*args):
        raise RuntimeError('fixture API unavailable')
    monkeypatch.setattr(module,'play_game',play)
    with pytest.raises(RuntimeError,match='API unavailable'):
        module.run(model,candidate,output,games=20,parallel_games=2)
    assert not (output/'result.json').exists()
    assert json.loads((output/'progress.json').read_text())['stage']=='failed'
    assert servers[0].stopped


def test_final_replica_explicitly_disables_async_scheduler(tmp_path,monkeypatch):
    from llmgo.rsi.service import VllmServer
    from llmgo.rsi.deepseek_final import PROTOCOL
    calls=[]
    monkeypatch.setattr('llmgo.rsi.service.subprocess.Popen',lambda argv,**kw:calls.append(argv))
    server=VllmServer(tmp_path/'model',PROTOCOL,tmp_path/'vllm.log',8000,'0',
                      python='fixture-python',max_num_seqs=20,async_scheduling=False,
                      compilation_config={'cudagraph_mode':'PIECEWISE'})
    server.start()
    assert '--no-async-scheduling' in calls[0]
    assert json.loads(calls[0][calls[0].index('--compilation-config')+1])=={'cudagraph_mode':'PIECEWISE'}
    server.stop()


def test_formal_student_request_has_no_socket_deadline(tmp_path,monkeypatch):
    import io
    from llmgo.rsi import deepseek_final as module
    calls=[]
    class Opener:
        def open(self,req,*,timeout):
            calls.append(timeout)
            return io.BytesIO(json.dumps({'choices':[{'message':{'content':r'\boxed{e4}'},'finish_reason':'stop'}]}).encode())
    monkeypatch.setattr('urllib.request.build_opener',lambda *args:Opener())
    class Episode:
        def __init__(self,*args,**kwargs):
            self.board=chess.Board();self.done=False;self.output=args[5];self.output.mkdir(parents=True)
            (self.output/'events.jsonl').write_text('')
        def advance(self):pass
        def event(self,*args,**kwargs):pass
        def act(self,uci,*args,**kwargs):
            assert uci=='e2e4';self.done=True
            (self.output/'summary.json').write_text(json.dumps({'completed':True,'aborted':False,'result':0,'plies':1,'reason':'fixture','random_moves':0}))
        def close(self):pass
    monkeypatch.setattr(module,'ChessEpisode',Episode)
    monkeypatch.setattr(module,'candidate_messages',lambda payload,fen,board,default:default)
    module.play_game(0,tmp_path,'http://fixture/v1',{'candidate_id':'fixture'})
    assert calls==[None]


def test_formal_opponent_request_has_no_deadline(monkeypatch):
    from llmgo.rsi import deepseek_final as module
    calls=[]
    def request(messages,request_id,*,timeout):
        calls.append(timeout)
        return {'choices':[{'message':{'content':r'\boxed{e4}'},'finish_reason':'stop'}]}
    monkeypatch.setattr(module,'deepseek_request',request)
    move,_=module.DeepSeekPlayer('white','fixture',0)._play_board(chess.Board())
    assert move=='e2e4' and calls==[None]
