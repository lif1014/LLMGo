import http.client
import importlib.util
import json
from pathlib import Path
import threading
import tempfile
import uuid

import pytest

import chess

from llmgo.rsi import local_match as module
from llmgo.rsi.contracts import atomic_json,model_identity,read_json


def test_long_match_id_uses_a_real_short_unix_socket_path():
    import socket
    script=Path(__file__).resolve().parents[1]/'scripts/submit_local_chess_match.py'
    spec=importlib.util.spec_from_file_location('submit_local_match',script)
    submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
    run_id='chess-j10-s5-vs-qwen3-8b-chess-20261106-retry1'
    with tempfile.TemporaryDirectory(prefix='lc-',dir='/tmp') as directory:
        root=Path(directory)
        old=root/'var/runtime'/run_id/'tmp'/str(uuid.uuid4())
        old.parent.mkdir(parents=True)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
            with pytest.raises(OSError):listener.bind(str(old))
        runtime=submit.match_runtime(root,run_id)
        receipt=submit.check_ipc_path(runtime)
        assert receipt['passed'] and receipt['path_bytes']<=107
        assert list((runtime/'tmp').iterdir())==[]
        assert runtime!=submit.match_runtime(root,run_id+'-other')


def test_submitter_uses_the_selected_fifteen_step_config(tmp_path,monkeypatch,capsys):
    script=Path(__file__).resolve().parents[1]/'scripts/submit_local_chess_match.py'
    spec=importlib.util.spec_from_file_location('submit_selected_match',script)
    submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
    student=tmp_path/'models/fifteen';student.mkdir(parents=True)
    atomic_json(student/'config.json',{});(student/'model.safetensors').write_bytes(b'fifteen')
    config={'run_id':'meta15-test','student_model':'models/fifteen',
        'student_sha256':model_identity(student)['sha256'],'student_cumulative_steps':15,
        'opponent_model':'models/opponent','opponent_revision':'pinned'}
    atomic_json(tmp_path/'selected.json',config)
    atomic_json(tmp_path/'configs/rsi/site.beijing.json',{'inference_python':'fixture','image':'fixture'})
    monkeypatch.setattr(submit.sys,'argv',[str(script),'--repo',str(tmp_path),'--config','selected.json'])
    monkeypatch.setattr(submit,'check_ipc_path',lambda path:{'passed':True})
    monkeypatch.setattr(submit,'source_snapshot',lambda root,target:{'root':str(target),'sha256':'fixture'})
    calls=[]
    def platform(site,argv):
        calls.append(argv)
        if argv[1]=='list':return {'tasks':[]}
        if '--dry-run' in argv:
            return {'request':{'task':{'resource_pool_id':523,'priority':3,'resources':{'gpu_count':1}}}}
        return {'id':123,'status':'Queued'}
    monkeypatch.setattr(submit,'cctl',platform)
    submit.main()
    plan=read_json(tmp_path/'var/runs/meta15-test/plan.json')
    assert plan['student_model']==str(student) and plan['student_cumulative_steps']==15
    assert plan['student_identity']==model_identity(student)
    protocol=read_json(tmp_path/'var/runs/meta15-test/final-test/protocol.json')
    assert protocol['student_cumulative_steps']==15 and protocol['parallel_games']==40
    assert sum(argv[1]=='create' and '--dry-run' not in argv for argv in calls)==1
    assert json.loads(capsys.readouterr().out)['task_id']==123


def test_two_models_share_one_gpu_and_play_all_forty_games(tmp_path,monkeypatch):
    models=[]
    for name in ('student','opponent'):
        path=tmp_path/name;path.mkdir();atomic_json(path/'config.json',{})
        (path/'model.safetensors').write_bytes(name.encode());models.append(path)
    plan={'output':str(tmp_path/'run'),'inference_python':'fixture','opponent_revision':'pinned',
          'student_cumulative_steps':15,
          **{name+'_model':str(path) for name,path in zip(('student','opponent'),models)},
          **{name+'_identity':model_identity(path) for name,path in zip(('student','opponent'),models)}}
    starts=[];ready=[];servers=[]
    class Server:
        def __init__(self,model,protocol,log,port,gpu,**kwargs):
            self.model=model;self.base_url=protocol['model_name'];self.options=kwargs
            self.started=False;self.stopped=False;servers.append(self)
            assert gpu=='0' and kwargs['max_num_seqs']==40
        def start(self):
            if starts:assert ready==['student']
            self.started=True;starts.append(self.model.name)
        def stop(self):self.stopped=True
    monkeypatch.setattr(module,'VllmServer',Server)
    monkeypatch.setattr(module,'visible_gpu_ids',lambda:['0'])
    monkeypatch.setattr(module,'_pick_ports',lambda *args:[8000,8001])
    def wait(services,timeout):
        assert timeout is None and services[0].started
        ready.append(services[0].model.name)
    monkeypatch.setattr(module,'_wait_servers',wait)
    barrier=threading.Barrier(40);requests=[]
    def request(url,protocol,prompt,*,timeout):
        assert ready==['student','opponent'] and timeout is None
        assert protocol['max_tokens']==16384 and protocol['enable_thinking'] is False
        fen=prompt[1]['content'].split('Current FEN string: ')[1].split('\n')[0]
        board=chess.Board(fen)
        if board.fullmove_number==1 and board.turn==chess.WHITE:barrier.wait(timeout=10)
        move={ (1,chess.WHITE):'f3',(1,chess.BLACK):'e5',(2,chess.WHITE):'g4',(2,chess.BLACK):'Qh4#'}[(board.fullmove_number,board.turn)]
        requests.append(url)
        return {'model':url,'choices':[{'message':{'content':'\\boxed{'+move+'}','reasoning_content':''},'finish_reason':'stop'}],'usage':{'completion_tokens':8}}
    monkeypatch.setattr(module,'request_completion',request)
    result=module.run(plan)
    assert result['status']=='completed' and result['games']==40
    assert (result['wins'],result['draws'],result['losses'])==(20,0,20)
    assert [s.options['gpu_memory_utilization'] for s in servers]==[.35,.55]
    assert all(s.stopped for s in servers)
    assert len(requests)==160
    output=Path(plan['output'])/'final-test'
    assert read_json(output/'protocol.json')['student_cumulative_steps']==15
    assert len(read_json(output/'replay-index.json')['files'])==80
    assert len(read_json(output/'random-moves.json')['games'])==40
    first=[json.loads(line) for line in (output/'games/qwen-chess-00/events.jsonl').read_text().splitlines()]
    assert first[0]['policy_version']=='run'
    assert any(e['type']=='opponent_request_started' and e['messages'] for e in first)
    assert next(e for e in first if e['type']=='opponent_move')['diagnostics']['raw_output']==r'\boxed{e5}'


def test_one_connection_error_does_not_hide_other_completed_games(tmp_path):
    def player(index,*args):
        if index==0:raise http.client.RemoteDisconnected('fixture disconnect')
        return {'index':index,'result':0}
    rows,failures=module.collect_games(tmp_path,'student','opponent',player=player)
    assert len(rows)==39 and failures[0]['index']==0
    assert read_json(tmp_path/'progress.json')['completed']==39
    assert read_json(tmp_path/'progress.json')['aborted']==1
    assert len(list((tmp_path/'game_results').glob('*.json')))==39


def test_opponent_invalid_answer_records_random_fallback(monkeypatch):
    monkeypatch.setattr(module,'request_completion',lambda *args,**kwargs:{'choices':[{'message':{'content':'invalid'},'finish_reason':'stop'}]})
    board=chess.Board();player=module.LocalOpponent('fixture',0);events=[]
    player.record=lambda kind,**data:events.append((kind,data))
    uci,detail=player._play_board(board)
    assert chess.Move.from_uci(uci) in board.legal_moves
    assert detail['fallback_reason']=='missing_box' and detail['raw_output']=='invalid'
    assert events[0][0]=='opponent_request_started'


def test_download_range_resumes_a_short_successful_http_body(tmp_path):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import urllib.request
    script=Path(__file__).resolve().parents[1]/'scripts/download_chess_opponent.py'
    spec=importlib.util.spec_from_file_location('download_chess_opponent',script)
    download=importlib.util.module_from_spec(spec);spec.loader.exec_module(download)
    body=b'0123456789abcdef';ranges=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            lo,hi=map(int,self.headers['Range'].removeprefix('bytes=').split('-'));ranges.append((lo,hi))
            payload=body[lo:hi+1]
            if len(ranges)==1:payload=payload[:5]
            self.send_response(206);self.send_header('Content-Range',f'bytes {lo}-{hi}/{len(body)}')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        target=tmp_path/'chunk.part'
        download.download_range(f'http://127.0.0.1:{server.server_port}/model',target,0,15,
            opener=lambda:urllib.request.build_opener(urllib.request.ProxyHandler({})),attempts=2)
        assert target.read_bytes()==body and ranges==[(0,15),(5,15)]
    finally:server.shutdown();server.server_close();thread.join()
