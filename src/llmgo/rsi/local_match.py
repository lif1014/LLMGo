"""Two local non-thinking models share one H800 and the existing chess referee."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

import chess

from llmgo.training.chess_r1 import ANSWER_PARSER,extract_answer,messages
from .artifacts import verify_source
from .checkpoint_benchmark import replay_index
from .contracts import MAX_GAME_PLIES,atomic_json,file_hash,model_identity,read_json,utc,worker_environment
from .deepseek_final import PROTOCOL
from .evaluator import request_completion
from .game import ChessEpisode
from .random_moves import write_report
from .service import VllmServer,_pick_ports,_wait_servers,visible_gpu_ids


def match_protocol(plan):
    return {'role':'independent_local_match','games':40,'white':20,'black':20,
        'opponent':{'kind':'local_model','model':'Qwen3-8B-Chess','repo_id':'ljcnju/Qwen3-8B-Chess',
            'revision':plan['opponent_revision'],'thinking':'disabled'},
        'student_protocol':dict(PROTOCOL),
        'opponent_protocol':{**PROTOCOL,'model_name':'Qwen3-8B-Chess'},
        'student_model_identity':plan['student_identity'],'opponent_model_identity':plan['opponent_identity'],
        'student_model_path':plan['student_model'],'opponent_model_path':plan['opponent_model'],
        'student_cumulative_steps':plan.get('student_cumulative_steps'),
        'answer_parser':ANSWER_PARSER,'opponent_answer_parser':ANSWER_PARSER,
        'parallel_games':40,'student_vllm_max_num_seqs':40,'opponent_vllm_max_num_seqs':40,
        'gpu_count':1,'student_gpu_memory_utilization':.35,'opponent_gpu_memory_utilization':.55,
        'student_request_timeout_seconds':None,'opponent_request_timeout_seconds':None,
        'service_ready_timeout_seconds':None,'max_plies':MAX_GAME_PLIES,'fivefold_repetition':True,
        'invalid_move_policy':'random_legal','prompt':'chess-r1-nonthinking','started':utc()}


class LocalOpponent:
    def __init__(self,url,index):
        self.url=url
        self.rng=random.Random(417000+index//2)
        self.protocol={**PROTOCOL,'model_name':'Qwen3-8B-Chess'}
        self.record=lambda *args,**kwargs:None

    def _play_board(self,board):
        prompt=messages(board.fen(),enable_thinking=False)
        request_id=f'local-opponent-{time.time_ns()}'
        self.record('opponent_request_started',request_id=request_id,messages=prompt)
        response=request_completion(self.url,self.protocol,prompt,timeout=None)
        choice=response['choices'][0];message=choice['message']
        content=message.get('content') or '';reasoning=message.get('reasoning_content') or ''
        parsed=extract_answer(content,board.fen(),reasoning=reasoning,finish_reason=choice.get('finish_reason'))
        uci=parsed.uci if parsed.legal else self.rng.choice(sorted(m.uci() for m in board.legal_moves))
        return uci,{'model':'Qwen3-8B-Chess','response_model':response.get('model'),
            'thinking':'disabled','request_id':request_id,'messages':prompt,
            'raw_output':content,'reasoning':reasoning,'finish_reason':choice.get('finish_reason'),
            'response_tokens':int((response.get('usage') or {}).get('completion_tokens') or 0),
            'parsed_san':parsed.san,'parsed_uci':parsed.uci,
            'fallback_reason':None if parsed.legal else parsed.error or 'invalid_final_answer'}

    def close(self):pass


def play_game(index,output,student_url,opponent_url):
    side='white' if index%2==0 else 'black'
    game_id=f'qwen-chess-{index:02d}'
    opponent={'level':'qwen3-chess','level_id':'Qwen3-8B-Chess','skill':None,
        'depth':None,'movetime_sec':None,'nodes':None,'threads':1,'hash_mb':0,'thinking':'disabled'}
    episode=ChessEpisode(game_id,output.parent.name,side,2,None,output/'games'/game_id,
        seed=417000+index//2,opponent_spec=opponent,
        engine_factory=lambda *args,**kwargs:LocalOpponent(opponent_url,index))
    episode.engine.record=episode.event
    try:
        episode.advance()
        while not episode.done:
            fen=episode.board.fen();prompt=messages(fen,enable_thinking=False)
            request_id=f'local-student-{time.time_ns()}'
            episode.event('model_request_started',request_id=request_id,messages=prompt)
            response=request_completion(student_url,PROTOCOL,prompt,timeout=None)
            choice=response['choices'][0];message=choice['message']
            content=message.get('content') or '';reasoning=message.get('reasoning_content') or ''
            parsed=extract_answer(content,fen,reasoning=reasoning,finish_reason=choice.get('finish_reason'))
            receipt={'request_id':request_id,'response':content,'raw_output':content,'reasoning':reasoning,
                'finish_reason':choice.get('finish_reason'),
                'response_tokens':int((response.get('usage') or {}).get('completion_tokens') or 0),
                'failure_reason':parsed.error,'format_valid':parsed.format_valid,'legal':parsed.legal,
                'uci':parsed.uci if parsed.legal else None}
            episode.event('model_response',**receipt)
            episode.act(receipt['uci'],[receipt],fallback_reason=parsed.error)
        summary=read_json(episode.output/'summary.json')
        return {'index':index,'game_id':game_id,'student_color':side,'result':summary['result'],
            'plies':summary['plies'],'reason':summary['reason'],'events':str(episode.output/'events.jsonl'),
            'events_sha256':file_hash(episode.output/'events.jsonl')}
    except Exception as error:
        if not episode.done:episode.abort(str(error))
        raise
    finally:episode.close()


def collect_games(output,student_url,opponent_url,*,player=play_game):
    """One failed future must not prevent collecting the other finished games."""
    rows=[];failures=[]
    def progress():
        atomic_json(output/'progress.json',{'stage':'playing','completed':len(rows),'total':40,
            'aborted':len(failures),'parallel_games':40,'updated':utc(),
            'wins':sum(r['result']==1 for r in rows),'draws':sum(r['result']==0 for r in rows),
            'losses':sum(r['result']==-1 for r in rows)})
    progress()
    with ThreadPoolExecutor(max_workers=40) as pool:
        pending={pool.submit(player,i,output,student_url,opponent_url):i for i in range(40)}
        for future in as_completed(pending):
            index=pending[future]
            try:
                row=future.result()
                atomic_json(output/'game_results'/f'{index:02d}.json',row);rows.append(row)
            except Exception as error:
                failures.append({'index':index,'error_type':type(error).__name__,'error':str(error)})
                atomic_json(output/'failures.json',failures)
            progress()
    rows.sort(key=lambda row:row['index'])
    return rows,failures


def run(plan):
    output=Path(plan['output'])/'final-test';output.mkdir(parents=True,exist_ok=True)
    if (output/'games').exists():raise FileExistsError('Use a new run for each independent match')
    for name in ('student','opponent'):
        if model_identity(plan[name+'_model'])!=plan[name+'_identity']:
            raise ValueError(name+' model identity changed')
    gpus=visible_gpu_ids()
    if len(gpus)!=1:raise RuntimeError('Local match requires exactly one allocated GPU')
    protocol=match_protocol(plan);atomic_json(output/'protocol.json',protocol)
    ports=_pick_ports(8000,2)
    services=[VllmServer(Path(plan[name+'_model']),protocol[name+'_protocol'],output/(name+'-vllm.log'),
        ports[i],gpus[0],python=plan['inference_python'],max_num_seqs=40,
        async_scheduling=False,compilation_config={'cudagraph_mode':'PIECEWISE'},
        gpu_memory_utilization=protocol[name+'_gpu_memory_utilization'])
        for i,name in enumerate(('student','opponent'))]
    stop=threading.Event()
    def index_loop():
        while not stop.wait(5):replay_index(output.parent)
    indexer=threading.Thread(target=index_loop,daemon=True);indexer.start()
    try:
        # Serialize profiling so the first server's allocations are stable when
        # the second one reserves its own explicit fraction of this same GPU.
        for name,service in zip(('student','opponent'),services,strict=True):
            atomic_json(output/'progress.json',{'stage':'loading_'+name,'completed':0,'total':40,'updated':utc()})
            service.start();error=_wait_servers([service],None)
            if error:raise RuntimeError(error)
        rows,failures=collect_games(output,services[0].base_url,services[1].base_url)
        wins=sum(r['result']==1 for r in rows);draws=sum(r['result']==0 for r in rows);losses=sum(r['result']==-1 for r in rows)
        result={'status':'completed' if len(rows)==40 and not failures else 'incomplete',
            'role':'independent_local_match','opponent':protocol['opponent'],'games':len(rows),
            'planned_games':40,'white':20,'black':20,'wins':wins,'draws':draws,'losses':losses,
            'win_rate':wins/len(rows) if rows else None,'score_rate':(wins+.5*draws)/len(rows) if rows else None,
            'aborted':len(failures),'rows':rows,'completed':utc()}
        atomic_json(output/'result.json',result)
        if failures:raise RuntimeError(f'{len(failures)} games aborted; {len(rows)}/40 collected')
        atomic_json(output/'progress.json',{'stage':'completed','completed':40,'total':40,'updated':utc()})
        return result
    except BaseException as error:
        atomic_json(output/'failure.json',{'type':type(error).__name__,'error':str(error),'updated':utc()})
        raise
    finally:
        stop.set();indexer.join()
        for service in reversed(services):service.stop()
        replay_index(output.parent);write_report(output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args(argv);plan=read_json(args.plan);run_root=Path(plan['output'])
    sys.stdout=sys.stderr=(run_root/'worker.log').open('a',buffering=1)
    try:
        verify_source(plan['source'])
        os.environ.update(worker_environment(plan['site']))
        if not plan.get('opponent_identity'):
            receipt=Path(plan['download_receipt'])
            while not receipt.is_file():
                waiting={'stage':'waiting_model_download','completed':0,'total':40,'updated':utc()}
                atomic_json(run_root/'progress.json',waiting)
                atomic_json(run_root/'final-test/progress.json',waiting)
                status=Path(plan['download_status'])
                if status.is_file() and read_json(status).get('stage')=='failed':
                    raise RuntimeError('Opponent download failed; inspect '+str(status))
                time.sleep(30)
            downloaded=read_json(receipt)
            if downloaded['revision']!=plan['opponent_revision']:
                raise ValueError('Downloaded opponent revision differs from the match plan')
            shards=[v for k,v in downloaded['files'].items() if k.endswith('.safetensors')]
            if len(shards)!=4 or not all(v['lfs_sha256_verified'] for v in shards):
                raise ValueError('Opponent shards have not passed their upstream LFS hashes')
            plan={**plan,'opponent_identity':downloaded['model_identity']}
            atomic_json(run_root/'resolved-plan.json',plan)
        atomic_json(run_root/'progress.json',{'stage':'starting','updated':utc()})
        run(plan)
        atomic_json(run_root/'progress.json',{'stage':'completed','updated':utc()})
    except BaseException as error:
        atomic_json(run_root/'failure.json',{'type':type(error).__name__,'error':str(error),'updated':utc()})
        traceback.print_exc();raise


if __name__=='__main__':main()
