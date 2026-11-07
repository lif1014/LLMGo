#!/usr/bin/env python3
"""Submit the user's one-GPU local-opponent match, preserving every receipt."""
import argparse
import hashlib
from pathlib import Path
import shlex
import socket
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.artifacts import source_snapshot
from llmgo.rsi.checkpoint_benchmark import cctl
from llmgo.rsi.contracts import atomic_json,identifier,model_identity,read_json,utc
from llmgo.rsi.local_match import match_protocol


def match_runtime(root,run_id):
    # vLLM places UUID-named AF_UNIX sockets below TMPDIR. Full experiment
    # names exceeded sockaddr_un.sun_path even though normal files worked.
    return root/'var/runtime'/('lc-'+hashlib.sha256(run_id.encode()).hexdigest()[:10])


def check_ipc_path(runtime):
    folder=runtime/'tmp';folder.mkdir(parents=True,exist_ok=True)
    path=folder/str(uuid.uuid4())
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
        try:
            listener.bind(str(path))
            return {'passed':True,'path_bytes':len(str(path).encode()),'runtime_root':str(runtime)}
        finally:
            if path.exists():path.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/user/lyh4170/LLMGo!'))
    parser.add_argument('--run-id',help='New run ID for a separately receipted retry')
    parser.add_argument('--config',type=Path,default=Path('configs/evaluation/qwen3-chess-20261106.json'))
    args=parser.parse_args();root=args.repo
    config=read_json(root/args.config)
    run_id=identifier(args.run_id or config['run_id'])
    run=root/'var/runs'/run_id;run.mkdir(parents=True,exist_ok=False)
    site=read_json(root/'configs/rsi/site.beijing.json')
    site['runtime_root']=str(match_runtime(root,run_id))
    atomic_json(run/'ipc-preflight.json',check_ipc_path(Path(site['runtime_root'])))
    student=root/config['student_model'];identity=model_identity(student)
    if identity['sha256']!=config['student_sha256']:raise ValueError('Wrong configured student checkpoint')
    where=root/'var/benchmarks'/run_id;source=source_snapshot(root,where/'source')
    plan={'output':str(run),'site':site,'source':source,'student_model':str(student),
        'student_identity':identity,'student_cumulative_steps':config['student_cumulative_steps'],
        'opponent_model':str(root/config['opponent_model']),'opponent_identity':None,
        'opponent_revision':config['opponent_revision'],'inference_python':site['inference_python'],
        'download_receipt':str(root/'assets/models/Qwen3-8B-Chess.download.json'),
        'download_status':str(root/'var/runs/qwen3-chess-download-20261106/progress.json')}
    atomic_json(run/'plan.json',plan)
    atomic_json(run/'final-test/protocol.json',match_protocol(plan))
    atomic_json(run/'progress.json',{'stage':'prepared','updated':utc()})
    atomic_json(run/'final-test/progress.json',{'stage':'prepared','completed':0,'total':40,'updated':utc()})
    task={'kind':'BATCH','project':'rsi','cluster':'beijing','resourcePool':'model-new',
        'priority':'PREEMPTABLE','billingAccountId':'N00069','gpuModel':'H800',
        'image':site['image'],'resources':{'gpuCount':1,'cpu':8,'memory':'96Gi'},
        'description':run_id+'-40games-40parallel-1gpu',
        'entry':shlex.join(['/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python','-B',
            str(Path(source['root'])/'scripts/run_local_chess_match.py'),'--plan',str(run/'plan.json')]),
        'envVars':[{'name':'HF_HUB_OFFLINE','value':'1'},{'name':'TRANSFORMERS_OFFLINE','value':'1'},
            {'name':'MAX_JOBS','value':'4'},{'name':'PYTHONDONTWRITEBYTECODE','value':'1'}]}
    atomic_json(where/'task-request.json',task)
    active=cctl(site,['job','list','--own','--project','rsi','--cluster','beijing',
        '--status','Running,Queued,Starting,Creating,Unknown','--limit','100'])
    atomic_json(where/'active-before.json',active)
    preview=cctl(site,['job','create','-f',str(where/'task-request.json'),'--dry-run'])
    mapped=preview['request']['task']
    if mapped['resource_pool_id']!=523 or mapped['priority']!=3 or mapped['resources']['gpu_count']!=1:
        raise ValueError('Wrong match pool, priority or GPU count')
    atomic_json(where/'task-dryrun.json',preview)
    atomic_json(where/'task-intent.json',{'status':'submitting','description':task['description'],'updated':utc()})
    created=cctl(site,['job','create','-f',str(where/'task-request.json')])
    atomic_json(where/'task-created.json',created)
    receipt={'task_id':created['id'],'platform_status':created['status'],'resource_pool':'model-new',
        'priority':'PREEMPTABLE','gpu_count':1,'submitted':utc(),'run_id':run_id}
    atomic_json(run/'submission.json',receipt);atomic_json(where/'task-intent.json',{'status':'submitted',**receipt})
    print(__import__('json').dumps(receipt),flush=True)


if __name__=='__main__':main()
