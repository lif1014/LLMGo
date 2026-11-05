"""Independent checkpoint arena: six jobs, persistent receipts, disk-only score collection."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import tomllib
import traceback

from .artifacts import source_snapshot, verify_source
from .checkpoint_elo import TABLE_SOURCE, performance
from .contracts import atomic_json, model_identity, read_json, utc, worker_environment

TERMINAL = {'Succeeded','Failed','Killed','Stopped','OOMKilled','Invalid','Deleted'}


def optional(path):
    return read_json(path) if path.is_file() else None


def collect_entry(entry, runs, config):
    run = runs/entry['run_id']
    submission = optional(run/'submission.json') or {}
    progress = optional(run/'final-test/progress.json') or optional(run/'progress.json') or {}
    result = optional(run/'final-test/result.json')
    rows = [read_json(p) for p in sorted((run/'final-test/game_results').glob('*.json'))]
    valid = [r for r in rows if r.get('result') in (-1,0,1)]
    counts = {'wins':sum(r['result']==1 for r in valid), 'draws':sum(r['result']==0 for r in valid),
              'losses':sum(r['result']==-1 for r in valid)}
    row = {**entry, 'task_id':submission.get('task_id'), 'platform_status':submission.get('platform_status'),
           'stage':progress.get('stage') or submission.get('platform_status') or 'prepared',
           'completed_games':len(valid), 'total_games':config['games'], **counts,
           'score_rate':(counts['wins']+.5*counts['draws'])/len(valid) if valid else None,
           'formal':False, 'elo':None, 'protocol_available':(run/'final-test/protocol.json').is_file()}
    if result and result.get('status') == 'completed':
        protocol = read_json(run/'final-test/protocol.json')
        expected = config['games']
        if (result['games'] != expected or len(valid) != expected
                or len({r['index'] for r in valid}) != expected
                or protocol['student_model_identity']['sha256'] != entry['expected_sha256']
                or protocol.get('answer_parser') != 'last-box-san-v1'
                or protocol.get('opponent_answer_parser') != 'last-box-san-v1'
                or protocol['parallel_games'] != config['parallel_games']
                or protocol.get('student_vllm_max_num_seqs') != config['parallel_games']
                or protocol['opponent']['model'] != 'deepseek-v4.1-flash'
                or protocol['opponent']['base_url'] != config['opponent']['base_url']
                or protocol['opponent']['thinking'] != 'disabled'
                or protocol['student_protocol']['enable_thinking'] is not False
                or protocol['max_plies'] != 300 or not protocol['fivefold_repetition']):
            raise ValueError('Benchmark result identity or game population differs: '+entry['id'])
        if any(result[k] != counts[k] for k in counts):
            raise ValueError('Formal result does not match executed game receipts')
        row.update(formal=True, stage='completed', **performance(**counts,teacher=config['teacher_rating']))
        row['elo'] = row['performance_rating']
    failure = optional(run/'failure.json') or optional(run/'final-test/failure.json')
    if not row['formal'] and submission.get('platform_status') in TERMINAL:
        row['stage'] = submission['platform_status']
    if failure and not row['formal']:
        row.update(stage='failed', error=failure.get('error') or failure.get('message'))
    return row


def catalog(batch):
    config = read_json(batch/'benchmark-plan.json')
    runs = Path(config['repo'])/'var/runs'
    rows = []
    for entry in config['entries']:
        try:
            rows.append(collect_entry(entry,runs,config))
        except (ValueError,KeyError) as error:
            receipt = optional(runs/entry['run_id']/'submission.json') or {}
            rows.append({**entry,'stage':'invalid_result','error':str(error),'formal':False,'elo':None,
                         'platform_status':receipt.get('platform_status'),'task_id':receipt.get('task_id')})
    report = {'schema_version':1,'batch_id':config['batch_id'],'updated':utc(),
              'teacher':{'model':'deepseek-v4.1-flash','thinking':'disabled','rating':config['teacher_rating']},
              'rating_method':'fide-table-8.1.1','rating_source':TABLE_SOURCE,
              'rating_note':'实验内表現分；得分率取最近整数百分比（半入）；0/100% 按表约定 ±800。',
              'games_per_checkpoint':config['games'],'parallel_games_per_gpu':config['parallel_games'],
              'entries':rows,'formal_checkpoints':sum(r['formal'] for r in rows),'total_checkpoints':len(rows),
              'protocol':{'answer_parser':'last-box-san-v1','max_plies':300,'fivefold_repetition':True},
              'complete':all(r['formal'] for r in rows)}
    atomic_json(batch/'results.json',report)
    atomic_json(runs/config['catalog_run']/'benchmark.json',report)
    return report


def prepare(repo, config_path):
    config = read_json(config_path)
    batch = repo/'var/benchmarks'/config['batch_id']
    batch.mkdir(parents=True,exist_ok=False)
    config['repo'] = str(repo)
    previous = read_json(repo/'var/runs'/config['reference_run']/'plan.json')
    config['opponent'] = previous['opponent']
    config['opponent']['version'] = 'non-thinking-last-box-san-v2'
    source = source_snapshot(repo,batch/'source')
    config['source'] = source
    candidate = repo/config['candidate']
    if (candidate/'harness.py').exists() or read_json(candidate/'skills.json') != []:
        raise ValueError('Benchmark uses the unchanged Phase1 prompt without harness or skills')
    for index,entry in enumerate(config['entries']):
        model = repo/entry['model']
        identity = model_identity(model)
        if identity['sha256'] != entry['expected_sha256']:
            raise ValueError('Checkpoint identity changed: '+entry['id'])
        entry['run_id'] = config['batch_id']+'-'+entry['id']
        run = repo/'var/runs'/entry['run_id']
        run.mkdir(parents=True,exist_ok=False)
        atomic_json(run/'plan.json',{'source':source,'model_path':str(model),'model_identity':identity,
                    'candidate':str(candidate),'entry':entry,'games':config['games'],
                    'parallel_games':config['parallel_games'],'opponent':config['opponent'],
                    'runtime_root':str(repo/'var/runtime'/f'e6-1106-{index}'),
                    'teacher_rating':config['teacher_rating']})
    atomic_json(batch/'benchmark-plan.json',config)
    catalog(batch)
    return batch


def request(config, entry, python):
    batch = Path(config['repo'])/'var/benchmarks'/config['batch_id']
    return {'kind':'BATCH','project':'rsi','cluster':'beijing',
            'resourcePool':config['resource_pool'],'priority':config['priority'],
            'billingAccountId':'N00069','gpuModel':'H800',
            'image':'share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb',
            'resources':{'gpuCount':1,'cpu':8,'memory':'64Gi'},
            'description':config['batch_id']+'-'+entry['id']+'-40games-40parallel',
            'entry':shlex.join([python,'-B',str(batch/'source/scripts/run_checkpoint_benchmark.py'),
                               'worker','--batch',str(batch),'--entry',entry['id']]),
            'envVars':[{'name':'HF_HUB_OFFLINE','value':'1'},{'name':'TRANSFORMERS_OFFLINE','value':'1'},
                       {'name':'MAX_JOBS','value':'4'},
                       {'name':'DEEPSEEK_PROXY_URL','value':'http://whitelist-proxy.cybertron.svc.cluster.local:7891'}]}


def cctl(site, arguments):
    from llmgo.remote.cctl_env import cctl_environment, cctl_failure_detail
    env = cctl_environment(token_file=site['cctl_token_file'],identity_file=site['cctl_identity_file'],
                           config_root=Path(site['repo'])/'var/runtime/benchmark-cctl-profile')
    for name in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy'):
        env.pop(name,None)
    env = {k:v for k,v in env.items() if not k.startswith('LLMGO_BENCH_PAYLOAD_')}
    result = subprocess.run([site['cctl_bin'],*arguments,'--no-input','-o','json'],
                            env=env,capture_output=True,text=True,timeout=90)
    if result.returncode:
        raise RuntimeError(cctl_failure_detail(result,env))
    return json.loads(result.stdout)


def submit(batch):
    config = read_json(batch/'benchmark-plan.json')
    site = read_json(Path(config['source']['root'])/'configs/rsi/site.beijing.json')
    # User explicitly authorizes all six independent 1-GPU jobs in model-new.
    active = cctl(site,['job','list','--own','--project','rsi','--cluster','beijing',
                        '--status','Creating,Created,Pending,Scheduling,Queued,Requeue,Starting,Running,Unknown','--limit','100'])
    atomic_json(batch/'active-before.json',active)
    for entry in config['entries']:
        folder = batch/'submissions'/entry['id']
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'created.json').exists():
            continue
        if (folder/'intent.json').exists():
            raise RuntimeError('Uncertain submission requires reconciliation: '+entry['id'])
        job = request(config,entry,site['python'])
        atomic_json(folder/'request.json',job)
        preview = cctl(site,['job','create','-f',str(folder/'request.json'),'--dry-run'])
        task = preview['request']['task']
        if not (task['resource_pool_id']==523 and task['priority']==3
                and task['resources']['gpu_count']==1 and task['cluster']=='beijing'
                and task['gpu_model'].lower()=='h800'):
            raise ValueError('Unexpected job resource mapping')
        atomic_json(folder/'dryrun.json',preview)
        atomic_json(folder/'intent.json',{'status':'submitting','description':job['description'],'utc':utc()})
        created = cctl(site,['job','create','-f',str(folder/'request.json')])
        atomic_json(folder/'created.json',created)
        atomic_json(folder/'intent.json',{'status':'submitted','task_id':created['id'],'utc':utc()})
        atomic_json(Path(config['repo'])/'var/runs'/entry['run_id']/'submission.json',
                    {'task_id':created['id'],'platform_status':created['status'],'resource_pool':config['resource_pool'],
                     'priority':config['priority'],'gpu_count':1,'submitted':utc()})
        catalog(batch)


def watch(batch):
    config = read_json(batch/'benchmark-plan.json')
    site = read_json(Path(config['source']['root'])/'configs/rsi/site.beijing.json')
    next_poll = 0
    while True:
        if time.time() >= next_poll:
            for entry in config['entries']:
                run = Path(config['repo'])/'var/runs'/entry['run_id']
                receipt = read_json(run/'submission.json')
                if receipt.get('platform_status') in TERMINAL:
                    continue
                try:
                    task = cctl(site,['job','get',str(receipt['task_id'])])
                    receipt.update(platform_status=task['status'],platform_checked=utc())
                    atomic_json(run/'submission.json',receipt)
                except RuntimeError as error:
                    atomic_json(batch/'last-platform-error.json',{'error':str(error),'utc':utc()})
            next_poll = time.time()+600
        report = catalog(batch)
        if all(r['formal'] or r.get('platform_status') in TERMINAL for r in report['entries']):
            atomic_json(batch/'finished.json',{'complete':report['complete'],'utc':utc()})
            return report
        time.sleep(30)  # Only reads shared-disk results; platform polling remains 600 s.


def worker(batch, entry_id):
    config = read_json(batch/'benchmark-plan.json')
    entry = next(e for e in config['entries'] if e['id']==entry_id)
    run = Path(config['repo'])/'var/runs'/entry['run_id']
    sys.stdout = sys.stderr = (run/'worker.log').open('a',buffering=1)
    plan = read_json(run/'plan.json')
    verify_source(plan['source'])
    site = read_json(Path(plan['source']['root'])/'configs/rsi/site.beijing.json')
    site['runtime_root'] = plan['runtime_root']
    os.environ.update(worker_environment(site))
    os.environ.update(DEEPSEEK_MODEL=plan['opponent']['model'],DEEPSEEK_BASE_URL=plan['opponent']['base_url'])
    credential = Path(plan['runtime_root'])/'deepseek-key'
    auth = tomllib.loads((Path(site['codex_home'])/'config.toml').read_text())
    provider = auth['model_providers'][auth['model_provider']]
    fd = os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:
        stream.write(provider['experimental_bearer_token'])
    os.environ['DEEPSEEK_KEY_FILE'] = str(credential)
    atomic_json(run/'progress.json',{'stage':'starting','updated':utc()})
    try:
        if model_identity(plan['model_path']) != plan['model_identity']:
            raise ValueError('Model differs from prepared checkpoint')
        from .deepseek_final import run as evaluate
        result = evaluate(plan['model_path'],plan['candidate'],run/'final-test',
                          inference_python=site['inference_python'],games=plan['games'],parallel_games=plan['parallel_games'])
        atomic_json(run/'performance.json',performance(result['wins'],result['draws'],result['losses'],config['teacher_rating']))
        from .random_moves import write_report
        write_report(run/'final-test')
        atomic_json(run/'progress.json',{'stage':'completed','updated':utc()})
    except BaseException as error:
        atomic_json(run/'failure.json',{'type':type(error).__name__,'error':str(error),'updated':utc()})
        traceback.print_exc()
        raise
    finally:
        credential.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['launch','watch','worker','collect'])
    parser.add_argument('--repo',type=Path,default=Path('/user/lyh4170/LLMGo!'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--batch',type=Path)
    parser.add_argument('--entry')
    args = parser.parse_args(argv)
    if args.action=='worker':
        return worker(args.batch,args.entry)
    if args.action=='launch':
        args.batch = prepare(args.repo,args.config)
    lock = (args.batch/'operator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.action=='launch':
        submit(args.batch)
    if args.action=='collect':
        return catalog(args.batch)
    return watch(args.batch)
