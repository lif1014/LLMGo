#!/usr/bin/env python3
"""Isolated GPU acceptance: real updates 1–5, native resume, updates 6–10.

Decisions are scripted engineering checks, not autonomous research or scores.
The normal protected Worker still performs all full L2/L3 games and exports.
"""
import argparse
from pathlib import Path
import sys
import threading
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.contracts import atomic_json, load_site, read_json
from llmgo.rsi.controller import Controller
from llmgo.rsi.ladder import batch_stats
from llmgo.rsi.memory import MemoryStore


class AcceptanceReview:
    def __init__(self,memory):
        self.memory=memory

    def turn(self,workspace,prompt):
        review=read_json(workspace/'context.json')['training_review']
        step=review['payload']['step']
        if step not in (5,10):
            raise ValueError('Unexpected acceptance boundary')
        value={'request_id':review['request_id'],'action':'continue' if step==5 else 'finish',
               'selected_step':step,'reason':'Scripted engineering acceptance of the native 5→10 resume; not a research decision'}
        atomic_json(workspace/'proposal.json',{'training_review':value})
        self.memory.append('acceptance_scripted_decision',value)
        return {'status':'ready','summary':value['reason'],'proposal_path':str(workspace/'proposal.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',default=str(Path(__file__).resolve().parents[1]/'configs/rsi/site.beijing.json'))
    parser.add_argument('--campaign',default='grpo-review-acceptance-20261103')
    args=parser.parse_args()
    site=load_site(args.site)
    if not site.get('training_review') or site['executor_mode']!='external':
        raise ValueError('Acceptance requires the reviewed training configuration')
    root=Path(site['campaign_root'])/args.campaign
    report=Path(site['repo'])/'var/campaigns/chess-grpo-v1/operator-audits/training-review-acceptance.json'
    if (root/'campaign.json').exists():
        controller=Controller(root)
        if not controller.state.get('engineering_acceptance'):
            raise ValueError('Refuse to use a research campaign for engineering acceptance')
    else:
        controller=Controller.create(site,args.campaign)
        controller.state.update(phase='PLANNING',engineering_acceptance=True)
        controller.save()
        ws=controller.workspace()
        config=ws/'configs/grpo/default.json'
        recipe=read_json(config)
        recipe.update(steps=10,prompt_batch_size=16,minibatch_prompts=8,
                      rollout_max_num_seqs=16,forward_micro_batch_size=2)
        atomic_json(config,recipe)
        atomic_json(ws/'proposal.json',{'hypothesis':'Engineering acceptance only: native resume at step 5 and complete step 10',
            'jobs':[{'argv':['{python}','scripts/run_grpo.py','--output','{output}','--model','{model}']}],
            'evaluation':False,'recommend':False})
        controller.prepare(ws/'proposal.json')
    stop=threading.Event()
    atomic_json(report,{'status':'running','kind':'scripted_engineering_acceptance','campaign':str(root),
        'description':'Real 5→10 native resume and 120 development games; scripted decisions, not autonomous research'})
    def executor():
        memory=MemoryStore(root/'executor')
        with memory.lock('service'):
            while not stop.is_set():
                try:
                    state=MemoryStore(root).state()
                    if state.get('active_job'):
                        controller.lifecycle.advance(state['jobs'][state['active_job']])
                except Exception as error:
                    memory.append('acceptance_executor_error',{'type':type(error).__name__,'error':str(error)})
                stop.wait(5)
    thread=threading.Thread(target=executor,daemon=True);thread.start()
    try:
        result=controller.run(backend=AcceptanceReview(controller.memory),poll_seconds=5)
        if result['status']!='completed':
            atomic_json(report,{'status':'not_passed','campaign':str(root),'controller':result})
            return 1
        job=next(job for job in controller.state['jobs'].values() if job['kind']=='research')
        output=Path(read_json(job['manifest'])['payload']['output'])
        resume=read_json(output/'grpo/resume.json')
        if resume['step']!=5:
            raise ValueError('Second segment did not restore the native step-5 checkpoint')
        curve=read_json(output/'scaling/curve.json')['points']
        if [row['step'] for row in curve]!=[0,5,10]:
            raise ValueError('Acceptance curve must contain step 0, 5 and 10')
        for step in (5,10):
            receipt=read_json(output/'grpo/checkpoints'/f'receipt-{step:08d}.json')
            if not receipt['resume_contains_optimizer_rng_and_data']:
                raise ValueError('Missing native state')
        for point in curve:
            for level in point['levels']:
                batch_stats(read_json(level['ref']['path']),expected_games=20)
        atomic_json(report,{'status':'passed','kind':'scripted_engineering_acceptance',
            'campaign':str(root),'output':str(output),'native_resume_step':resume['step'],
            'steps':[0,5,10],'games':120,'final_checkpoint':controller.state['final_checkpoint']})
        return 0
    except Exception as error:
        atomic_json(report,{'status':'failed','campaign':str(root),'type':type(error).__name__,
            'error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        stop.set();thread.join(timeout=10)


if __name__=='__main__':
    raise SystemExit(main())
