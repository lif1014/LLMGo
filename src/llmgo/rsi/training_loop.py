"""One allocated job alternates native GRPO, protected games and Meta review."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .contracts import atomic_json, digest, file_hash, inside, model_identity, protocol, read_json, utc
from .evaluator import engine_identity, evaluate, evaluate_many
from .ladder import batch_stats, opponent
from .training_review import publish, decision
from .research_control import require_research
from llmgo.training.chess_r1_config import token_budgets


def freeze_recipe(payload, output):
    candidate = Path(payload['candidate_source']['root'])
    argv = payload['command']['argv']
    template = Path(argv[argv.index('--template')+1].replace('{candidate}',str(candidate))) if '--template' in argv else candidate/'configs/grpo/default.json'
    if not template.is_absolute():
        template = candidate/template
    template = inside(candidate,template)
    recipe = read_json(template)
    # Reject invalid token budgets before allocating time to the step-0 panel.
    token_budgets(recipe)
    recipe.update(model_path=payload['model_path'], run_id=output.name, run_directory=output.name,
                  attempt=output.name, allocated_gpus=payload.get('resources',{}).get('gpu_count',payload['site']['gpu_count']),
                  training_review=True)
    recipe.setdefault('validation_every',5)
    recipe.setdefault('checkpoint_every',5)
    if recipe.get('enable_thinking') is not False or recipe.get('harness') or recipe.get('skills') or recipe.get('sft'):
        raise ValueError('Reviewed training requires the fixed non-thinking RL protocol')
    for key in ('steps','validation_every','checkpoint_every'):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError('Invalid training interval: '+key)
    path = output/'grpo/recipe.json'
    if path.exists() and read_json(path) != recipe:
        raise ValueError('Frozen recipe changed')
    atomic_json(path,recipe)
    return recipe


class TrainingLoop:
    def __init__(self,payload, *, runner=subprocess.run, evaluator=evaluate, engine_probe=engine_identity,
                 clock=time.monotonic, sleep=time.sleep,panel_evaluator=evaluate_many):
        self.p = payload
        self.site = payload['site']
        self.output = inside(self.site['user_root'],payload['output'])
        self.root = self.output/'scaling'
        self.root.mkdir(parents=True,exist_ok=True)
        self.recipe = freeze_recipe(payload,self.output)
        self.runner,self.evaluator,self.engine_probe = runner,evaluator,engine_probe
        self.panel_evaluator=panel_evaluator
        self.clock,self.sleep = clock,sleep
        self.campaign = Path(self.site['campaign_root'])/payload['campaign_id']
        self.history = read_json(self.root/'curve.json')['points'] if (self.root/'curve.json').exists() else []
        self.ancestry=[]
        if payload.get('parent_curve'):
            parent=read_json(inside(self.site['user_root'],payload['parent_curve']))
            self.ancestry=[row for row in parent.get('ancestry',[])+parent['points']
                           if row['lineage_step']<payload.get('lineage_steps',0)]
        self.cost = read_json(self.root/'cost.json') if (self.root/'cost.json').exists() else {'training_seconds':0.,'evaluation_seconds':0.,'review_seconds':0.}
        identity = {'recipe_sha256':digest(self.recipe),'candidate_source':payload['candidate_source']['sha256'],
                    'protected_source':payload['source']['sha256'],'job_id':payload['job_id'],
                    'parent_curve_sha256':file_hash(payload['parent_curve']) if payload.get('parent_curve') else None}
        if (self.root/'identity.json').exists() and read_json(self.root/'identity.json') != identity:
            raise ValueError('Training loop identity changed')
        atomic_json(self.root/'identity.json',identity)

    def progress(self,stage,step,**extra):
        atomic_json(self.output/'grpo/progress.json',{'stage':stage,'step':step,'updated':utc(),**extra})

    def command(self,argv,stage,step,env=None):
        self.progress(stage,step)
        (self.output/'logs').mkdir(exist_ok=True)
        started=self.clock()
        with (self.output/'logs'/f'{stage}-{step:08d}.log').open('a') as log:
            self.runner(argv,cwd=self.p['candidate_source']['root'],env=env or os.environ.copy(),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
        return self.clock()-started

    def train(self,target):
        env=dict(os.environ,CHESS_R1_SEGMENT_TARGET=str(target))
        seconds=self.command([sys.executable,'-B',str(Path(self.p['candidate_source']['root'])/'scripts/chess_r1_gpu_entry.py'),
            str(self.output/'grpo/recipe.json'),'--output',str(self.output/'grpo')],'training',target,env)
        self.cost['training_seconds']+=seconds
        atomic_json(self.root/'cost.json',self.cost)
        receipt=read_json(self.output/'grpo/checkpoints'/f'receipt-{target:08d}.json')
        if receipt['step'] != target or not receipt.get('resume_contains_optimizer_rng_and_data'):
            raise ValueError('Training did not commit the requested native optimizer checkpoint')
        return receipt

    def export(self,step):
        target=self.root/'models'/f'step-{step:08d}'
        python=str(Path(self.site['verl_venv'])/'bin/python')
        # Export is editable training code. Use the candidate's frozen exporter
        # so its fixes apply; chess evaluation still uses the protected source.
        seconds=self.command([python,'-B',str(Path(self.p['candidate_source']['root'])/'scripts/export_grpo.py'),
            '--output',str(self.output/'grpo'),'--step',str(step),'--target',str(target)],'exporting',step)
        self.cost['export_seconds']=self.cost.get('export_seconds',0.)+seconds
        atomic_json(self.root/'cost.json',self.cost)
        receipt=read_json(target.with_name(target.name+'.submission.json'))
        if model_identity(receipt['model_path']) != receipt['model_identity']:
            raise ValueError('Exported model identity changed')
        return receipt['model_path']

    def validation(self,step,model):
        if any(point['step']==step for point in self.history):
            return next(point for point in self.history if point['step']==step)
        self.progress('validating_games',step)
        identity=model_identity(model)
        rows=[];batches=[]
        before=self.clock()
        for level in (2,3):
            spec=opponent(level)
            engine=self.engine_probe(self.site['engine'],spec)
            pin={k:v for k,v in engine.items() if k!='spec'}
            if self.p.get('engine_pin') and self.p['engine_pin']!=pin:
                raise ValueError('Validation engine differs from the campaign baseline')
            pin_path=self.root/'engine-pin.json'
            if pin_path.exists() and read_json(pin_path)!=pin:
                raise ValueError('Validation engine changed during training')
            atomic_json(pin_path,pin)
            out=self.root/'evaluations'/f'step-{step:08d}'/f'L{level}'
            out.mkdir(parents=True,exist_ok=True)
            payload={**self.p,'kind':'evaluate','role':'training_validation','output':str(out),
                'candidate_id':self.p['candidate_id'], 'track':'player','model_path':model,
                'expected_model_sha256':identity['sha256'],'protocol':protocol(spec,self.site['engine']),
                'engine_identity':engine,'seed_schedule':[10000*int(level*10)+i for i in range(10)],
                'comparison_id':f'validation20:L{level}:b0', 'reference_source':self.p['candidate_source']}
            if self.site.get('eval_shared_levels'):
                payload.update(panel_output=str(out.parent),games_output=str(self.output),
                    game_id_prefix=f"{self.p['job_id']}-s{step:08d}-L{level}")
            batches.append(payload)
        paths=(self.panel_evaluator(batches) if self.site.get('eval_shared_levels')
               else [self.evaluator(payload) for payload in batches])
        if len(paths)!=len(batches):raise ValueError('Incomplete validation panel')
        for payload,path in zip(batches,paths,strict=True):
            level=payload['protocol']['opponent']['level']
            # Completed levels are reused only through the protected evaluator's game checks.
            path=Path(path)
            metrics=read_json(path)
            if (metrics['model_identity']['sha256']!=identity['sha256']
                    or metrics['protocol_identity']!=digest(payload['protocol'])
                    or metrics['source_identity']!=self.p['source']['sha256']):
                raise ValueError('Validation returned the wrong model/protocol/source')
            stats=batch_stats(metrics,expected_games=20)
            rows.append({'level':level,'track':'player','statistics':stats,
                         'ref':{'path':str(path),'sha256':file_hash(path)}})
        self.cost['evaluation_seconds']+=self.clock()-before
        atomic_json(self.root/'cost.json',self.cost)
        score=sum(row['statistics']['score'] for row in rows)/2
        se=math.sqrt(sum(row['statistics']['pair_standard_error']**2 for row in rows))/2
        lineage=self.p.get('lineage_steps',0)+step
        point={'step':step,'lineage_step':lineage,'model_path':model,'model_identity':identity,
            'score':score,'pair_standard_error':se,'descriptive_interval':[max(0,score-1.96*se),min(1,score+1.96*se)],
            'levels':rows,'recipe_sha256':digest(self.recipe),'candidate_source':self.p['candidate_source']['sha256'],
            'cost':self.cost.copy(),'gpu_count':self.recipe['allocated_gpus'],
            'lineage_gpu_hours':self.p.get('lineage_gpu_hours',0.)+sum(self.cost.values())*self.recipe['allocated_gpus']/3600,
            'cost_scope':'Recorded training, export, evaluation and review stages; excludes queue/startup and failed-stage time.', 'utc':utc()}
        self.history.append(point)
        best=max(self.history,key=lambda value:value['score'])
        running=max((value['score'] for value in self.ancestry),default=-1.)
        for value in self.history:
            running=max(running,value['score']);value['best_observed']=running
        atomic_json(self.root/'curve.json',{'points':self.history,'ancestry':self.ancestry,'parent_curve':self.p.get('parent_curve'),
            'note':'Adaptive development scores; raw points and observed best are both retained.'})
        atomic_json(self.root/'best.json',{'step':best['step'],'score':best['score'],'model_path':best['model_path']})
        return point

    def review(self,point):
        step=point['step']
        receipt=self.output/'grpo/checkpoints'/f'receipt-{step:08d}.json'
        payload={'job_id':self.p['job_id'],'campaign_id':self.p['campaign_id'],'step':step,
            'max_steps':self.recipe['steps'],'recipe_sha256':digest(self.recipe),
            'candidate_source_sha256':self.p['candidate_source']['sha256'],
            'checkpoint':{'path':str(receipt),'sha256':file_hash(receipt)},
            'evaluations':[row['ref'] for row in point['levels']],
            'available_steps':[row['step'] for row in self.history],
            'curve_path':str(self.root/'curve.json'),'model_path':point['model_path']}
        path=publish(self.root/'reviews'/f'step-{step:08d}',payload)
        atomic_json(self.root/'pending-review.json',{'path':str(path),'request_id':read_json(path)['request_id']})
        self.progress('awaiting_meta_review',step,request=str(path))
        started=self.clock()
        while True:
            require_research(self.campaign)
            answer=decision(path)
            if answer is not None:
                self.cost['review_seconds']+=self.clock()-started
                atomic_json(self.root/'cost.json',self.cost)
                atomic_json(path.with_name('ack.json'),{'request_id':answer['request_id'],'action':answer['action'],'utc':utc()})
                return answer
            self.sleep(2)

    def run(self):
        final=self.root/'final.json'
        if final.exists():
            return read_json(final)
        if not self.history:
            self.validation(0,self.p['model_path'])
        interval=self.recipe['validation_every']
        current=max(row['step'] for row in self.history)
        while True:
            require_research(self.campaign)
            if current:
                answer=self.review(next(row for row in self.history if row['step']==current))
                if answer['action']!='continue':
                    break
            target=min(current+interval,self.recipe['steps'])
            if target<=current:
                raise ValueError('Exhausted training plan without a final decision')
            self.train(target)
            model=self.export(target)
            self.validation(target,model)
            current=target
        selected=next(row for row in self.history if row['step']==answer['selected_step'])
        if model_identity(selected['model_path'])!=selected['model_identity']:
            raise ValueError('Selected checkpoint changed')
        result={'status':'ready','model_path':selected['model_path'],'model_identity':selected['model_identity'],
            'training_loop_action':answer['action'],'selected_step':selected['step'],
            'lineage_steps':self.p.get('lineage_steps',0)+selected['step'],
            'lineage_gpu_hours':self.p.get('lineage_gpu_hours',0.)+sum(self.cost.values())*self.recipe['allocated_gpus']/3600,
            'reason':answer['reason'],'evaluations':selected['levels'],'curve':str(self.root/'curve.json'),
            'checkpoint_receipt':str(self.output/'grpo/checkpoints'/f'receipt-{selected["step"]:08d}.json') if selected['step'] else None}
        if answer['action']!='review_timeout':
            atomic_json(final,result)
        atomic_json(self.output/'submission.json',result)
        self.progress('selected' if answer['action']=='finish' else answer['action'],current)
        return result
