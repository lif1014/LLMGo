"""Safe training boundaries, durable review and HIGH admission without a cluster."""
import json
from pathlib import Path
import subprocess

import pytest

from llmgo.rsi.contracts import atomic_json, digest, file_hash, model_identity, read_json
from llmgo.rsi.training_review import publish, request, validate, commit, decision, expire
from llmgo.rsi.training_loop import TrainingLoop
from llmgo.remote.high_admission import high_slot, AdmissionBusy


def review_request(tmp_path):
    checkpoint=tmp_path/'checkpoint.json';atomic_json(checkpoint,{'step':5})
    metrics=tmp_path/'metrics.json';atomic_json(metrics,{'score':.5})
    return publish(tmp_path/'review',{'step':5,'max_steps':10,'available_steps':[0,5],
        'checkpoint':{'path':str(checkpoint),'sha256':file_hash(checkpoint)},
        'evaluations':[{'path':str(metrics),'sha256':file_hash(metrics)}]})


def test_review_is_bound_to_evidence_and_committed_once(tmp_path):
    path=review_request(tmp_path);r=request(path)
    value={'request_id':r['request_id'],'action':'continue','reason':'loss evidence supports another segment','selected_step':5}
    assert decision(path) is None
    assert commit(path,value)==commit(path,value)
    with pytest.raises(ValueError,match='cannot be replaced'):
        commit(path,{**value,'action':'finish'})
    with pytest.raises(ValueError,match='current optimizer'):
        validate(path,{**value,'selected_step':0})
    with pytest.raises(ValueError,match='evaluated checkpoint'):
        validate(path,{**value,'selected_step':4})
    atomic_json(tmp_path/'metrics.json',{'score':1})
    with pytest.raises(ValueError,match='evidence changed'):
        decision(path)


def test_expired_review_rejects_a_late_meta_answer(tmp_path):
    path=review_request(tmp_path);r=request(path)
    assert expire(path)
    with pytest.raises(ValueError,match='expired'):
        commit(path,{'request_id':r['request_id'],'action':'finish','reason':'too late','selected_step':5})
    assert decision(path) is None


def test_export_intermediate_checkpoint_without_final_result(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('review_export',Path(__file__).resolve().parents[1]/'scripts/export_grpo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output=tmp_path/'experiment/grpo';base=tmp_path/'base';base.mkdir()
    atomic_json(base/'config.json',{});atomic_json(base/'tokenizer_config.json',{'fixture':True})
    atomic_json(output/'recipe.json',{'steps':10,'model_path':str(base)})
    checkpoint=output/'native-checkpoints/global_step_5';(checkpoint/'actor').mkdir(parents=True)
    (checkpoint/'actor/shard.pt').write_bytes(b'weights')
    atomic_json(output/'checkpoints/receipt-00000005.json',{'step':5,'native_verl':True,'path':str(checkpoint),'file_sizes':{'actor/shard.pt':7}})
    target=output.parent/'scaling/models/step-00000005';calls=[]
    def merge(argv,**kw):
        calls.append(argv);dest=Path(argv[argv.index('--target_dir')+1]);atomic_json(dest/'config.json',{})
        (dest/'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setattr(module.subprocess,'run',merge)
    monkeypatch.setattr(module,'normalize_export_weights',lambda a,b:{'fixture':True})
    monkeypatch.setattr(module.sys,'argv',['export','--output',str(output),'--step','5','--target',str(target)])
    module.main();module.main()
    assert len(calls)==1 and not (output/'result.json').exists()
    receipt=read_json(target.with_name(target.name+'.submission.json'))
    assert receipt['model_identity']==model_identity(target)
    assert (target/'tokenizer_config.json').is_file()
    monkeypatch.setattr(module.sys,'argv',['export','--output',str(output),'--step','10','--target',str(target)])
    with pytest.raises(RuntimeError,match='identity changed'):module.main()


@pytest.fixture
def loop_fixture(tmp_path):
    root=tmp_path;repo=root/'repo';repo.mkdir();model=root/'initial';model.mkdir()
    atomic_json(model/'config.json',{'step':0})
    (model/'model.safetensors').write_bytes(b'fixture weights')
    atomic_json(repo/'configs/grpo/default.json',{'steps':10,'enable_thinking':False,
        'resume_on_restart':True,'validation_every':5,'checkpoint_every':5})
    output=repo/'var/runs/demo-j000001'
    source={'root':str(repo),'sha256':'protected'}
    p={'site':{'user_root':str(root),'repo':str(repo),'gpu_count':2,'engine':'fixture',
        'campaign_root':str(root/'campaigns'),'verl_venv':str(root/'verl'),'review_timeout_seconds':5},
        'source':source,'candidate_source':{**source,'sha256':'candidate'},'model_path':str(model),
        'job_id':'j000001','campaign_id':'demo','candidate_id':'r000001','output':str(output),
        'command':{'argv':['{python}','scripts/run_grpo.py','--template','{candidate}/configs/grpo/default.json']},
        'lineage_steps':15,'lineage_gpu_hours':2.}
    calls=[];tick=[0.]
    def clock():
        tick[0]+=1.;return tick[0]
    def runner(argv,**kw):
        if any(str(part).endswith('chess_r1_gpu_entry.py') for part in argv):
            step=int(kw['env']['CHESS_R1_SEGMENT_TARGET']);calls.append(('train',step))
            previous=max((int(f.stem.split('-')[1]) for f in (output/'grpo/checkpoints').glob('receipt-*.json')),default=0)
            assert previous==step-5
            atomic_json(output/'grpo/checkpoints'/f'receipt-{step:08d}.json',
                {'step':step,'resume_contains_optimizer_rng_and_data':True,'native_verl':True})
        else:
            step=int(argv[argv.index('--step')+1]);target=Path(argv[argv.index('--target')+1])
            calls.append(('export',step));atomic_json(target/'config.json',{'step':step})
            (target/'model.safetensors').write_bytes(f'fixture weights {step}'.encode())
            atomic_json(target.with_name(target.name+'.submission.json'),{'model_path':str(target),'model_identity':model_identity(target)})
        return subprocess.CompletedProcess(argv,0)
    def evaluator(payload):
        step=json.loads((Path(payload['model_path'])/'config.json').read_text())['step']
        calls.append(('eval',step,payload['protocol']['opponent']['level']))
        # Include a decline, proving best-envelope != raw curve.
        games=[{'game_id':f'{i}-{side}','pair_id':i,'model_seat':side,
                'result':'draw' if step!=5 else f'{side}_win'} for i in range(10) for side in ('white','black')]
        path=Path(payload['output'])/'metrics.json'
        atomic_json(path,{'games':games,'model_identity':model_identity(payload['model_path']),
            'source_identity':payload['source']['sha256'],'protocol_identity':digest(payload['protocol'])})
        return path
    def sleeper(_):
        path=Path(read_json(output/'scaling/pending-review.json')['path']);r=request(path)
        step=r['payload']['step']
        commit(path,{'request_id':r['request_id'],'action':'continue' if step==5 else 'finish',
            'selected_step':step if step==5 else 5,'reason':'inspect every loss; select earlier observed best'})
    return p,dict(runner=runner,evaluator=evaluator,engine_probe=lambda _,spec:{'engine':'fixture','spec':spec},clock=clock,sleep=sleeper),calls


def test_five_then_ten_review_selects_earlier_checkpoint_and_replays(loop_fixture):
    p,kwargs,calls=loop_fixture
    result=TrainingLoop(p,**kwargs).run()
    assert result['training_loop_action']=='finish' and result['selected_step']==5
    assert result['lineage_steps']==20 and result['lineage_gpu_hours']>2
    assert [row for row in calls if row[0]=='train']==[('train',5),('train',10)]
    assert len([row for row in calls if row[0]=='eval'])==6
    history=read_json(Path(p['output'])/'scaling/curve.json')['points']
    assert [row['lineage_step'] for row in history]==[15,20,25]
    assert [row['score'] for row in history]==[.5,1.,.5]
    assert [row['best_observed'] for row in history]==[.5,1.,1.]
    count=len(calls);assert TrainingLoop(p,**kwargs).run()==result;assert len(calls)==count
    last=Path(p['output'])/'scaling/reviews/step-00000010/request.json'
    r=request(last)
    with pytest.raises(ValueError,match='exhausted'):
        validate(last,{'request_id':r['request_id'],'action':'continue','reason':'another segment'})


def test_restart_after_validation_reuses_checkpoint_before_training(loop_fixture):
    p,kwargs,calls=loop_fixture
    loop=TrainingLoop(p,**kwargs)
    loop.validation(0,p['model_path']);loop.train(5);model=loop.export(5);loop.validation(5,model)
    def finish(_):
        path=Path(read_json(Path(p['output'])/'scaling/pending-review.json')['path']);r=request(path)
        commit(path,{'request_id':r['request_id'],'action':'replan','selected_step':5,'reason':'new reward hypothesis'})
    result=TrainingLoop(p,**{**kwargs,'sleep':finish}).run()
    assert result['training_loop_action']=='replan'
    assert [row for row in calls if row[0]=='train']==[('train',5)]
    assert len([row for row in calls if row[0]=='eval'])==4


def test_training_validation_submits_both_levels_to_one_pool(loop_fixture):
    p,kwargs,calls=loop_fixture
    p['site']['eval_shared_levels']=True
    panels=[]
    def panel(payloads):
        panels.append(payloads)
        return [kwargs['evaluator'](item) for item in payloads]
    result=TrainingLoop(p,**kwargs,panel_evaluator=panel).run()
    assert result['training_loop_action']=='finish'
    assert len(panels)==3  # initial, step 5, step 10; one shared call each
    for batch in panels:
        assert [p['protocol']['opponent']['level'] for p in batch]==[2,3]
        assert len({p['panel_output'] for p in batch})==1
        assert len({p['games_output'] for p in batch})==1
        assert batch[0]['game_id_prefix']!=batch[1]['game_id_prefix']


def test_review_timeout_retains_model_and_native_receipt(loop_fixture):
    p,kwargs,calls=loop_fixture
    result=TrainingLoop(p,**{**kwargs,'sleep':lambda _:None}).run()
    assert result['training_loop_action']=='review_timeout' and result['selected_step']==5
    assert Path(result['checkpoint_receipt']).is_file()
    assert not (Path(p['output'])/'scaling/final.json').exists()


def test_high_slot_blocks_other_own_jobs_and_unknown_intents(tmp_path):
    class Transport:
        site={'user_root':str(tmp_path),'repo':str(tmp_path/'repo')}
        rows=[]
        def command(self,args):
            return {'tasks':self.rows} if args[:2]==['task','list'] else {'status':'Succeeded'}
    transport=Transport();first=tmp_path/'one/submission.json';second=tmp_path/'two/submission.json'
    with high_slot(transport,first):atomic_json(first,{'status':'UNKNOWN'})
    with pytest.raises(AdmissionBusy,match='uncertain'):
        with high_slot(transport,second):raise AssertionError('must not submit')
    atomic_json(first,{'task_id':12})
    transport.rows=[{'priority':'HIGH','status':'Running'}]
    with pytest.raises(AdmissionBusy,match='own HIGH'):
        with high_slot(transport,second):raise AssertionError('must not submit')
    transport.rows=[{'priority':'NORMAL','status':'Running'}]
    transport.rows.append({'status':'Succeeded'})  # Legacy terminal records may omit priority.
    with high_slot(transport,second):atomic_json(second,{'status':'UNKNOWN'})


def test_high_transport_submits_once_and_reserves_unknown_outcome(tmp_path,monkeypatch):
    from llmgo.remote.cctl import CctlExecutor, SubmissionUncertain
    from llmgo.rsi.memory import MemoryStore
    site={'user_root':str(tmp_path),'repo':str(tmp_path/'repo'),'priority':'HIGH','cctl_bin':'cctl','project':'rsi'}
    calls=[]
    def runner(argv,**kw):
        calls.append(argv[1:3])
        if argv[1:3]==['pool','list']:value=[{'name':'model-autosft'}]
        elif argv[1:3]==['task','list']:value={'tasks':[]}
        else:
            assert argv[1:3]==['job','create']
            raise subprocess.TimeoutExpired(argv,90)
        return subprocess.CompletedProcess(argv,0,json.dumps(value),'')
    t=CctlExecutor(site,MemoryStore(tmp_path/'memory'),runner)
    monkeypatch.setattr(t,'request',lambda path:(['job','create'],str(path)))
    first=tmp_path/'first/manifest.json';second=tmp_path/'second/manifest.json'
    with pytest.raises(SubmissionUncertain):t.submit(first)
    assert read_json(first.with_name('submission.json'))['priority']=='HIGH'
    with pytest.raises(AdmissionBusy):t.submit(second)
    assert calls.count(['job','create'])==1


def test_controller_review_guards_code_and_finishes_without_duplicate_evaluation(loop_fixture):
    from llmgo.rsi.controller import Controller
    p,kwargs,calls=loop_fixture
    site=p['site'];site.update(worktree_root=str(Path(site['user_root'])/'workspaces'),base_model=p['model_path'])
    config=Path(site['repo'])/'configs/grpo/default.json'
    atomic_json(config,{**read_json(config),'answer_protocol':'chess-r1-boxed-san-v2'})
    c=Controller.create(site,'demo');c.state['phase']='PLANNING';c.save()
    workspace=c.workspace()
    atomic_json(workspace/'proposal.json',{'hypothesis':'fixture','jobs':[{'argv':['{python}','scripts/run_grpo.py']}],'evaluation':False})
    c.prepare(workspace/'proposal.json');c.step()
    job=c.state['jobs'][c.state['active_job']]
    p=read_json(job['manifest'])['payload'];loop=TrainingLoop(p,**kwargs)
    # Only publish a review: Controller must discover it without scheduler calls.
    loop.validation(0,p['model_path']);loop.train(5);point=loop.validation(5,loop.export(5))
    def handoff(_):
        response=c.step();assert response['status']=='needs_training_review'
        review=request(response['request']);ws=Path(response['workspace']);proposal=ws/'proposal.json'
        value={'request_id':review['request_id'],'action':'continue','selected_step':5,'reason':'fixture loss analysis'}
        atomic_json(proposal,{'training_review':value})
        cfg=ws/'configs/grpo/default.json';original=cfg.read_bytes();atomic_json(cfg,{**read_json(cfg),'steps':20})
        with pytest.raises(ValueError,match='frozen training plan'):
            c.finish_training_review({'proposal_path':str(proposal)})
        cfg.write_bytes(original)
        atomic_json(proposal,{'training_review':{**value,'action':'finish'}})
        c.finish_training_review({'proposal_path':str(proposal)})
    loop.sleep=handoff
    answer=loop.review(point);assert answer['action']=='finish'
    result=loop.run();artifact=Path(p['output'])/'research-result.json';atomic_json(artifact,result)
    c._consume(job,{'path':str(artifact),'sha256':file_hash(artifact)},p)
    c.state['active_job']=None;c.save();c.step();c.step()
    assert c.state['phase']=='COMPLETE'
    assert c.state['final_checkpoint']['model_path']==result['model_path']
    assert len(c.state['jobs'])==1


def test_replan_preserves_review_edits_for_next_round(loop_fixture):
    from llmgo.rsi.controller import Controller
    p,kwargs,calls=loop_fixture
    site=p['site'];site.update(worktree_root=str(Path(site['user_root'])/'workspaces'),base_model=p['model_path'])
    cfg=Path(site['repo'])/'configs/grpo/default.json'
    atomic_json(cfg,{**read_json(cfg),'answer_protocol':'chess-r1-boxed-san-v2'})
    c=Controller.create(site,'demo');c.state['phase']='PLANNING';c.save();ws=c.workspace()
    atomic_json(ws/'proposal.json',{'hypothesis':'fixture','jobs':[{'argv':['{python}','scripts/run_grpo.py']}],'evaluation':False})
    c.prepare(ws/'proposal.json');c.step();job=c.state['jobs'][c.state['active_job']]
    payload=read_json(job['manifest'])['payload'];loop=TrainingLoop(payload,**kwargs)
    loop.validation(0,payload['model_path']);loop.train(5);point=loop.validation(5,loop.export(5))
    def replan(_):
        response=c.step();r=request(response['request']);review_ws=Path(response['workspace'])
        config=review_ws/'configs/grpo/default.json';atomic_json(config,{**read_json(config),'steps':20})
        proposal=review_ws/'proposal.json';atomic_json(proposal,{'training_review':{'request_id':r['request_id'],'action':'replan','selected_step':5,'reason':'new curriculum'}})
        c.finish_training_review({'proposal_path':str(proposal)})
    loop.sleep=replan;assert loop.review(point)['action']=='replan'
    result=loop.run();path=Path(payload['output'])/'research-result.json';atomic_json(path,result)
    c._consume(job,{'path':str(path),'sha256':file_hash(path)},payload);c.state['active_job']=None;c.save();c.step();c.step()
    assert c.state['phase']=='PLANNING'
    ws=c.workspace();assert read_json(ws/'configs/grpo/default.json')['steps']==20
    assert c.state['working']['lineage_steps']==5
