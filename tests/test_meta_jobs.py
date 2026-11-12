import json
from pathlib import Path

import pytest

from llmgo.rsi.contracts import atomic_json,read_json
from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.meta_jobs import GpuMetaBackend,ApiMetaBackend,choose_pool,pool_snapshot,meta_backend,meta_overview


@pytest.mark.parametrize('free,queues,want',[
    ([1,8,4,1],[0,0,0,0],'model-new'),
    ([0,0,4,1],[3,1,2,0],'model-claw-tool'),
    ([0,0,0,2],[1,1,1,9],'model-codeagent2'),
    ([4,4,4,4],[0,0,0,0],'model-autosft'),
    ([0,0,0,0],[5,2,3,4],'model-new'),
])
def test_available_card_first_policy(free,queues,want):
    from llmgo.rsi.meta_jobs import POOL_IDS
    snapshots={p:{'free_gpus':f,'queued_people':q,'queued_tasks':q} for p,f,q in zip(POOL_IDS,free,queues)}
    assert choose_pool(snapshots)==want
    if want=='model-autosft':assert choose_pool(snapshots,exclude={'model-autosft'})=='model-new'


def test_queue_counts_people_not_jobs_and_only_h800():
    class Platform:
        def command(self,args):
            if args[0]=='pool':return {'edges':{'resource_quotas':[{'name':'h800','capacity':8,'used':8,'edges':{'cluster':{'name':'beijing'}}}]}}
            assert '--own' not in args
            return {'tasks':[{'creator':'same','gpuModel':'h800','resourcesTotal':{'gpuCount':1}}]*7+
                [{'creator':'cpu','resourcesTotal':{}},{'creator':'other','gpuModel':'h100','resourcesTotal':{'gpuCount':1}}]}
    assert pool_snapshot(Platform(),'model-autosft')['queued_people']==1


def fixture(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();(repo/'script.py').write_text('print(1)')
    workspace=tmp_path/'workspace';workspace.mkdir();(workspace/'proposal.json').write_text('{"hypothesis":"original"}')
    atomic_json(workspace/'context.json',{})
    memory=MemoryStore(tmp_path/'campaign')
    site={'meta_execution':'gpu_job','meta_model':'Qwen3.5-4B','meta_thinking':True,'repo':str(repo),
        'user_root':str(tmp_path),'project':'rsi','billing_account_id':'fixture','image':'fixture','control_python':'python',
        'resource_pool':'model-autosft','priority':'HIGH'}
    return site,memory,workspace


def test_killed_result_and_edits_are_discarded_then_reselect(tmp_path):
    site,memory,workspace=fixture(tmp_path)
    clock=[0.];creates=[];stages=[]
    class Platform:
        def command(self,args):
            if args[:2]==['job','get']:
                index=int(args[2]);attempt=sorted((memory.root/'meta_requests').glob('*/attempts/*'))[index-1]
                result={'request_id':read_json(attempt/'worker-request.json')['request_id'],
                    'answer':{'status':'needs_input','summary':'INVALID' if index==1 else 'VALID','proposal_path':None}}
                atomic_json(attempt/'result.json',result)
                if index==1:(workspace/'proposal.json').write_text('invalid edit')
                else:assert read_json(workspace/'proposal.json')['hypothesis']=='original'
                desc=read_json(attempt/'report.json')['description']
                return {'id':index,'status':'Killed' if index==1 else 'Succeeded','stopReason':'PREEMPTED',
                    'description':desc,'project':'rsi','cluster':'beijing','resourcePoolId':523 if index==1 else 525,
                    'priority':'PREEMPTABLE' if index==1 else 'HIGH','resourcesTotal':{'gpuCount':1}}
            raise AssertionError(args)
    class Backend(GpuMetaBackend):
        def select_pool(self,**kwargs):return ('model-new' if not creates else 'model-autosft'),{}
        def submit(self,attempt,report):
            creates.append(attempt)
            report.update(task_id=len(creates),description='own-meta-'+str(len(creates)),status='queued',next_poll_at=clock[0]+600)
            atomic_json(attempt/'report.json',report)
    def sleep(seconds):clock[0]+=seconds
    backend=Backend(site,memory,transport=Platform(),clock=lambda:clock[0],sleep=sleep)
    answer=backend.turn(workspace,'research')
    assert answer['summary']=='VALID' and len(creates)==2
    assert read_json(creates[0]/'report.json')['status']=='invalid'
    assert (creates[0]/'discarded-workspace/proposal.json').read_text()=='invalid edit'
    assert meta_overview(memory.root,site)['current']['status']=='completed'
    assert [read_json(a/'report.json')['priority'] for a in creates]==['PREEMPTABLE','HIGH']
    # Restart must return the successful result without another submission.
    assert backend.turn(workspace,'research')==answer and len(creates)==2


def test_queued_task_only_moves_after_identity_check(tmp_path):
    site,memory,workspace=fixture(tmp_path);clock=[0.];created=[];stopped=[];gets=[]
    class Platform:
        def command(self,args):
            if args[:2]==['job','stop']:
                assert args[3:]==['--confirm','tasks/'+args[2]]
                stopped.append(args[2]);return {}
            if args[:2]==['job','get']:
                i=int(args[2]);a=created[i-1];gets.append((i,clock[0]));report=read_json(a/'report.json')
                state='Queued' if i==1 and not stopped else 'Killed' if i==1 else 'Succeeded'
                if state=='Succeeded':atomic_json(a/'result.json',{'request_id':report['request_id'],'answer':{'status':'needs_input','summary':'ok','proposal_path':None}})
                return {'id':i,'status':state,'description':report['description'],'project':'rsi','cluster':'beijing',
                    'resourcePoolId':523 if i==1 else 525,'priority':report['priority'],'resourcesTotal':{'gpuCount':1}}
            raise AssertionError(args)
    class Backend(GpuMetaBackend):
        def select_pool(self,**kwargs):
            return ('model-new' if not created else 'model-autosft'),{'model-new':{'free_gpus':0},'model-autosft':{'free_gpus':1}}
        def submit(self,a,r):
            created.append(a);r.update(task_id=len(created),description='mine',next_poll_at=clock[0]+600)
            atomic_json(a/'report.json',r)
    backend=Backend(site,memory,transport=Platform(),clock=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    backend.turn(workspace,'research')
    assert stopped==['1'] and gets[:2]==[(1,600.),(1,1200.)]


def test_preemptable_submission_is_one_gpu_and_intent_survives_uncertainty(tmp_path):
    from llmgo.remote.cctl import SubmissionUncertain
    site,memory,workspace=fixture(tmp_path);attempt=tmp_path/'attempt';attempt.mkdir()
    atomic_json(attempt/'worker-request.json',{'source':{'root':'/frozen'}})
    calls=[]
    class Platform:
        def command(self,args):
            calls.append(args)
            if '--dry-run' in args:return {'request':{'task':{'resource_pool_id':523,'priority':3,'resources':{'gpu_count':1}}}}
            raise OSError('response lost')
    backend=GpuMetaBackend(site,memory,transport=Platform())
    report={'pool':'model-new','request_id':'request','status':'prepared'}
    with pytest.raises(SubmissionUncertain):backend.submit(attempt,report)
    args=calls[-1];assert args[args.index('--priority')+1]=='PREEMPTABLE' and args[args.index('--gpu')+1]=='1'
    assert read_json(attempt/'submission.json')['status']=='SUBMITTING'


def test_unknown_telemetry_uses_preemptable_without_guessing_high(tmp_path):
    site,memory,workspace=fixture(tmp_path)
    class Platform:
        def command(self,args):raise OSError('unavailable')
    pool,snapshot=GpuMetaBackend(site,memory,transport=Platform()).select_pool()
    assert pool=='model-new' and snapshot['model-new']['warning'] and not snapshot['high_admission']['available']


@pytest.mark.parametrize('current,target,move',[(0,0,False),(0,1,True),(1,1,False),(1,2,True),(None,2,False)])
def test_queued_tasks_keep_position_without_more_available_cards(current,target,move):
    from llmgo.rsi.meta_jobs import better_free_capacity
    assert better_free_capacity('model-new','model-claw-tool',{
        'model-new':{'free_gpus':current},'model-claw-tool':{'free_gpus':target}}) is move


def test_api_and_legacy_share_projection_without_gpu(tmp_path,monkeypatch):
    site,memory,workspace=fixture(tmp_path);site.update(meta_execution='api',meta_model='deepseek-v4.1-flash')
    assert isinstance(meta_backend(site,memory),ApiMetaBackend)
    calls=[]
    def turn(self,w,p):
        calls.append((w,p));self.session={'status':'running','token_usage':{'total':{'outputTokens':12}}}
        if self.status_mirror:atomic_json(self.status_mirror,self.session)
        return {'status':'needs_input','summary':'fixture','proposal_path':None}
    monkeypatch.setattr('llmgo.rsi.meta.CodexBackend.turn',turn)
    answer=meta_backend(site,memory).turn(workspace,'request')
    data=meta_overview(memory.root,site)
    assert answer['summary']=='fixture' and len(calls)==1
    assert data['current']['execution']=='api' and data['current']['status']=='completed'
    assert data['meta_gpu_hours']==0
    atomic_json(tmp_path/'legacy/meta-session.json',{'status':'paused'})
    assert meta_overview(tmp_path/'legacy',site)['current']['legacy']


def test_gpu_worker_uses_same_codex_runner_and_releases_service(tmp_path,monkeypatch):
    from llmgo.rsi import meta_gpu_worker as worker
    from llmgo.rsi.artifacts import source_snapshot
    site,memory,workspace=fixture(tmp_path)
    site.update(meta_model_path='/model',inference_python='fixture',meta_context_length=65536)
    attempt=tmp_path/'attempt';attempt.mkdir()
    source=source_snapshot(Path(site['repo']),attempt/'source')
    request={'site':site,'attempt_root':str(attempt),'campaign_root':str(memory.root),'request_id':'key',
        'source':source,'workspace':str(workspace),'prompt':'research'}
    atomic_json(attempt/'worker-request.json',request)
    instances=[]
    class Server:
        def __init__(self,model,protocol,log,port,gpu,**kwargs):
            self.options=kwargs;self.base_url='http://127.0.0.1:8100/v1';self.stopped=False;instances.append(self)
        def start(self):pass
        def stop(self):self.stopped=True
    class Backend:
        def __init__(self,s,m):
            assert s['meta_thinking'] is True
            assert 'deepseek' not in str(s['codex_home'])
            assert Path(s['meta_bridge_root'])==attempt/'bridge'
        def turn(self,w,p):
            assert w==workspace and 'GPU研究尝试' in p
            return {'status':'needs_input','summary':'ok','proposal_path':None}
    monkeypatch.setattr(worker,'visible_gpu_ids',lambda:['0'])
    monkeypatch.setattr(worker,'_pick_ports',lambda *args:[8100])
    monkeypatch.setattr(worker,'_wait_servers',lambda *args:None)
    # Do not let this unit test alter the real process environment/cache roots.
    monkeypatch.setattr(worker,'worker_environment',lambda s:{})
    worker.run(attempt/'worker-request.json',server_class=Server,backend_class=Backend)
    assert instances[0].stopped and instances[0].options['max_num_seqs']==1
    assert instances[0].options['tool_call_parser']=='qwen3_coder'
    assert read_json(attempt/'worker.json')['stage']=='completed'
    assert read_json(attempt/'result.json')['answer']['summary']=='ok'


def test_clean_start_excludes_teacher_history_and_routes_outputs(tmp_path):
    from llmgo.rsi.clean_start import prepare,DOCS
    repo=tmp_path/'repo';repo.mkdir()
    for folder in ('src','scripts','tests','.agents','configs/grpo','assets','var/runs','docs'):(repo/folder).mkdir(parents=True,exist_ok=True)
    (repo/'src/code.py').write_text('print(1)')
    (repo/'var/runs/old-results.json').write_text('SECRET_RESEARCH_RESULT')
    (repo/'docs/grpo-maintenance.md').write_text('OLD_RESEARCH_CONCLUSION')
    for name in DOCS:(repo/'docs'/name).write_text('generic contract '+name)
    atomic_json(repo/'configs/grpo/default.json',{})
    site=tmp_path/'site.json';atomic_json(site,{'repo':str(repo)})
    target=tmp_path/'clean';result=prepare(repo,target,site)
    assert result.exists() and not (target/'docs/grpo-maintenance.md').exists()
    assert (target/'docs/grpo-pipeline.md').read_text()=='generic contract clean-grpo-pipeline.md'
    assert (target/'docs/rsi-operations.md').read_text()=='generic contract meta-research-operations.md'
    assert (target/'AGENTS.md').read_text()=='generic contract meta-workspace-agents.md'
    assert not (target/'.git').exists() and (target/'var/runs').resolve()==repo/'var/runs'
    manifest=read_json(target/'clean-start.json')
    assert not any('old-results' in n or 'maintenance' in n for n in manifest['source_files'])
    with pytest.raises(FileExistsError):prepare(repo,target,site)


def test_high_selection_excludes_own_training_and_keeps_other_pools(tmp_path):
    site,memory,workspace=fixture(tmp_path)
    class Platform:
        def __init__(self):self.site=site
        def command(self,args):
            if args[:2]==['task','list']:return {'tasks':[{'id':9,'priority':'HIGH','status':'Running'}]}
            if args[0]=='pool':
                free={'model-autosft':10,'model-new':2,'model-claw-tool':4,'model-codeagent2':1}[args[2]]
                return {'edges':{'resource_quotas':[{'name':'h800','capacity':free,'used':0,'edges':{'cluster':{'name':'beijing'}}}]}}
            return {'tasks':[]}
    pool,selection=GpuMetaBackend(site,memory,transport=Platform()).select_pool()
    assert pool=='model-claw-tool' and not selection['high_admission']['available']


@pytest.mark.parametrize('pool,pool_id',[('model-autosft',525),('model-new',523),('model-claw-tool',515),('model-codeagent2',514)])
def test_each_pool_submits_exact_priority_and_one_gpu(tmp_path,pool,pool_id):
    from llmgo.rsi.meta_jobs import PRIORITIES
    site,memory,workspace=fixture(tmp_path);attempt=tmp_path/'attempt';attempt.mkdir()
    atomic_json(attempt/'worker-request.json',{'source':{'root':'/frozen'}})
    calls=[]
    class Platform:
        def __init__(self):self.site=site
        def command(self,args):
            calls.append(args)
            if args[:2]==['task','list']:return {'tasks':[]}
            if '--dry-run' in args:return {'request':{'task':{'resource_pool_id':pool_id,'priority':1 if pool_id==525 else 3,'resources':{'gpu_count':1}}}}
            return {'id':7,'status':'Queued'}
    GpuMetaBackend(site,memory,transport=Platform()).submit(attempt,{'pool':pool,'priority':PRIORITIES[pool],'request_id':'fixture'})
    args=calls[-1]
    assert args[args.index('--gpu')+1]=='1' and args[args.index('--priority')+1]==PRIORITIES[pool]
    if pool=='model-autosft':assert str(attempt/'submission.json') in read_json(Path(site['repo'])/'var/runtime/high-admission/reservations.json')


def test_high_race_falls_back_without_duplicate_create(tmp_path):
    site,memory,workspace=fixture(tmp_path);attempt=tmp_path/'attempt';attempt.mkdir()
    atomic_json(attempt/'worker-request.json',{'source':{'root':'/frozen'}});creates=[]
    class Platform:
        def __init__(self):self.site=site
        def command(self,args):
            if args[:2]==['task','list']:return {'tasks':[{'id':8,'priority':'HIGH','status':'Running'}]}
            if '--dry-run' in args:
                auto=args[args.index('--resource-pool')+1]=='model-autosft'
                return {'request':{'task':{'resource_pool_id':525 if auto else 514,'priority':1 if auto else 3,'resources':{'gpu_count':1}}}}
            creates.append(args);return {'id':7,'status':'Queued'}
    class Backend(GpuMetaBackend):
        def select_pool(self,**kwargs):
            assert 'model-autosft' in kwargs['exclude'];return 'model-codeagent2',{}
    Backend(site,memory,transport=Platform()).submit(attempt,{'pool':'model-autosft','priority':'HIGH','request_id':'fixture'})
    assert len(creates)==1 and read_json(attempt/'submission.json')['resource_pool']=='model-codeagent2'


def test_meta_uncertain_high_reserves_training_slot_and_own_queued_is_not_competitor(tmp_path):
    from llmgo.remote.high_admission import high_available,high_slot,AdmissionBusy
    from llmgo.remote.cctl import SubmissionUncertain
    site,memory,workspace=fixture(tmp_path);attempt=tmp_path/'attempt';attempt.mkdir()
    atomic_json(attempt/'worker-request.json',{'source':{'root':'/frozen'}})
    class Platform:
        def __init__(self):self.site=site;self.rows=[]
        def command(self,args):
            if args[:2]==['task','list']:return {'tasks':self.rows}
            if args[:2]==['job','get']:return {'status':'Queued'}
            if '--dry-run' in args:return {'request':{'task':{'resource_pool_id':525,'priority':1,'resources':{'gpu_count':1}}}}
            raise OSError('unknown create outcome')
    transport=Platform()
    with pytest.raises(SubmissionUncertain):
        GpuMetaBackend(site,memory,transport=transport).submit(attempt,{'pool':'model-autosft','priority':'HIGH','request_id':'fixture'})
    with pytest.raises(AdmissionBusy):
        with high_slot(transport,tmp_path/'training/submission.json'):raise AssertionError('must not create')
    intent=attempt/'submission.json';data=read_json(intent);data['task_id']=7;atomic_json(intent,data)
    transport.rows=[{'name':'tasks/7','priority':'HIGH','status':'Queued'}]
    assert high_available(transport,intent)==(True,None)
    assert high_available(transport)[0] is False
