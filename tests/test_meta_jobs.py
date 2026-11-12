import json
from pathlib import Path

import pytest

from llmgo.rsi.contracts import atomic_json,read_json
from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.meta_jobs import GpuMetaBackend,ApiMetaBackend,choose_pool,pool_snapshot,meta_backend,meta_overview


@pytest.mark.parametrize('af,aq,nf,nq,want',[(1,8,8,0,'model-autosft'),(0,4,8,0,'model-autosft'),
    (0,5,1,5,'model-new'),(0,5,0,4,'model-new'),(0,5,0,5,'model-autosft'),(0,8,0,8,'model-autosft')])
def test_pool_thresholds(af,aq,nf,nq,want):
    assert choose_pool({'free_gpus':af,'queued_people':aq},{'free_gpus':nf,'queued_people':nq})==want


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
                    'priority':'PREEMPTABLE' if index==1 else 'NORMAL','resourcesTotal':{'gpuCount':1}}
            raise AssertionError(args)
    class Backend(GpuMetaBackend):
        def select_pool(self):return ('model-new' if not creates else 'model-autosft'),{}
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
    assert all(read_json(a/'report.json')['priority']!='HIGH' for a in creates)
    # Restart must return the successful result without another submission.
    assert backend.turn(workspace,'research')==answer and len(creates)==2


def test_queued_task_only_moves_after_identity_check(tmp_path):
    site,memory,workspace=fixture(tmp_path);clock=[0.];created=[];stopped=[];gets=[]
    class Platform:
        def command(self,args):
            if args[:2]==['job','stop']:stopped.append(args[2]);return {}
            if args[:2]==['job','get']:
                i=int(args[2]);a=created[i-1];gets.append((i,clock[0]));report=read_json(a/'report.json')
                state='Queued' if i==1 and not stopped else 'Killed' if i==1 else 'Succeeded'
                if state=='Succeeded':atomic_json(a/'result.json',{'request_id':report['request_id'],'answer':{'status':'needs_input','summary':'ok','proposal_path':None}})
                return {'id':i,'status':state,'description':report['description'],'project':'rsi','cluster':'beijing',
                    'resourcePoolId':523 if i==1 else 525,'priority':report['priority'],'resourcesTotal':{'gpuCount':1}}
            raise AssertionError(args)
    class Backend(GpuMetaBackend):
        def select_pool(self):return ('model-new' if not created else 'model-autosft'),{}
        def submit(self,a,r):
            created.append(a);r.update(task_id=len(created),description='mine',next_poll_at=clock[0]+600)
            atomic_json(a/'report.json',r)
    backend=Backend(site,memory,transport=Platform(),clock=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    backend.turn(workspace,'research')
    assert stopped==['1'] and gets[:2]==[(1,600.),(1,1200.)]


def test_meta_submission_is_one_gpu_never_high_and_intent_survives_uncertainty(tmp_path):
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


def test_unknown_telemetry_uses_normal_autosft(tmp_path):
    site,memory,workspace=fixture(tmp_path)
    class Platform:
        def command(self,args):raise OSError('unavailable')
    pool,snapshot=GpuMetaBackend(site,memory,transport=Platform()).select_pool()
    assert pool=='model-autosft' and snapshot['warning']


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
    from llmgo.rsi.clean_start import prepare
    repo=tmp_path/'repo';repo.mkdir()
    for folder in ('src','scripts','tests','.agents','configs/grpo','assets','var/runs','docs'):(repo/folder).mkdir(parents=True,exist_ok=True)
    (repo/'src/code.py').write_text('print(1)')
    (repo/'var/runs/old-results.json').write_text('SECRET_RESEARCH_RESULT')
    (repo/'docs/grpo-maintenance.md').write_text('OLD_RESEARCH_CONCLUSION')
    for name in ('lichess-data.md','training-reviews.md','meta-execution.md'):(repo/'docs'/name).write_text('generic contract')
    atomic_json(repo/'configs/grpo/default.json',{})
    site=tmp_path/'site.json';atomic_json(site,{'repo':str(repo)})
    target=tmp_path/'clean';result=prepare(repo,target,site)
    assert result.exists() and not (target/'docs/grpo-maintenance.md').exists()
    assert not (target/'.git').exists() and (target/'var/runs').resolve()==repo/'var/runs'
    manifest=read_json(target/'clean-start.json')
    assert not any('old-results' in n or 'maintenance' in n for n in manifest['source_files'])
    with pytest.raises(FileExistsError):prepare(repo,target,site)
