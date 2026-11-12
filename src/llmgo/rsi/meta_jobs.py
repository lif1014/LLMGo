"""Common Meta request records and on-demand GPU execution with shared HIGH admission."""
from __future__ import annotations

from datetime import datetime,timezone
from contextlib import nullcontext
import json
from pathlib import Path
import shlex
import shutil
import time

from .artifacts import source_snapshot
from .contracts import atomic_json,digest,file_hash,inside,read_json,utc
from .memory import MemoryStore
from .meta_protocol import MetaFailure
from .research_control import require_research,ResearchPaused
from llmgo.remote.cctl import CctlExecutor,SubmissionUncertain,_task_id
from llmgo.remote.high_admission import high_available,high_slot,AdmissionBusy

QUEUED={'Creating','Created','Pending','Scheduling','Queued','Requeue'}
TERMINAL={'Succeeded','Failed','Killed','Stopped','OOMKilled','ImagePullBackOff','Invalid'}
POOL_IDS={'model-autosft':525,'model-new':523,'model-claw-tool':515,'model-codeagent2':514}
PRIORITIES={pool:'HIGH' if pool=='model-autosft' else 'PREEMPTABLE' for pool in POOL_IDS}


def meta_mode(site):
    return site.get('meta_execution','api')


def request_key(workspace,prompt):
    context=Path(workspace)/'context.json'
    return digest({'workspace':str(workspace),'prompt':prompt,'context':file_hash(context) if context.exists() else None})


def seconds_between(start,end=None):
    if not start:return 0.
    def stamp(v):return datetime.fromisoformat(v.replace('Z','+00:00')).timestamp()
    return max(0.,(stamp(end) if end else time.time())-stamp(start))


def pool_snapshot(transport,pool):
    """Quota headroom plus distinct visible H800 queue submitters; never list their commands."""
    detail=transport.command(['pool','get',pool])
    quotas=[q for q in detail.get('edges',{}).get('resource_quotas',[])
        if q.get('name','').lower()=='h800' and q.get('edges',{}).get('cluster',{}).get('name')=='beijing']
    if not quotas:raise ValueError('No beijing H800 quota in '+pool)
    free=sum(max(0,q['capacity']-q['used']) for q in quotas)
    people=set();queued=0
    for page in range(1,101):
        data=transport.command(['job','list','--cluster','beijing','--pool',pool,
            '--status',','.join(sorted(QUEUED)),'--page',str(page),'--limit','100'])
        rows=data if isinstance(data,list) else data.get('tasks',[])
        for row in rows:
            resources=row.get('resourcesTotal') or row.get('resourcesPerNode') or {}
            if not resources.get('gpuCount') or str(row.get('gpuModel','h800')).lower()!='h800':continue
            queued+=1
            creator=row.get('creator')
            if isinstance(creator,dict):creator=creator.get('id') or creator.get('name')
            if creator:people.add(str(creator))
        if len(rows)<100:break
    else:raise ValueError('Queue listing incomplete')
    return {'pool':pool,'free_gpus':free,'queued_people':len(people),'queued_tasks':queued,
        'free_scope':'H800 quota capacity minus used, not a placement guarantee','observed':utc()}


def choose_pool(snapshots,*,exclude=()):
    """Prefer available H800s, then the shorter queue; fixed order breaks exact ties."""
    pools=[p for p in POOL_IDS if p not in exclude and 'free_gpus' in snapshots.get(p,{})]
    if not pools:
        # Authorized preemptable fallback; unknown headroom is never reported as zero.
        return next(p for p in PRIORITIES if p!='model-autosft' and p not in exclude)
    free=[p for p in pools if snapshots[p]['free_gpus']>=1]
    order=list(POOL_IDS)
    if free:
        return min(free,key=lambda p:(-snapshots[p]['free_gpus'],snapshots[p]['queued_people'],
            snapshots[p]['queued_tasks'],order.index(p)))
    return min(pools,key=lambda p:(snapshots[p]['queued_people'],snapshots[p]['queued_tasks'],order.index(p)))


def better_free_capacity(current,target,snapshots):
    """Do not lose queue position by cycling through equally full pools."""
    before=snapshots.get(current,{}).get('free_gpus')
    after=snapshots.get(target,{}).get('free_gpus')
    return before is not None and after is not None and after>=1 and after>before


def request_record(root,workspace,prompt,site):
    key=request_key(workspace,prompt);folder=Path(root)/'meta_requests'/key
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'request.json'
    if not path.exists():
        context=Path(workspace)/'context.json'
        kind='review' if context.exists() and read_json(context).get('training_review') else 'research'
        atomic_json(path,{'schema_version':1,'request_id':key,'workspace':str(workspace),
            'prompt':prompt,'kind':kind,'model':site.get('meta_model'),'thinking':site.get('meta_thinking'),
            'execution':meta_mode(site),'created':utc(),'site':site})
    return folder,read_json(path)


def meta_overview(root,site):
    """Read-only, common Web projection. No scheduler/model calls or Memory rebuilds."""
    root=Path(root);rows=[]
    for request in sorted((root/'meta_requests').glob('*/request.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]:
        data=read_json(request);folder=request.parent
        attempts=[]
        for path in sorted((folder/'attempts').glob('*/report.json')):
            a=read_json(path);session=path.parent/'session.json'
            if session.exists():
                s=read_json(session)
                a['activity']=s.get('activity') or s.get('status');a['token_usage']=s.get('token_usage')
                a['last_activity']=s.get('updated');a['thread_id']=s.get('thread_id')
            worker=path.parent/'worker.json'
            if worker.exists():
                w=read_json(worker)
                a['worker_stage']=w.get('stage');a['started']=w.get('started')
                a['finished']=w.get('finished');a['worker_error']=w.get('error')
                a['gpu_hours']=seconds_between(w.get('started'),a.get('ended') or w.get('finished'))/3600 if data['execution']=='gpu_job' else None
            a['artifact']=str(path.relative_to(root));attempts.append(a)
        row={k:data.get(k) for k in ['request_id','kind','model','thinking','execution','created']}
        row.update(workspace=data['workspace'],attempts=attempts,status=attempts[-1]['status'] if attempts else 'prepared')
        rows.append(row)
    if rows:current=rows[0]
    else:
        legacy=read_json(root/'meta-session.json') if (root/'meta-session.json').exists() else {}
        current={'model':site.get('meta_model'),'thinking':site.get('meta_thinking'),
            'execution':meta_mode(site),'status':legacy.get('status','idle'),'attempts':[],
            'legacy':True,'token_usage':legacy.get('token_usage')}
    return {'schema_version':1,'current':current,'requests':rows,
        'meta_gpu_hours':sum(a.get('gpu_hours') or 0 for r in rows for a in r['attempts']),
        'cost_scope':'Last 30 Meta requests; worker lifetime including model startup and tool execution; excludes queue. Invalid attempts retained as cost, never adopted as results.'}


def publish_meta_overview(root,site):
    atomic_json(Path(root)/"meta-overview.json",meta_overview(root,site))


class ApiMetaBackend:
    def __init__(self,site,memory):self.site,self.memory=site,memory
    def turn(self,workspace,prompt):
        from .meta import CodexBackend
        folder,request=request_record(self.memory.root,workspace,prompt,self.site)
        attempt=folder/'attempts/0001';attempt.mkdir(parents=True,exist_ok=True)
        report={'status':'running','execution':'api','model':self.site.get('meta_model'),'thinking':self.site.get('meta_thinking'),'started':utc()}
        atomic_json(attempt/'report.json',report)
        backend=CodexBackend(self.site,self.memory)
        backend.status_mirror=attempt/'session.json'
        backend.overview_publisher=lambda:publish_meta_overview(self.memory.root,self.site)
        try:
            answer=backend.turn(workspace,prompt)
            atomic_json(attempt/'result.json',{'request_id':request['request_id'],'answer':answer})
            report.update(status='completed',ended=utc());return answer
        except Exception as error:
            report.update(status='failed',error=str(error),ended=utc());raise
        finally:
            atomic_json(attempt/'report.json',report)
            publish_meta_overview(self.memory.root,self.site)


class GpuMetaBackend:
    def __init__(self,site,memory,*,transport=None,clock=time.time,sleep=time.sleep):
        self.site,self.memory,self.clock,self.sleep=site,memory,clock,sleep
        self.transport=transport or CctlExecutor(site,MemoryStore(memory.root/'executor'),record_commands=True)

    def turn(self,workspace,prompt):
        folder,request=request_record(self.memory.root,workspace,prompt,self.site)
        baseline=folder/'baseline'
        if not baseline.exists():
            source_snapshot(workspace,baseline)
            if (Path(workspace)/'context.json').exists():shutil.copy2(Path(workspace)/'context.json',baseline/'context.json')
        while True:
            require_research(self.memory.root)
            reports=sorted((folder/'attempts').glob('*/report.json'))
            if reports and read_json(reports[-1])['status']=='completed':
                return read_json(reports[-1].parent/'result.json')['answer']
            active=reports[-1].parent if reports and read_json(reports[-1])['status'] not in {'invalid','failed'} else None
            if active is None:
                active=folder/'attempts'/f'{len(reports)+1:04d}'
                active.mkdir(parents=True,exist_ok=True)
                pool,snapshot=self.select_pool()
                report={'status':'prepared','attempt':len(reports)+1,'pool':pool,'priority':PRIORITIES[pool],
                    'gpu_count':1,'selection':snapshot,'request_id':request['request_id'],'created':utc()}
                # Freeze executable code outside the editable research workspace.
                if not (active/'worker-request.json').exists():
                    source=(read_json(active/'source/source-manifest.json') if (active/'source/source-manifest.json').exists()
                        else source_snapshot(Path(self.site['repo']),active/'source'))
                    atomic_json(active/'worker-request.json',{**request,'campaign_root':str(self.memory.root),
                        'attempt_root':str(active),'source':source})
                atomic_json(active/'report.json',report)
            report=read_json(active/'report.json')
            if not report.get('task_id'):
                self.submit(active,report)
                report=read_json(active/'report.json')
            now=self.clock()
            if report.get('terminal_task') or now>=report.get('next_poll_at',0):
                task=report.get('terminal_task') or self.transport.command(['job','get',str(report['task_id'])])
                self.verify_task(task,report)
                status=task['status'];report.update(platform_status=status,next_poll_at=now+600,updated=utc())
                atomic_json(active/'report.json',report)
                if status in TERMINAL:
                    report['ended']=task.get('updateTime') or utc()
                    report['terminal_task']={k:task.get(k) for k in ('id','name','status','stopReason','updateTime','description','project','cluster','resourcePoolId','priority','resourcesTotal')}
                    atomic_json(active/'report.json',report)
                    if status=='Succeeded':
                        if not (active/'result.json').exists():
                            report['status']='waiting_for_artifact';atomic_json(active/'report.json',report)
                            publish_meta_overview(self.memory.root,self.site);self.sleep(5);continue
                        result=read_json(active/'result.json')
                        if result['request_id']!=request['request_id']:raise ValueError('Mismatched Meta result')
                        report['status']='completed';atomic_json(active/'report.json',report)
                        publish_meta_overview(self.memory.root,self.site)
                        self.memory.append('meta_job_completed',{'request_id':request['request_id'],'task_id':report['task_id']})
                        return result['answer']
                    report.update(status='invalid_pending_restore',reason=task.get('stopReason') or status)
                    atomic_json(active/'report.json',report)
                    self.memory.append('meta_job_invalid',{'request_id':request['request_id'],'task_id':report['task_id'],'reason':report['reason']})
                    self.restore_workspace(workspace,baseline,active)
                    report['status']='invalid';atomic_json(active/'report.json',report)
                    if status not in {'Killed','Stopped'} and report['reason'] not in {'PREEMPTED','USER_STOPPED'}:
                        raise MetaFailure('Meta GPU task '+status+'; see '+str(active),code='gpu_worker_failed',retryable=False)
                    continue
                report['status']='queued' if status in QUEUED else 'running'
                if status in QUEUED:
                    pool,snapshot=self.select_pool(current_intent=active/'submission.json')
                    if pool!=report['pool'] and better_free_capacity(report['pool'],pool,snapshot):
                        # Only this identity-checked, not-yet-running Meta allocation can move pools.
                        self.transport.command(['job','stop',str(report['task_id'])])
                        report.update(status='switching_pool',selection=snapshot)
                atomic_json(active/'report.json',report)
            publish_meta_overview(self.memory.root,self.site)
            self.sleep(5)

    def select_pool(self,*,exclude=(),current_intent=None):
        snapshots={}
        for pool in PRIORITIES:
            try:snapshots[pool]=pool_snapshot(self.transport,pool)
            except Exception as error:snapshots[pool]={'warning':str(error)}
        blocked=set(exclude)
        try:
            available,reason=high_available(self.transport,current_intent)
        except Exception as error:
            available,reason=False,str(error)
        snapshots['high_admission']={'available':available,'reason':reason}
        if not available:blocked.add('model-autosft')
        selected=choose_pool(snapshots,exclude=blocked)
        return selected,snapshots

    def submit(self,attempt,report):
        request=read_json(attempt/'worker-request.json');pool=report['pool']
        priority=report.get('priority',PRIORITIES[pool])
        worker=Path(request['source']['root'])/'scripts/run_meta_gpu.py'
        argv=['job','create','--project',self.site['project'],'--cluster','beijing','--resource-pool',pool,
            '--billing-account-id',self.site['billing_account_id'],'--priority',priority,
            '--image',self.site['image'],'--gpu','1','--gpu-model','h800',
            '--cpu',str(self.site.get('meta_cpu',8)),'--memory',str(self.site.get('meta_memory_gib',64)),
            '--description',f"llmgo-meta:{self.memory.root.name}:{report['request_id']}:{attempt.name}",
            '--entry',shlex.join([self.site['control_python'],'-B',str(worker),'--request',str(attempt/'worker-request.json')])]
        intent=attempt/'submission.json'
        if intent.exists():
            old=read_json(intent)
            if old.get('task_id'):created=old
            else:
                transport=CctlExecutor({**self.site,'resource_pool':pool,'priority':priority},self.transport.memory,runner=self.transport.runner)
                created=transport.reconcile(intent)
        else:
            preview=self.transport.command([*argv,'--dry-run'])
            mapped=preview['request']['task']
            if mapped['resource_pool_id']!=POOL_IDS[pool] or mapped['priority']!={'HIGH':1,'NORMAL':2,'PREEMPTABLE':3}[priority] or mapped['resources']['gpu_count']!=1:
                raise ValueError('Meta submission resources differ from request')
            atomic_json(attempt/'dryrun.json',preview)
            description=argv[argv.index('--description')+1]
            try:
                with high_slot(self.transport,intent) if priority=='HIGH' else nullcontext():
                    atomic_json(intent,{'status':'SUBMITTING','description':description,'resource_pool':pool,'priority':priority,'created':utc()})
                    try:task=self.transport.command(argv)
                    except Exception as error:raise SubmissionUncertain('Meta submission uncertain; reconcile existing intent') from error
                    created={**read_json(intent),'task_id':_task_id(task),'status':task['status']}
                    atomic_json(intent,created)
            except AdmissionBusy:
                # Another controller may have taken HIGH since selection. No create has run.
                pool,snapshot=self.select_pool(exclude={'model-autosft'})
                report.update(pool=pool,priority=PRIORITIES[pool],selection=snapshot)
                atomic_json(attempt/'report.json',report)
                return self.submit(attempt,report)
        report.update(task_id=created['task_id'],description=created['description'],status='queued',submitted=utc(),next_poll_at=self.clock()+600)
        atomic_json(attempt/'report.json',report)
        self.memory.append('meta_job_submitted',{'request_id':report['request_id'],'task_id':report['task_id'],'pool':pool,'priority':priority})

    def verify_task(self,task,report):
        if (_task_id(task)!=report['task_id'] or task.get('description')!=report['description']
            or task.get('resourcePoolId')!=POOL_IDS[report['pool']] or task.get('priority')!=report['priority']
            or task.get('project')!=self.site['project'] or task.get('cluster')!='beijing'
            or task.get('resourcesTotal',{}).get('gpuCount')!=1):
            raise ValueError('Meta GPU task identity changed')

    @staticmethod
    def restore_workspace(workspace,baseline,attempt):
        # Archive all edits from an invalid allocation, even a proposal written before the kill.
        workspace=Path(workspace)
        archive=attempt/'discarded-workspace'
        if archive.exists():
            if workspace.exists():shutil.rmtree(workspace)
        elif workspace.exists():workspace.rename(archive)
        shutil.copytree(baseline,workspace,ignore=shutil.ignore_patterns('source-manifest.json'))


def meta_backend(site,memory):
    if meta_mode(site)=='api':return ApiMetaBackend(site,memory)
    if meta_mode(site)=='gpu_job':return GpuMetaBackend(site,memory)
    raise ValueError('Unknown meta_execution')
