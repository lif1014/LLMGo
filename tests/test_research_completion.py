"""Durable, explicit completion; never infer readiness from a proposal draft."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from llmgo.rsi.contracts import atomic_json, digest, file_hash, read_json
from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.meta import CodexBackend
from llmgo.rsi.meta_protocol import ResearchHandshake, MetaFailure


@pytest.fixture
def research(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    atomic_json(workspace/'proposal.json',{'hypothesis':'Test actual RL evidence','jobs':[],'evaluation':False})
    memory=MemoryStore(tmp_path/'campaign')
    handshake=ResearchHandshake(memory.root,workspace,digest('first-request'))
    handshake.publish()
    return workspace,memory,handshake


def test_completion_helper_binds_request_proposal_and_all_source_files(research):
    workspace,memory,h=research
    # A completely written proposal draft must never authorize submission.
    assert h.read() is None
    command=read_json(h.request_path)['finish_command']
    done=subprocess.run([*command,'--summary','Verified the experiment'],capture_output=True,text=True,check=True)
    answer=json.loads(done.stdout)
    assert h.read()==answer and answer['proposal_path']==str(workspace/'proposal.json')
    assert not (workspace/'result.json').exists()
    # Unrelated research code changed after commit: require another explicit commit.
    (workspace/'train.py').write_text('changed = True\n')
    with pytest.raises(ValueError,match='changed after completion'):h.read()
    subprocess.run([*command,'--summary','Rechecked after edits'],capture_output=True,check=True)
    assert h.read()['summary']=='Rechecked after edits'


@pytest.mark.parametrize('mode',['wrong_request','wrong_workspace','proposal_changed','template','escape','duplicate'])
def test_stale_partial_or_conflicting_commits_cannot_authorize_a_draft(research,mode,tmp_path):
    workspace,memory,h=research
    answer={'status':'ready','summary':'Done','proposal_path':'proposal.json'}
    if mode=='template':
        atomic_json(workspace/'proposal.json',{'hypothesis':'Describe the next experiment'})
        with pytest.raises(ValueError,match='template'):h.commit(answer)
        return
    if mode=='escape':
        answer['proposal_path']=str(tmp_path/'outside.json')
        with pytest.raises(ValueError,match='outside'):h.commit(answer)
        return
    h.commit(answer);record=read_json(h.result_path)
    if mode=='wrong_request':record['request_id']=digest('second-request')
    elif mode=='wrong_workspace':record['workspace']=str(tmp_path/'different')
    elif mode=='proposal_changed':atomic_json(workspace/'proposal.json',{'hypothesis':'A different experiment'})
    elif mode=='duplicate':
        h.result_path.write_text('{"request_id":"first","request_id":"second"}')
        with pytest.raises(ValueError,match='Duplicate'):h.read()
        return
    if mode!='proposal_changed':atomic_json(h.result_path,record)
    with pytest.raises(ValueError):h.read()


def serve(tmp_path, script):
    path=tmp_path/'server.py';path.write_text(script)
    def popen(argv,**kwargs):return subprocess.Popen([sys.executable,str(path)],**kwargs)
    return popen


PROGRESS_SERVER=r'''
import json,sys,subprocess
from pathlib import Path
count=0;threads=0
def emit(x):print(json.dumps(x),flush=True)
for line in sys.stdin:
 r=json.loads(line);method=r.get('method')
 if 'id' not in r:continue
 if method=='initialize':emit({'id':r['id'],'result':{}})
 elif method in ('thread/start','thread/resume'):
  threads+=1;emit({'id':r['id'],'result':{'thread':{'id':'thread-'+str(threads),'turns':[]}}})
 elif method=='turn/start':
  count+=1;t='turn-'+str(count)
  emit({'id':r['id'],'result':{'turn':{'id':t}}})
  # BODY
  emit({'method':'item/completed','params':{'turnId':t,'item':{'type':'agentMessage','text':text}}})
  emit({'method':'turn/completed','params':{'turn':{'id':t,'status':'completed','items':[]}}})
'''


def backend(tmp_path,workspace,memory,body):
    source=PROGRESS_SERVER.replace('  # BODY','\n'.join('  '+line for line in body.splitlines()))
    return CodexBackend({'user_root':str(tmp_path),'codex_home':str(tmp_path/'home'),'codex_bin':'fake'},
                        memory,popen=serve(tmp_path,source))


def test_productive_research_can_span_many_completions_without_a_research_cap(research,tmp_path):
    workspace,memory,_=research
    body="""if count<5:
 emit({'method':'item/completed','params':{'turnId':t,'item':{'type':'commandExecution','exitCode':0}}})
 text='Investigated one more issue; continuing research.'
else:text=json.dumps({'status':'ready','summary':'Five productive turns completed','proposal_path':'proposal.json'})"""
    b=backend(tmp_path,workspace,memory,body)
    assert b.turn(workspace,'Research')['status']=='ready'
    assert len(memory.recent(20,'meta_continuation'))==4
    assert not memory.recent(20,'meta_protocol_recovery')


def test_completion_file_survives_prose_final_reply_and_crash_before_prepare(research,tmp_path):
    workspace,memory,_=research
    body=f"""request=next(Path({str(memory.root)!r}).glob('meta-requests/*/request.json'))
cmd=json.loads(request.read_text())['finish_command']
subprocess.run(cmd+['--summary','Verified and explicitly finished'],check=True,capture_output=True)
emit({{'method':'item/completed','params':{{'turnId':t,'item':{{'type':'commandExecution','exitCode':0}}}}}})
text='Completed. The saved completion file is the handoff.'"""
    # This fixture has a prior, unrelated request descriptor as well.
    body=body.replace("request=next(","request=max(").replace(".glob('meta-requests/*/request.json'))", ".glob('meta-requests/*/request.json'),key=lambda p:p.stat().st_mtime_ns)")
    b=backend(tmp_path,workspace,memory,body)
    first=b.turn(workspace,'Research');assert first['status']=='ready'
    assert b.session['result_source']=='file'
    def forbidden(*args,**kwargs):raise AssertionError('A durable completion must not invoke the model again')
    restored=CodexBackend(b.site,memory,popen=forbidden)
    assert restored.turn(workspace,'Research')==first


def test_commit_survives_crash_before_backend_saves_any_final_answer(research,tmp_path):
    workspace,memory,_=research
    request_id=digest({'workspace':str(workspace),'prompt':'Research','context':None})
    h=ResearchHandshake(memory.root,workspace,request_id);h.publish()
    committed=h.commit({'status':'ready','summary':'Committed before pipe closed','proposal_path':'proposal.json'})
    atomic_json(memory.root/'meta-session.json',{'status':'running','process':None,
        'request_key':request_id,'thread_id':'old','turn_id':'old-turn','answer':None})
    def forbidden(*args,**kwargs):raise AssertionError('An explicit valid commit should recover offline')
    b=CodexBackend({'user_root':str(tmp_path)},memory,popen=forbidden)
    assert b.turn(workspace,'Research')==committed
    assert b.session['result_source']=='file'


def test_legacy_failed_format_retry_is_migrated_and_continued(research,tmp_path):
    workspace,memory,_=research
    atomic_json(memory.root/'meta-session.json',{'status':'failed','process':None,
        'request_key':digest({'workspace':str(workspace),'prompt':'Research','context':None}),
        'thread_id':'old','turn_id':'old-turn','review':{'acted':True,'format_retry':True}})
    body="text=json.dumps({'status':'ready','summary':'Recovered old work','proposal_path':'proposal.json'})"
    b=backend(tmp_path,workspace,memory,body)
    assert b.turn(workspace,'Research')['status']=='ready'
    assert 'format_retry' not in b.session['review']


def test_only_final_message_can_authorize_ready(research,tmp_path):
    workspace,memory,_=research
    body="""if count==1:
 emit({'method':'item/completed','params':{'turnId':t,'item':{'type':'agentMessage','text':json.dumps({'status':'ready','summary':'Earlier example','proposal_path':'wrong.json'})}}})
 text='That was only an example; I still need to finish.'
else:text=json.dumps({'status':'ready','summary':'Actually complete','proposal_path':'proposal.json'})"""
    assert backend(tmp_path,workspace,memory,body).turn(workspace,'Research')['summary']=='Actually complete'


def test_persistent_empty_completions_use_bounded_fresh_sessions_then_retryable_error(research,tmp_path):
    workspace,memory,_=research
    b=backend(tmp_path,workspace,memory,"text='Let me continue.'")
    with pytest.raises(MetaFailure,match='without an explicit') as caught:b.turn(workspace,'Research')
    assert caught.value.retryable and caught.value.code=='completion_protocol'
    assert len(memory.recent(10,'meta_protocol_recovery'))==2
    assert b.session['process'] is None
    assert (workspace/'proposal.json').exists()
