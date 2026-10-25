"""Exercise the real JSONL subprocess transport with an isolated fake server."""
import json
import subprocess
import sys

import pytest

from llmgo.rsi.contracts import atomic_json, read_json
from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.meta import CodexBackend


SERVER = r'''
import json, sys
expected_mode = "workspace-write" if "inner" in sys.argv else "danger-full-access"
def emit(obj):
    print(json.dumps(obj), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" not in request:
        continue
    rid = request["id"]
    if method == "initialize":
        emit({"id":rid,"result":{}})
    elif method in ("thread/start", "thread/resume"):
        if request["params"].get("sandbox") != expected_mode:
            emit({"id":rid,"error":{"code":-32600,"message":"Invalid SandboxMode"}})
            continue
        emit({"id":rid,"result":{"thread":{"id":"thread-one", "turns":[]}}})
    elif method == "turn/start":
        policy=request["params"]["sandboxPolicy"]
        assert policy["type"] == ("workspaceWrite" if "inner" in sys.argv else "externalSandbox")
        assert policy["networkAccess"] == (True if "inner" in sys.argv else "enabled")
        emit({"id":rid,"result":{"turn":{"id":"turn-one"}}})
        emit({"method":"item/agentMessage/delta","params":{"turnId":"turn-one","delta":"visible analysis"}})
        emit({"method":"thread/tokenUsage/updated","params":{"threadId":"thread-one","tokenUsage":{"total":{"inputTokens":100,"outputTokens":20},"last":{"totalTokens":120}}}})
        status = "failed" if request["params"]["input"][0]["text"] == "fail" else "completed"
        answer = json.dumps({"status":"ready","summary":"one hypothesis","proposal_path":"proposal.json"})
        emit({"method":"item/completed","params":{"turnId":"turn-one","item":{"id":"item-one","type":"agentMessage","text":answer}}})
        emit({"method":"turn/completed","params":{"turn":{"id":"turn-one","status":status,"items":[]}}})
'''


@pytest.mark.parametrize("isolation", ["external", "workspace-write"])
def test_codex_jsonl_resume_visible_events_and_crash_result_reuse(tmp_path, isolation):
    server = tmp_path/"fake-server.py"
    server.write_text(SERVER)
    workspace = tmp_path/"workspace"
    workspace.mkdir()
    atomic_json(workspace/"context.json", {"round": 1})
    memory = MemoryStore(tmp_path/"memory")
    site = {"user_root": str(tmp_path), "codex_home": str(tmp_path/"codex"), "codex_bin": "fake"}
    site["meta_sandbox_mode"] = isolation
    calls = []
    def popen(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"]["CODEX_HOME"] == site["codex_home"]
        return subprocess.Popen([sys.executable, str(server), *(["inner"] if isolation=="workspace-write" else [])], **kwargs)
    backend = CodexBackend(site, memory, popen=popen)
    first = backend.turn(workspace, "prepare")
    assert first["status"] == "ready"
    # Controller crash before prepare: consume the saved answer instead of another API turn.
    restored = CodexBackend(site, memory, popen=popen)
    assert restored.turn(workspace, "prepare") == first and len(calls) == 1
    # A changed context is a new turn in the same persisted thread.
    atomic_json(workspace/"context.json", {"round": 1, "invalid_proposal": "retry"})
    assert restored.turn(workspace, "prepare") == first and len(calls) == 2
    sends = [e["body"]["message"] for e in memory.recent(100, "meta_send")]
    assert any(e.get("method") == "thread/resume" for e in sends)
    assert memory.search("visible analysis")
    assert read_json(memory.root/"meta-session.json")["process"] is None
    assert read_json(memory.root/"meta-session.json")["sandbox_mode"] == isolation
    assert read_json(memory.root/"meta-session.json")["token_usage"]["total"]["inputTokens"] == 100
    with pytest.raises(RuntimeError, match="turn failed"):
        restored.turn(workspace, "fail")
    assert read_json(memory.root/"meta-session.json")["status"] == "failed"
    assert memory.recent(1, "meta_failure")[0]["body"]["error_type"] == "RuntimeError"
    assert memory.recent(1, "meta_status")[0]["body"]["status"] == "failed"


def test_config_token_reaches_meta_child_without_leaking_to_journal(tmp_path):
    home=tmp_path/'codex-home';home.mkdir()
    (home/'config.toml').write_text('model_provider="custom"\n[model_providers.custom]\n'
        'env_key="OPENAI_API_KEY"\nexperimental_bearer_token="opaque-meta-test-secret"\n')
    server=tmp_path/'server.py'
    server.write_text('import os,sys\nprint(os.environ["OPENAI_API_KEY"],file=sys.stderr,flush=True)\n'+SERVER)
    workspace=tmp_path/'workspace';workspace.mkdir()
    memory=MemoryStore(tmp_path/'memory')
    site={'user_root':str(tmp_path),'codex_home':str(home),'codex_bin':'fake'}
    def popen(argv,**kwargs):
        assert kwargs['env']['OPENAI_API_KEY']=='opaque-meta-test-secret'
        return subprocess.Popen([sys.executable,str(server)],**kwargs)
    result=CodexBackend(site,memory,popen=popen).turn(workspace,'prepare')
    assert result['status']=='ready'
    assert 'opaque-meta-test-secret' not in memory.events.read_text()


def test_unknown_isolation_mode_fails_before_starting_codex(tmp_path):
    site={'user_root':str(tmp_path),'codex_home':str(tmp_path/'home'),
          'codex_bin':'fake','meta_sandbox_mode':'externla'}
    def forbidden(*args,**kwargs):
        raise AssertionError('Invalid isolation must not start a process')
    backend=CodexBackend(site,MemoryStore(tmp_path/'memory'),popen=forbidden)
    with pytest.raises(ValueError,match='meta_sandbox_mode'):
        backend.turn(tmp_path,'prepare')



@pytest.mark.parametrize("failed", [False, True], ids=["completed-turn", "failed-turn"])
def test_shutdown_preserves_trailing_protocol_usage_and_stderr(tmp_path, failed):
    # The child flushes final usage/diagnostics only after the parent's input closes.
    # These are real pipe bytes, deliberately arriving after turn/completed.
    tail = r'''
emit({"method":"thread/tokenUsage/updated","params":{"threadId":"thread-one","tokenUsage":{"total":{"inputTokens":123,"outputTokens":45},"last":{"totalTokens":168}}}})
emit({"id":999,"method":"late/serverRequest","params":{}})
print("shutdown diagnostic Bearer abcdef0123456789", file=sys.stderr, flush=True)
'''
    server = tmp_path/"trailing-server.py"
    server.write_text(SERVER+tail)
    workspace = tmp_path/"workspace"
    workspace.mkdir()
    memory = MemoryStore(tmp_path/"memory")
    site = {"user_root": str(tmp_path), "codex_home": str(tmp_path/"codex"), "codex_bin": "fake"}
    def popen(argv, **kwargs):
        return subprocess.Popen([sys.executable, str(server)], **kwargs)
    backend = CodexBackend(site, memory, popen=popen)
    if failed:
        with pytest.raises(RuntimeError, match="turn failed"):
            backend.turn(workspace, "fail")
    else:
        assert backend.turn(workspace, "prepare")["status"] == "ready"
    session = read_json(memory.root/"meta-session.json")
    assert session["token_usage"]["total"]["inputTokens"] == 123
    assert session["status"] == ("failed" if failed else "completed")
    assert session["process"] is None
    diagnostics = memory.recent(20, "meta_stderr")
    assert any("shutdown diagnostic" in e["body"]["message"]["text"] for e in diagnostics)
    assert "abcdef0123456789" not in memory.events.read_text()
    updates = [e for e in memory.recent(100, "meta_event")
               if e["body"]["message"].get("method") == "thread/tokenUsage/updated"]
    assert len(updates) == 2
    assert len({e["id"] for e in updates}) == 2
    assert len(memory.search("late/serverRequest")) == 1


@pytest.mark.parametrize('mode', ['progress', 'blocked', 'malformed'])
def test_premature_completion_is_clarified_once_in_same_thread(tmp_path, mode):
    server = tmp_path/'server.py'
    server.write_text(r'''
import json,sys
counter=0
def emit(v):print(json.dumps(v),flush=True)
for line in sys.stdin:
 r=json.loads(line);m=r.get('method')
 if 'id' not in r:continue
 if m=='initialize':emit({'id':r['id'],'result':{}})
 elif m in ('thread/start','thread/resume'):
  emit({'id':r['id'],'result':{'thread':{'id':'thread-test','turns':[]}}})
 elif m=='turn/start':
  assert 'outputSchema' not in r['params']
  counter+=1;t='t'+str(counter)
  emit({'id':r['id'],'result':{'turn':{'id':t}}})
  if counter==1:
   a={'status':'needs_input','summary':"I'll start reading.",'proposal_path':None}
   text='Let me begin' if sys.argv[1]=='malformed' else json.dumps(a)
  else:
   if sys.argv[1]=='blocked':a={'status':'needs_input','summary':'The authorized dataset is missing; please provide its path.','proposal_path':None}
   else:
    emit({'method':'item/completed','params':{'turnId':t,'item':{'type':'commandExecution','exitCode':0}}})
    a={'status':'ready','summary':'Analyzed actual data','proposal_path':'proposal.json'}
   text=json.dumps(a)
  emit({'method':'item/completed','params':{'turnId':t,'item':{'type':'agentMessage','text':text}}})
  emit({'method':'turn/completed','params':{'turn':{'id':t,'status':'completed','items':[]}}})
''')
    memory=MemoryStore(tmp_path/'memory')
    site={'user_root':str(tmp_path),'codex_home':str(tmp_path/'home'),'codex_bin':'fake'}
    calls=[]
    def popen(argv, **kwargs):
        calls.append(argv)
        return subprocess.Popen([sys.executable,str(server),mode],**kwargs)
    backend=CodexBackend(site,memory,popen=popen)
    answer=backend.turn(tmp_path,'Research now')
    assert answer['status']==('needs_input' if mode=='blocked' else 'ready')
    sends=[e['body']['message'] for e in memory.recent(100,'meta_send')]
    assert sum(e.get('method')=='turn/start' for e in sends)==2
    assert len(memory.recent(100,'meta_continuation'))==1
    assert backend.session['turn_id']=='t2'
    assert CodexBackend(site,memory,popen=popen).turn(tmp_path,'Research now')==answer
    assert len(calls)==1


def test_unreviewed_crash_recovery_does_not_accept_no_action_result(tmp_path):
    memory=MemoryStore(tmp_path/'memory')
    context=tmp_path/'context.json';atomic_json(context,{'round':1})
    from llmgo.rsi.contracts import digest,file_hash
    answer={'status':'needs_input','summary':"I'll start reading.",'proposal_path':None}
    previous={'id':'prior-turn','status':'completed','items':[{'type':'agentMessage','text':json.dumps(answer)}]}
    atomic_json(memory.root/'meta-session.json',{'thread_id':'thread-one','turn_id':'prior-turn',
        'request_key':digest({'workspace':str(tmp_path),'prompt':'prepare','context':file_hash(context)}),
        'status':'running','process':None})
    server=tmp_path/'server.py'
    server.write_text(SERVER.replace('"turns":[]', '"turns":'+repr([previous])))
    site={'user_root':str(tmp_path),'codex_home':str(tmp_path/'home'),'codex_bin':'fake'}
    def popen(argv,**kwargs):return subprocess.Popen([sys.executable,str(server)],**kwargs)
    backend=CodexBackend(site,memory,popen=popen)
    result=backend.turn(tmp_path,'prepare')
    assert result['status']=='ready'
    assert memory.recent(10,'meta_continuation')[0]['body']['reason']=='needs_input_without_actions'
    assert memory.recent(10,'meta_recovered')[0]['body']['review_pending']


@pytest.mark.parametrize('wrapper', ['{}', '```json\n{}\n```', 'Completed and verified.\n\n```JSON\n{}\n```\nDetails saved.'])
def test_final_result_accepts_json_fence_after_prose(wrapper):
    answer={'status':'ready','summary':'Verified experiment','proposal_path':'/owned/proposal.json'}
    assert CodexBackend._answer(wrapper.format(json.dumps(answer)))==answer


@pytest.mark.parametrize('text', [
    '{"status":"ready","summary":"done","proposal_path":null}',
    '{"status":"ready","summary":"done"}',
    '{"status":"ready","status":"needs_input","summary":"done","proposal_path":null}',
    '```json\n{"status":"ready","summary":"done","proposal_path":"p"}',
    'text {"status":"ready","summary":"done","proposal_path":"p"} text',
    '```json\n{"status":"ready","summary":"done","proposal_path":"p"}\n```\n```json\n{"status":"ready","summary":"different","proposal_path":"q"}\n```',
])
def test_result_parser_rejects_partial_ambiguous_and_invalid_contract(text):
    assert CodexBackend._answer(text) is None


def record_stopped_result(memory, workspace, text, *, completed=True):
    atomic_json(memory.root/'meta-session.json',{'status':'failed','process':None,'cwd':str(workspace),
        'thread_id':'saved-thread','turn_id':'saved-turn'})
    memory.append('meta_event',{'thread_id':'saved-thread','message':{'method':'item/completed','params':{
        'threadId':'saved-thread','turnId':'saved-turn','item':{'type':'agentMessage','text':text}}}})
    memory.append('meta_event',{'thread_id':'saved-thread','message':{'method':'turn/completed','params':{
        'threadId':'saved-thread','turn':{'id':'saved-turn','status':'completed' if completed else 'failed','items':[]}}}})


def test_recover_recorded_ready_without_spawning_or_model_call(tmp_path):
    memory=MemoryStore(tmp_path/'memory')
    answer={'status':'ready','summary':'Already done','proposal_path':str(tmp_path/'proposal.json')}
    record_stopped_result(memory,tmp_path,'Verified:\n```json\n'+json.dumps(answer)+'\n```')
    def forbidden(*args,**kwargs):raise AssertionError('Recovery must not start Codex')
    backend=CodexBackend({'user_root':str(tmp_path)},memory,popen=forbidden)
    assert backend.recover_completed(tmp_path)==answer
    assert backend.session['status']=='completed'
    assert memory.recent(1,'meta_result_recovered')[0]['body']['inference_performed'] is False


@pytest.mark.parametrize('mode',['failed_turn','wrong_workspace','live_process','needs_input','wrong_turn','invalid_final'])
def test_recovery_requires_matching_terminal_evidence(tmp_path,mode):
    memory=MemoryStore(tmp_path/'memory')
    answer={'status':'ready','summary':'Already done','proposal_path':str(tmp_path/'proposal.json')}
    if mode=='needs_input':answer.update(status='needs_input',proposal_path=None)
    text=json.dumps(answer)
    if mode=='invalid_final':text='I will start working.'
    record_stopped_result(memory,tmp_path,text,completed=mode!='failed_turn')
    session=read_json(memory.root/'meta-session.json')
    if mode=='live_process':session['process']={'pid':123}
    if mode=='wrong_turn':session['turn_id']='different-turn'
    atomic_json(memory.root/'meta-session.json',session)
    backend=CodexBackend({'user_root':str(tmp_path)},memory)
    with pytest.raises(ValueError):backend.recover_completed(tmp_path/'other' if mode=='wrong_workspace' else tmp_path)
    assert not memory.recent(1,'meta_result_recovered')
