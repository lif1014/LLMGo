"""One allocation owns Qwen thinking inference and the existing Codex tool runner."""
import argparse
import os
from pathlib import Path
import sys
import traceback

from .artifacts import verify_source
from .contracts import atomic_json,read_json,utc,worker_environment
from .memory import MemoryStore
from .meta import CodexBackend
from .service import VllmServer,_pick_ports,_wait_servers,visible_gpu_ids


def run(request_path,*,server_class=VllmServer,backend_class=CodexBackend):
    request=read_json(request_path);site=dict(request['site']);attempt=Path(request['attempt_root'])
    log=(attempt/'worker.log').open('a',buffering=1)
    before=(sys.stdout,sys.stderr);sys.stdout=sys.stderr=log
    started=utc();service=None
    def stage(value,**extra):
        atomic_json(attempt/'worker.json',{'stage':value,'started':started,'updated':utc(),**extra})
    try:
        verify_source(request['source'])
        # A short stable cache root supports Unix sockets and cache reuse between allocations.
        site['runtime_root']=str(Path(site['repo'])/'var/runtime/mqwen')
        os.environ.update(worker_environment(site))
        gpus=visible_gpu_ids()
        if len(gpus)!=1:raise RuntimeError('Meta requires exactly one allocated GPU')
        port=_pick_ports(8100,1)[0]
        protocol={'model_name':site['meta_model'],'max_tokens':site.get('meta_max_output_tokens',16384)}
        service=server_class(Path(site['meta_model_path']),protocol,attempt/'vllm.log',port,gpus[0],
            python=site['inference_python'],max_num_seqs=1,async_scheduling=False,
            gpu_memory_utilization=.85,model_context_length=site.get('meta_context_length',65536),
            tool_call_parser='qwen3_coder')
        stage('loading_model');service.start()
        error=_wait_servers([service],None)
        if error:raise RuntimeError(error)
        # Independent home: no DeepSeek AGENTS, sessions, auth or learned preferences.
        home=attempt/'codex-home';home.mkdir(exist_ok=True)
        config='\n'.join([
            'model = '+repr(site['meta_model']), 'model_provider = "llmgo_qwen_gpu"',
            'model_context_window = '+str(site.get('meta_context_length',65536)),
            'model_auto_compact_token_limit = '+str(site.get('meta_compact_tokens',45000)),
            '[model_providers.llmgo_qwen_gpu]','name = "Qwen GPU"',
            'base_url = '+repr(service.base_url),'wire_api = "responses"',
            'experimental_bearer_token = "local-loopback-only"','requires_openai_auth = false',
            'supports_websockets = false'])+'\n'
        (home/'config.toml').write_text(config);(home/'config.toml').chmod(0o600)
        site.update(codex_home=str(home),meta_thinking=True)
        stage('running')
        memory=MemoryStore(request['campaign_root'])
        backend=backend_class(site,memory)
        backend.session_path=attempt/'session.json';backend.session={}
        prompt=request['prompt']+f"\n本次GPU研究尝试标识：{attempt.name}。只提交本次研究交接，推理卡不用于学生训练。"
        if site.get('meta_engineering_check'):
            from .connection_check import check_connection
            environment=dict(os.environ,CODEX_HOME=str(home),LLMGO_META_THINKING='true',
                LLMGO_META_RUNTIME=str(attempt/'bridge'),LLMGO_META_MAX_OUTPUT_TOKENS=str(site.get('meta_max_output_tokens',16384)))
            acceptance=check_connection(site,attempt/'tool-acceptance',timeout=600,env=environment,tools_check=True)
            if not acceptance['ok']:raise RuntimeError('GPU Meta tool acceptance failed: '+str(acceptance))
            answer={'status':'needs_input','summary':'GPU Meta engineering check passed; no research or training started','proposal_path':None}
        else:
            answer=backend.turn(Path(request['workspace']),prompt)
        stage('releasing_gpu')
        service.stop();service=None
        atomic_json(attempt/'result.json',{'request_id':request['request_id'],'attempt':attempt.name,'answer':answer,'finished':utc()})
        stage('completed',finished=utc())
    except BaseException as error:
        stage('failed',error=str(error),finished=utc());traceback.print_exc();raise
    finally:
        if service is not None:service.stop()
        sys.stdout,sys.stderr=before;log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    run(parser.parse_args().request)
