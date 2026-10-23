"""Keep Codex tools while adapting its Responses transport to a Chat API."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import inspect
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.parse
import urllib.request
import uuid

from .codex_auth import resolve_auth
from .connection_check import scrub


def chat_base(url):
    url = url.rstrip('/')
    if url.endswith('/chat/completions'):
        url = url[:-len('/chat/completions')]
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError('Invalid upstream base_url')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use a base_url without embedded credentials, query or fragment')
    return url + '/v1' if not parsed.path else url


def proxy_environment(environment):
    env = dict(environment)
    proxy = 'http://whitelist-proxy.cybertron.svc.cluster.local:7891'
    for name in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'):
        env[name] = proxy
    for name in ('ALL_PROXY','all_proxy'):
        env.pop(name, None)
    local = 'localhost,127.0.0.1,::1,.svc,.cluster.local,cybertron.modelbest.co'
    env['NO_PROXY'] = env['no_proxy'] = local
    return env


def create_app(model, api_base, upstream_key, local_token, trace_path, responder=None):
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    if responder is None:
        import litellm
        litellm.telemetry = False
        responder = litellm.aresponses
    known = (upstream_key, local_token)
    trace_path = Path(trace_path)
    trace = trace_path.open('a', encoding='utf-8')
    os.chmod(trace_path, 0o600)

    @asynccontextmanager
    async def lifespan(app):
        yield
        trace.close()

    app = FastAPI(lifespan=lifespan)

    def record(kind, **body):
        trace.write(json.dumps(scrub({'kind':kind, 'ts':time.time(), **body}, known), ensure_ascii=False) + '\n')
        trace.flush()

    def document(value):
        return value.model_dump(mode='json', exclude_none=True) if hasattr(value, 'model_dump') else value

    @app.get('/health')
    async def health():
        return {'status':'ready', 'upstream_api':'chat_completions'}

    @app.post('/v1/responses')
    async def responses(request: Request):
        token = request.headers.get('authorization', '').removeprefix('Bearer ')
        if not hmac.compare_digest(token, local_token):
            raise HTTPException(401, 'Invalid local bridge token')
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 32*1024*1024:
                raise HTTPException(413, 'Request exceeds 32 MiB')
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError('Expected JSON object')
        except (ValueError, TypeError):
            raise HTTPException(400, 'Expected Responses JSON object')
        rid = uuid.uuid4().hex
        record('request', request_id=rid, payload=payload)
        options = dict(payload)
        selected_model = payload.get('model') or model
        options.update(model='openai/'+selected_model, api_base=chat_base(api_base), api_key=upstream_key,
                       use_chat_completions_api=True, num_retries=0, drop_params=True)
        try:
            response = await responder(**options)
        except Exception as error:
            message = scrub(str(error), known)
            record('error', request_id=rid, error=type(error).__name__, detail=message)
            status = getattr(error, 'status_code', 502)
            if not isinstance(status, int) or not 400 <= status <= 599:
                status = 502
            return JSONResponse({'error':{'message':message, 'type':'upstream_error'}}, status_code=status)
        if not payload.get('stream'):
            result = document(response)
            record('response', request_id=rid, response=result)
            return JSONResponse(result)

        async def stream():
            try:
                async for item in response:
                    data = document(item)
                    record('stream', request_id=rid, event=data)
                    kind = data.get('type', 'message')
                    yield 'event: '+kind+'\ndata: '+json.dumps(data, ensure_ascii=False)+'\n\n'
            except Exception as error:
                message = scrub(str(error), known)
                record('stream_error', request_id=rid, error=type(error).__name__, detail=message)
                data = {'type':'error', 'code':'upstream_error', 'message':message}
                yield 'event: error\ndata: '+json.dumps(data, ensure_ascii=False)+'\n\n'
            finally:
                close = getattr(response, 'aclose', None)
                if close:
                    completion = close()
                    if inspect.isawaitable(completion):
                        await completion
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control':'no-cache'})
    return app


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def bridge_exit_error(returncode, log_path, secrets=()):
    detail = f"Chat bridge exited with code {returncode}"
    if isinstance(returncode,int) and returncode < 0:
        try:detail += f" ({signal.Signals(-returncode).name})"
        except ValueError:pass
    # Small startup diagnostic only; full logs remain at the referenced path.
    try:
        with Path(log_path).open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-6000))
            tail=scrub(stream.read().decode('utf-8','replace'),secrets).strip()
    except OSError:
        tail=''
    return detail+'; see '+str(log_path)+ ('\n'+tail if tail else ' (no service log output)')


def runtime_identity():
    release = {}
    path = Path('/etc/os-release')
    if path.is_file():
        for line in path.read_text().splitlines():
            key,sep,value=line.partition('=')
            if sep and key in {'ID','VERSION_ID'}:release[key]=value.strip('"')
    return {'python':list(sys.version_info[:3]),'base_prefix':sys.base_prefix,'os_release':release}


def compatible_bridge_python(default, *, root=Path('/user/lyh4170')):
    # A copied interpreter can start but crash when importing host stdlib native
    # modules (ctypes). Select only a verified environment for this runtime.
    prefix = root/'venvs/llmgo-meta-bridge-devspace-20261023'
    receipt = prefix/'installation.json'
    if receipt.is_file():
        installed = json.loads(receipt.read_text())
        if installed.get('runtime_identity') == runtime_identity() and installed.get('status') == 'installed':
            python = prefix/'bin/python'
            if python.is_file():return str(python)
    return default


def launch(arguments, *, prefix='/user/lyh4170/tools/codex-reused-20261023', bridge_python=None):
    prefix = Path(prefix)
    root = Path('/user/lyh4170')
    if bridge_python is None:
        bridge_python = compatible_bridge_python(str(root/'venvs/llmgo-meta-bridge-20261023/bin/python'))
    assert prefix.resolve().is_relative_to(root)
    home = Path(os.environ.get('CODEX_HOME', root/'.codex-rsi'))
    assert home.resolve().is_relative_to(root)
    config = tomllib.loads((home/'config.toml').read_text())
    env, auth, known = resolve_auth(config, os.environ)
    env['CODEX_HOME'] = str(home)
    env['PATH'] = str(prefix/'bin') + os.pathsep + env.get('PATH', os.defpath)
    env['TMPDIR'] = str(prefix/'_runtime/tmp')
    env['XDG_CACHE_HOME'] = str(prefix/'_runtime/cache')
    env = proxy_environment(env)
    binary = str(prefix/'codex-native')
    informational = arguments and (arguments[0] in {'login','logout','help','completion','features','mcp','sandbox'}
                    or any(arg in {'--version','-V','--help','-h'} for arg in arguments))
    if informational:
        # Help/version/login checks need no local API server.
        result = subprocess.run([binary,*arguments], env=env)
        return result.returncode
    provider = config.get('model_provider','custom')
    if not provider.replace('_','').replace('-','').isalnum() or provider in {'openai','ollama','lmstudio'}:
        raise ValueError('The Chat bridge requires a named custom provider')
    upstream = config['model_providers'][provider]
    key = upstream.get('experimental_bearer_token') or env.get(auth['api_key_env']) or '__not_configured__'
    model = config['model']
    runtime = prefix/'_runtime/bridge'/uuid.uuid4().hex
    runtime.mkdir(parents=True, mode=0o700)
    os.chmod(runtime,0o700)
    local_token = secrets.token_urlsafe(32)
    known = (*known,key,local_token)
    server_env = {**env, 'LLMGO_UPSTREAM_API_KEY':key, 'LLMGO_BRIDGE_TOKEN':local_token,
                  'PYTHONFAULTHANDLER':'1','PYTHONUNBUFFERED':'1',
                  'LITELLM_LOCAL_MODEL_COST_MAP':'True', 'LITELLM_TELEMETRY':'False'}
    server_env.pop('PYTHONPATH',None)
    server_env.pop('PYTHONHOME',None)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    entry = Path(__file__).resolve().parents[3]/'scripts/serve_meta_bridge.py'
    server, native = None, None
    readers = []
    logfile = (runtime/'bridge.log').open('a',encoding='utf-8')
    os.chmod(runtime/'bridge.log',0o600)
    log_lock = threading.Lock()

    def consume(stream, forward=False):
        for line in stream:
            clean = scrub(line,known)
            with log_lock:
                logfile.write(clean);logfile.flush()
            if forward:
                sys.stderr.write(clean);sys.stderr.flush()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    old_handlers = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        print('LLMGo Chat bridge trace: '+str(runtime),file=sys.stderr,flush=True)
        server = subprocess.Popen([bridge_python,'-X','faulthandler','-u',str(entry),'--model',model,'--api-base',chat_base(upstream['base_url']),
                    '--port',str(port),'--trace',str(runtime/'events.jsonl')],
                    env=server_env,cwd=runtime,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,bufsize=1,start_new_session=True)
        thread = threading.Thread(target=consume,args=(server.stdout,),daemon=True);thread.start();readers.append(thread)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic()+90
        while time.monotonic()<deadline:
            if server.poll() is not None:
                thread.join(timeout=2)
                raise RuntimeError(bridge_exit_error(server.returncode,runtime/'bridge.log',known))
            try:
                with opener.open(f'http://127.0.0.1:{port}/health',timeout=1) as response:
                    if response.status==200:break
            except OSError:
                time.sleep(.2)
        else:
            raise TimeoutError('Chat bridge startup timed out; see '+str(runtime/'bridge.log'))
        native_env = dict(env)
        for name in {auth['api_key_env'],'OPENAI_API_KEY','CODEX_API_KEY'}:
            native_env.pop(name,None)
        native_env['LLMGO_BRIDGE_TOKEN'] = local_token
        overrides = {'base_url':f'http://127.0.0.1:{port}/v1', 'env_key':'LLMGO_BRIDGE_TOKEN',
                     'experimental_bearer_token':local_token,'requires_openai_auth':False,'supports_websockets':False}
        command = [binary]
        for name,value in overrides.items():
            command += ['-c',f'model_providers.{provider}.{name}='+json.dumps(value)]
        command += arguments
        native = subprocess.Popen(command,env=native_env,stderr=subprocess.PIPE,text=True,bufsize=1,start_new_session=True)
        thread=threading.Thread(target=consume,args=(native.stderr,True),daemon=True);thread.start();readers.append(thread)
        return native.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        stop_owned(native)
        stop_owned(server)
        for thread in readers:thread.join(timeout=2)
        logfile.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


def serve_main(argv=None):
    parser=argparse.ArgumentParser(description='Local Codex Responses to Chat Completions adapter')
    parser.add_argument('--model',required=True)
    parser.add_argument('--api-base',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    args=parser.parse_args(argv)
    import uvicorn
    app=create_app(args.model,args.api_base,os.environ['LLMGO_UPSTREAM_API_KEY'],os.environ['LLMGO_BRIDGE_TOKEN'],args.trace)
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
