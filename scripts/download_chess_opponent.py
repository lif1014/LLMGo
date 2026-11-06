#!/usr/bin/env python3
"""Download the pinned public ChessArena model through hf-mirror, with receipts."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.contracts import atomic_json,model_identity,utc

MODEL_ID='ljcnju/Qwen3-8B-Chess'
REVISION='9bf822f4090b9704c16354d8c55d6239925608ae'
ENDPOINT='https://hf-mirror.com'


def download_range(url,partial,start,end,*,opener,attempts=8):
    """Keep short HTTP bodies and resume at the exact next verified byte range."""
    wanted=end-start+1
    for attempt in range(attempts):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==wanted:return
        if offset>wanted:raise ValueError('Range part exceeds its assigned interval')
        lo=start+offset
        request=urllib.request.Request(url+f'?download=true&llmgo_range={lo}-{end}&attempt={attempt}',
            headers={'Range':f'bytes={lo}-{end}','User-Agent':'LLMGo-opponent-download/2.0'})
        try:
            with opener().open(request,timeout=120) as response:
                if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {lo}-'):
                    raise RuntimeError('Mirror did not honor the requested byte range')
                with partial.open('ab') as stream:
                    remaining=wanted-offset
                    while remaining:
                        block=response.read(min(4*1024*1024,remaining))
                        if not block:break
                        stream.write(block);remaining-=len(block)
            if partial.stat().st_size==wanted:return
        except Exception:
            if attempt==attempts-1:raise
        if attempt<attempts-1:time.sleep(min(10,2**attempt))
    raise RuntimeError('Incomplete range after retries: '+partial.name)


def main():
    root=Path('/user/lyh4170/LLMGo!')
    target=root/'assets/models/Qwen3-8B-Chess'
    output=root/'var/runs/qwen3-chess-download-20261106'
    output.mkdir(parents=True,exist_ok=True)
    target.mkdir(parents=True,exist_ok=True)
    receipt=target.parent/'Qwen3-8B-Chess.download.json'
    if receipt.is_file():
        old=json.loads(receipt.read_text())
        if old['revision']==REVISION and old['model_identity']==model_identity(target):
            atomic_json(output/'progress.json',{**old,'stage':'completed','updated':utc()})
            return
        raise RuntimeError('Existing downloaded model differs; preserve for review')
    proxy='http://whitelist-proxy.cybertron.svc.cluster.local:7891'
    def opener():
        return urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy,'http':proxy}))
    with opener().open(ENDPOINT+'/api/models/'+MODEL_ID+'/revision/'+REVISION+'?blobs=true',timeout=60) as response:
        metadata=json.load(response)
    if metadata['sha']!=REVISION:raise ValueError('Mirror resolved another revision')
    siblings=[row for row in metadata['siblings'] if not row['rfilename'].startswith('.')]
    atomic_json(output/'metadata.json',metadata)
    parts=root/'assets/models/.Qwen3-8B-Chess-parts';parts.mkdir(exist_ok=True)
    state={'stage':'downloading','repo_id':MODEL_ID,'revision':REVISION,'endpoint':ENDPOINT,
           'model_path':str(target),'started':utc(),'files':{}}
    lock=threading.Lock()
    def publish():
        with lock:
            state['updated']=utc()
            state['downloaded_bytes']=sum(p.stat().st_size for p in target.iterdir() if p.is_file() and not p.name.endswith('.assembled'))+sum(p.stat().st_size for p in parts.glob('*.part'))
            atomic_json(output/'progress.json',state)
    def finish(row):
        name=row['rfilename'];path=target/name;expected_sha=(row.get('lfs') or {}).get('sha256')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        actual=digest.hexdigest()
        if expected_sha and expected_sha!=actual:raise RuntimeError('Model LFS hash differs: '+name)
        with lock:state['files'][name]={'bytes':path.stat().st_size,'sha256':actual,'lfs_sha256_verified':bool(expected_sha)}
        publish()
    def small_file(row):
        name=row['rfilename'];path=target/name;partial=target/(name+'.partial')
        expected_size=row.get('size') or (row.get('lfs') or {}).get('size')
        for attempt in range(4):
            try:
                if not path.is_file():
                    offset=partial.stat().st_size if partial.exists() else 0
                    headers={'User-Agent':'LLMGo-opponent-download/1.0'}
                    if offset:headers['Range']='bytes='+str(offset)+'-'
                    url=ENDPOINT+'/'+MODEL_ID+'/resolve/'+REVISION+'/'+urllib.parse.quote(name)+'?download=true'
                    request=urllib.request.Request(url,headers=headers)
                    with opener().open(request,timeout=120) as response:
                        append=offset>0 and response.status==206
                        if append and not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):
                            raise RuntimeError('Download resumed at another byte offset')
                        with partial.open('ab' if append else 'wb') as stream:
                            while block:=response.read(4*1024*1024):stream.write(block)
                    partial.rename(path)
                size=path.stat().st_size
                if expected_size and size!=expected_size:raise RuntimeError('Model file size differs: '+name)
                finish(row)
                return
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    try:
        publish()
        shards=[];requests=[]
        for row in siblings:
            name=row['rfilename'];path=target/name
            if not name.endswith('.safetensors'):
                small_file(row);continue
            size=row['size'];prefix=target/(name+'.partial')
            if path.is_file() and path.stat().st_size==size:
                finish(row);continue
            if path.is_file():
                if prefix.exists():raise RuntimeError('Two incomplete versions require review: '+name)
                path.rename(prefix)
            offset=prefix.stat().st_size if prefix.exists() else 0
            if offset>size:raise RuntimeError('Existing prefix is larger than its shard')
            ranges=[]
            for lo in range(offset,size,64*1024*1024):
                hi=min(size-1,lo+64*1024*1024-1);part=parts/f'{name}.{lo}-{hi}.part'
                ranges.append(part);requests.append((ENDPOINT+'/'+MODEL_ID+'/resolve/'+REVISION+'/'+name,part,lo,hi))
            shards.append((row,prefix,ranges))
        errors=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures={pool.submit(download_range,*args,opener=opener):args[1].name for args in requests}
            while futures:
                done,pending=concurrent.futures.wait(futures,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:future.result()
                    except Exception as error:errors.append({'part':futures[future],'error':str(error)})
                futures={f:futures[f] for f in pending}
                with lock:state.update(parts_remaining=len(futures),errors=errors)
                publish()
        if errors:raise RuntimeError('Some byte ranges failed; retained for resume: '+str(errors[:3]))
        for row,prefix,ranges in shards:
            path=target/row['rfilename'];assembled=target/(path.name+'.assembled')
            with assembled.open('wb') as stream:
                for piece in ([prefix] if prefix.exists() else [])+ranges:
                    with piece.open('rb') as source:shutil.copyfileobj(source,stream,8*1024*1024)
            if assembled.stat().st_size!=row['size']:raise RuntimeError('Assembled shard size differs')
            assembled.rename(path);finish(row)
            for piece in ([prefix] if prefix.exists() else [])+ranges:piece.unlink()
        index=json.loads((target/'model.safetensors.index.json').read_text())
        if not all((target/name).is_file() for name in set(index['weight_map'].values())):
            raise RuntimeError('Missing indexed model shard')
        template=json.loads((target/'tokenizer_config.json').read_text())['chat_template']
        if 'enable_thinking' not in template:raise RuntimeError('Model template has no non-thinking switch')
        state.update(stage='completed',model_identity=model_identity(target),finished=utc())
        atomic_json(receipt,state);publish()
        atomic_json(output/'result.json',state)
        print(json.dumps({'stage':'completed','model_path':str(target),'revision':REVISION,
                          'downloaded_bytes':state['downloaded_bytes']}),flush=True)
    except BaseException as error:
        state.update(stage='failed',error_type=type(error).__name__,error=str(error));publish()
        raise


if __name__=='__main__':main()
