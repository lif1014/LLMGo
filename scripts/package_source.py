#!/usr/bin/env python3
"""Package and verify this chess RSI pipeline, excluding credentials and run data."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
FOLDERS=('src','scripts','configs','.agents','docs','tests','engines')
TOP=('README.md','plan.md','skills_docs.md','requirements-rsi-control.txt','requirements-rsi-worker.txt','.gitignore','.gitattributes')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(path):
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries')
        for m in members:
            n=PurePosixPath(m.name)
            if not m.isfile() or n.is_absolute() or '..' in n.parts or not n.parts or n.parts[0]!='LLMGo!':
                raise ValueError('Unsafe archive member')
        manifest=json.load(tar.extractfile('LLMGo!/UPLOAD-MANIFEST.json'))
        expected={'LLMGo!/'+name for name in manifest['files']}|{'LLMGo!/UPLOAD-MANIFEST.json'}
        if set(names)!=expected:raise ValueError('Archive file set differs from manifest')
        for name,identity in manifest['files'].items():
            m=tar.getmember('LLMGo!/'+name)
            data=tar.extractfile(m)
            h=hashlib.sha256()
            for block in iter(lambda:data.read(4*1024*1024),b''):h.update(block)
            if h.hexdigest()!=identity['sha256'] or m.size!=identity['bytes'] or m.mode!=identity['mode']:
                raise ValueError('Archive content/mode mismatch: '+name)
    return {'archive':str(path),'files':len(manifest['files']),'sha256':sha(path),'verified':True}


def build(output):
    files=[]
    for folder in FOLDERS:
        for p in sorted((ROOT/folder).rglob('*')):
            if '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.suffix=='.pyc':continue
            if p.is_symlink():raise ValueError('Unexpected source symlink: '+str(p))
            if p.is_file():files.append(p)
    files += [ROOT/name for name in TOP if (ROOT/name).is_file()]
    for p in files:
        if p.name in {'auth.json','config.toml','.env','token','token_expire_at'} or p.suffix in {'.pem','.key'}:
            raise ValueError('Credential-like file must not be packaged: '+str(p))
    try:head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):head=None
    manifest={'created':datetime.now(timezone.utc).isoformat(),'git_base':head,'worktree_snapshot':True,
              'pipeline':'autonomous-chess-v2','base_image':'share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb',
              'excluded':['.git','runs','credentials','Python environments','student weights','LC0 weights'],
              'files':{p.relative_to(ROOT).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777} for p in files}}
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.tmp')
    with tarfile.open(tmp,'w:gz',compresslevel=6) as tar:
        for p in files:tar.add(p,arcname='LLMGo!/'+p.relative_to(ROOT).as_posix(),recursive=False)
        data=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        member=tarfile.TarInfo('LLMGo!/UPLOAD-MANIFEST.json');member.size=len(data);member.mode=0o644
        tar.addfile(member,io.BytesIO(data))
    tmp.replace(output)
    result=verify(output)
    output.with_name(output.name+'.sha256').write_text(result['sha256']+'  '+output.name+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output',type=Path)
    g.add_argument('--verify',type=Path)
    args=p.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else build(args.output),ensure_ascii=False))

if __name__=='__main__':main()
