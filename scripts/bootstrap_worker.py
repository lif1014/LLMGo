#!/usr/bin/env python3
"""Install overlays on the specified ptdbench image without replacing its GPU stack."""
import argparse
import fcntl
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import urllib.request

BASE_IMAGE = "share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True, type=Path)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    if sys.prefix != sys.base_prefix:
        # Always inspect the original image stack, not a previously activated overlay.
        command = [sys._base_executable,str(Path(__file__).resolve()),"--prefix",str(args.prefix)]
        if args.check_only:
            command.append("--check-only")
        subprocess.run(command,check=True)
        return
    prefix = args.prefix.absolute()
    if prefix.is_symlink() or prefix.resolve() != prefix or not prefix.is_relative_to(Path("/user/lyh4170")):
        raise ValueError("Use a dedicated, nonsymlink environment inside /user/lyh4170")
    prefix.parent.mkdir(parents=True,exist_ok=True)
    with prefix.with_suffix(".install.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        owner = prefix/"llmgo-owner.json"
        if prefix.exists() and (not owner.is_file() or json.loads(owner.read_text()).get("application") != "llmgo-rsi"):
            raise RuntimeError("Refusing to modify an existing environment not owned by this RSI installer")
        prefix.mkdir(exist_ok=True)
        owner.write_text(json.dumps({"application":"llmgo-rsi","base_image":BASE_IMAGE})+"\n")
        for name,path in {"PIP_CACHE_DIR":prefix/"_runtime/cache/pip",
                          "XDG_CACHE_HOME":prefix/"_runtime/cache",
                          "TMPDIR":prefix/"_runtime/tmp"}.items():
            path.mkdir(parents=True,exist_ok=True)
            os.environ[name]=str(path)
        os.environ["PYTHONDONTWRITEBYTECODE"]="1"
        os.environ["PIP_CONFIG_FILE"]="/dev/null"
        os.environ["PIP_REQUIRE_VIRTUALENV"]="true"
        # Ignore inherited pip destinations/configuration from unrelated tasks.
        for key in list(os.environ):
            if key.startswith("PIP_") and key not in {"PIP_CACHE_DIR","PIP_CONFIG_FILE","PIP_REQUIRE_VIRTUALENV"}:
                os.environ.pop(key,None)
        for key in ("PYTHONPATH","PYTHONHOME","PYTHONUSERBASE"):
            os.environ.pop(key,None)
        os.environ["PYTHONNOUSERSITE"]="1"
        sys.dont_write_bytecode=True
        pip_options = None
        def install(python, constraints, requirements):
            nonlocal pip_options
            if pip_options is None:
                index="https://mirrors.ustc.edu.cn/pypi/simple"
                failures=[]
                for proxy in (None,"http://whitelist-proxy.cybertron.svc.cluster.local:7891"):
                    try:
                        handler=urllib.request.ProxyHandler({"http":proxy,"https":proxy} if proxy else {})
                        with urllib.request.build_opener(handler).open(index+"/python-chess/",timeout=12) as response:
                            if response.status != 200:
                                raise RuntimeError("HTTP "+str(response.status))
                        pip_options=["--index-url",index,"--timeout","30","--retries","1"]
                        if proxy:
                            pip_options += ["--proxy",proxy]
                        print(json.dumps({"download_index":index,"route":"cluster_proxy" if proxy else "direct"}),flush=True)
                        break
                    except Exception as exc:
                        failures.append(type(exc).__name__+": "+str(exc))
                if pip_options is None:
                    raise RuntimeError("USTC index unreachable via direct and cluster proxy: "+"; ".join(failures))
            subprocess.run([python,"-m","pip","install",*pip_options,"--constraint",str(constraints),"-r",str(requirements)],check=True)
        repo = Path(__file__).resolve().parents[1]
        requirements = repo/"requirements-rsi-worker.txt"
        import hashlib
        fingerprint = hashlib.sha256(requirements.read_bytes()+(repo/"requirements-rsi-control.txt").read_bytes()+Path(__file__).read_bytes()).hexdigest()
        marker = prefix/"llmgo-environment.json"
        if args.check_only:
            if not marker.exists():
                raise RuntimeError("Worker overlay not installed")
            print(marker.read_text())
            return
        critical = {}
        for dist in metadata.distributions():
            name = dist.metadata.get("Name", "").lower().replace("_","-")
            if name in {"torch","vllm","torchvision","torchaudio","triton","tokenizers"} or name.startswith("nvidia-"):
                critical[name] = dist.version
        if not {"torch","vllm","tokenizers"}.issubset(critical):
            raise RuntimeError("Run this installer inside the specified ptdbench GPU image")
        # This image lacks working ensurepip. Its system pip can install into
        # a system-site-packages venv using the venv interpreter's own prefix.
        if not marker.exists():
            venv.EnvBuilder(with_pip=False,system_site_packages=True).create(prefix)
        python = str(prefix/"bin/python")
        if not marker.exists() or json.loads(marker.read_text()).get("requirements_sha256") != fingerprint:
            constraints = prefix/"base-stack.constraints.txt"
            constraints.write_text("".join(f"{k}=={v}\n" for k,v in sorted(critical.items())))
            # Preserve torch/CUDA/vLLM/tokenizers; Transformers 4.57 lacks Qwen3.5.
            install(python,constraints,requirements)
        # vLLM 0.17 declares Transformers<5 and ships its own Qwen3.5 config.
        # HF training needs Transformers5.2. Keep separate overlays on one base image.
        inference = prefix.with_name(prefix.name+"-inference")
        if not (inference/"llmgo-environment.json").exists() or json.loads((inference/"llmgo-environment.json").read_text()).get("requirements_sha256") != fingerprint:
            infer_owner=inference/"llmgo-owner.json"
            if inference.exists() and (not infer_owner.is_file() or json.loads(infer_owner.read_text()).get("application") != "llmgo-rsi"):
                raise RuntimeError("Refusing to modify an unrelated inference environment")
            inference.mkdir(exist_ok=True)
            infer_owner.write_text(json.dumps({"application":"llmgo-rsi","base_image":BASE_IMAGE})+"\n")
            venv.EnvBuilder(with_pip=False,system_site_packages=True).create(inference)
            infer_constraints = inference/"base-stack.constraints.txt"
            infer_versions = {**critical,"transformers":metadata.version("transformers")}
            infer_constraints.write_text("".join(f"{k}=={v}\n" for k,v in sorted(infer_versions.items())))
            install(str(inference/"bin/python"),infer_constraints,repo/"requirements-rsi-control.txt")
        infer_check = subprocess.run([str(inference/"bin/python"),"-c",'import vllm,chess,httpx; from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config; import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ("torch","transformers","vllm","tokenizers")}))'],capture_output=True,text=True)
        if infer_check.returncode:
            raise RuntimeError("Inference overlay import failed: "+infer_check.stderr[-6000:])
        check = subprocess.run([python,"-c",'import torch,peft,chess,httpx,fastapi,uvicorn; from transformers import Qwen3_5ForConditionalGeneration; import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ("torch","transformers","peft","tokenizers")}))'],capture_output=True,text=True)
        if check.returncode:
            raise RuntimeError("HF training overlay import failed: "+check.stderr[-6000:])
        launcher = prefix/"bin/vllm"
        launcher.write_text('#!'+str(inference/"bin/python")+'\nfrom vllm.entrypoints.cli.main import main\nif __name__ == "__main__": main()\n')
        launcher.chmod(0o755)
        (inference/"llmgo-environment.json").write_text(json.dumps({"base_image":BASE_IMAGE,
              "versions":json.loads(infer_check.stdout.strip().splitlines()[-1]),"requirements_sha256":fingerprint},indent=2)+"\n")
        record = {"base_image":BASE_IMAGE,"requirements_sha256":fingerprint,
                  "base_stack":critical,"versions":json.loads(check.stdout.strip().splitlines()[-1]),
                  "inference_versions":json.loads(infer_check.stdout.strip().splitlines()[-1]),
                  "inference_python":str(inference/"bin/python"),"gpu_tested":False,"note":"Import validation only; run doctor --worker in a GPU job"}
        marker.write_text(json.dumps(record,indent=2)+"\n")
        subprocess.run([python,"-m","pip","freeze"],stdout=(prefix/"installed.freeze.txt").open("w"),check=True)
        print(json.dumps(record))


if __name__ == "__main__":
    main()
