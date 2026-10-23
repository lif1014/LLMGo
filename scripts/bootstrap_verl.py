#!/usr/bin/env python3
"""Install a dedicated verl PPO environment, preserving the image GPU stack."""
import argparse
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import venv
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--requirements", type=Path, default=Path(__file__).resolve().parents[1]/"requirements-rsi-verl.txt")
    args = parser.parse_args()
    if sys.prefix != sys.base_prefix:
        subprocess.run([sys._base_executable, str(Path(__file__).resolve()), "--prefix", str(args.prefix), "--requirements", str(args.requirements)], check=True)
        return
    prefix = args.prefix.absolute()
    if prefix.resolve() != prefix or not prefix.is_relative_to(Path("/user/lyh4170/venvs")):
        raise ValueError("verl environment must be a dedicated nonsymlink path under /user/lyh4170/venvs")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(args.requirements.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    with prefix.with_suffix(".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        owner = prefix/"llmgo-owner.json"
        if prefix.exists() and (not owner.is_file() or json.loads(owner.read_text()).get("application") != "llmgo-verl-ppo"):
            raise RuntimeError("Refusing to alter an environment owned by another task")
        prefix.mkdir(exist_ok=True)
        owner.write_text(json.dumps({"application":"llmgo-verl-ppo"})+"\n")
        for key in list(os.environ):
            if key.startswith("PIP_") or key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
                os.environ.pop(key, None)
        os.environ.update(PIP_CONFIG_FILE="/dev/null", PIP_REQUIRE_VIRTUALENV="true", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        for name, leaf in {"PIP_CACHE_DIR":"cache/pip", "XDG_CACHE_HOME":"cache", "TMPDIR":"tmp"}.items():
            path = prefix/"_runtime"/leaf
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        marker = prefix/"llmgo-environment.json"
        if marker.exists() and json.loads(marker.read_text()).get("requirements_sha256") == fingerprint:
            print(marker.read_text())
            return
        critical = {}
        for dist in metadata.distributions():
            name = dist.metadata.get("Name", "").lower().replace("_", "-")
            if name in {"torch", "vllm", "torchvision", "torchaudio", "triton", "tokenizers"} or name.startswith("nvidia-"):
                critical[name] = dist.version
        if not {"torch", "vllm", "tokenizers"}.issubset(critical):
            raise RuntimeError("Use the designated ptdbench image")
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(prefix)
        constraints = prefix/"base-stack.constraints.txt"
        constraints.write_text("".join(f"{k}=={v}\n" for k,v in sorted(critical.items())))
        python = str(prefix/"bin/python")
        # No extras: verl[vllm] would downgrade the image's vLLM/CUDA stack.
        command = [python,"-m","pip","install","--index-url","https://mirrors.ustc.edu.cn/pypi/simple", "--timeout","30","--retries","1","-c",str(constraints),"-r",str(args.requirements)]
        errors = []
        for proxy in (None,"http://whitelist-proxy.cybertron.svc.cluster.local:7891"):
            try:
                handler = urllib.request.ProxyHandler({"http":proxy,"https":proxy} if proxy else {})
                with urllib.request.build_opener(handler).open("https://mirrors.ustc.edu.cn/pypi/simple/verl/",timeout=15) as response:
                    if response.status != 200:
                        raise RuntimeError("USTC mirror unavailable")
                if proxy:
                    command += ["--proxy",proxy]
                break
            except Exception as exc:
                errors.append(type(exc).__name__+": "+str(exc))
        else:
            raise RuntimeError("USTC direct/proxy routes failed: "+"; ".join(errors))
        subprocess.run(command, check=True)
        check = subprocess.run([python,"-c",'import verl,ray,tensordict,vllm; from transformers import Qwen3_5ForConditionalGeneration; from verl.trainer.main_ppo import TaskRunner; from verl.trainer.ppo.core_algos import register_adv_est; import json,importlib.metadata as m; print(json.dumps({k:m.version(k) for k in ("verl","torch","vllm","transformers","ray","numpy")}))'],capture_output=True,text=True)
        if check.returncode:
            raise RuntimeError("verl import validation failed: "+check.stderr[-8000:])
        receipt = {"requirements_sha256":fingerprint,"versions":json.loads(check.stdout.strip().splitlines()[-1]),"gpu_tested":False,"note":"Import validation only"}
        marker.write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
