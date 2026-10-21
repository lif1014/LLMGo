#!/usr/bin/env python3
"""Bootstrap the shared overlay in a fresh image, then exec the worker interpreter."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--manifest",required=True)
p.add_argument("--manifest-sha256",required=True)
args = p.parse_args()
wrapper = json.loads(Path(args.manifest).read_text())
# Verify the externally supplied hash before using any installation path.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.artifacts import load_manifest
load_manifest(args.manifest,args.manifest_sha256)
site = wrapper["payload"]["site"]
prefix = Path(site["worker_venv"])
repo = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(repo/"scripts/bootstrap_worker.py"),"--prefix",str(prefix)],check=True)
env = {**os.environ,"PATH":str(prefix/"bin")+os.pathsep+os.environ.get("PATH","")}
os.execve(str(prefix/"bin/python"), [str(prefix/"bin/python"),str(repo/"scripts/rsi_worker.py"),
          "--manifest",args.manifest,"--manifest-sha256",args.manifest_sha256],env)
