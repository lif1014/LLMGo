"""Run a registered free-form research command or protected chess evaluation."""
from __future__ import annotations
import importlib.metadata
import os
import sys
import subprocess
import time
import traceback
from pathlib import Path
from .artifacts import load_manifest
from .contracts import atomic_json, file_hash, inside, model_identity, read_json, utc, worker_environment
from .memory import MemoryStore


def execute(manifest_path, expected_sha=None):
    wrapper = load_manifest(manifest_path, expected_sha)
    payload, site = wrapper["payload"], wrapper["payload"]["site"]
    if Path(payload["model_path"]).resolve() != Path(site["base_model"]).resolve():
        inside(site["user_root"], payload["model_path"])
    os.environ.update(worker_environment(site))
    os.environ["LLMGO_RSI_WORKER"] = "1"
    output = inside(site["user_root"], payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    memory = MemoryStore(output/"worker")
    with memory.lock("execution"):
        started = time.monotonic()
        def progress(stage, message, **extra):
            atomic_json(output/"progress.json", {"stage":stage, "message":message, "updated":utc(),
                "job_id":payload["job_id"], "kind":payload["kind"], **extra})
        if (output/"result.json").exists():
            result = read_json(output/"result.json")
            if result["manifest_sha256"] != wrapper["sha256"] or file_hash(result["artifact"]) != result["artifact_sha256"]:
                raise ValueError("Completed worker artifact changed")
            return result
        versions = {}
        for package in ("torch", "transformers", "peft", "vllm", "verl", "python-chess"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        inference = None
        if site.get("inference_python"):
            marker = Path(site["inference_python"]).parent.parent/"llmgo-environment.json"
            if marker.is_file():
                inference = read_json(marker)
        atomic_json(output/"environment.json", {"versions":versions,"inference_environment":inference,"manifest_sha256":wrapper["sha256"],
                    "image":site["image"], "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"), "utc":utc()})
        try:
            if payload["kind"] == "evaluate":
                from .evaluator import evaluate
                progress("evaluating", f"受保护裁判执行单档{payload['protocol']['games']}局，研究方法和候选Harness不受限")
                artifact = evaluate(payload)
            elif payload["kind"] == "research":
                candidate = inside(site["user_root"], payload["candidate_source"]["root"])
                command = payload["command"]
                argv = command["argv"]
                # Placeholder substitution is argv-based; no implicit shell evaluation.
                substitutions = {"{python}": sys.executable, "{candidate}":str(candidate),
                    "{output}":str(output), "{model}":payload["model_path"],
                    "{source}":payload["source"]["root"]}
                argv = [str(x) for x in argv]
                for key,value in substitutions.items():
                    argv = [x.replace(key,value) for x in argv]
                env = {**os.environ, "PYTHONPATH":str(candidate/"src"),
                       "LLMGO_OUTPUT":str(output), "LLMGO_MODEL":payload["model_path"],
                       "LLMGO_CANDIDATE":str(candidate), "LLMGO_SITE":str(Path(manifest_path).resolve()),
                       "LLMGO_RESUME_FROM":payload.get("resume_from", "")}
                progress("research", command.get("label", "执行自主研究任务"), argv=argv)
                with (output/"research.log").open("a") as log:
                    subprocess.run(argv, cwd=candidate, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                # Optional receipt may point to any exported own-directory model.
                artifact = output/"research-result.json"
                receipt = read_json(output/"submission.json") if (output/"submission.json").is_file() else {}
                if not isinstance(receipt, dict):
                    raise ValueError("submission.json must be an object")
                if receipt.get("model_path"):
                    inside(site["user_root"], receipt["model_path"])
                    receipt["model_identity"] = model_identity(receipt["model_path"])
                atomic_json(artifact, {**receipt, "argv":argv, "candidate_source_identity":payload["candidate_source"]["sha256"]})
            else:
                raise ValueError("Unknown worker action")
            result = {"status":"succeeded", "manifest_sha256":wrapper["sha256"],
                "artifact":str(artifact), "artifact_sha256":file_hash(artifact), "utc":utc(),
                "worker_wall_seconds":time.monotonic()-started, "allocated_gpu_count":payload.get("resources",{}).get("gpu_count",site["gpu_count"]),
                "cost_scope":"worker execution, excludes queue/startup"}
            atomic_json(output/"result.json",result)
            progress("completed", "产物已落盘，等待Executor收集",artifact=str(artifact))
            return result
        except Exception as exc:
            progress("failed","任务失败，保留完整证据",error_type=type(exc).__name__)
            atomic_json(output/"failure.json", {"error":str(exc), "type":type(exc).__name__,
                "traceback":traceback.format_exc(), "utc":utc(), "manifest_sha256":wrapper["sha256"]})
            raise
