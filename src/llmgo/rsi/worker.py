"""Fixed GPU task entry, selected exclusively by a hashed manifest."""
from __future__ import annotations

import importlib.metadata
import os
import time
import traceback
from pathlib import Path

from .artifacts import load_manifest
from .contracts import atomic_json, file_hash, inside, model_identity, read_json, utc, worker_environment
from .memory import MemoryStore


def execute(manifest_path, expected_sha=None):
    wrapper = load_manifest(manifest_path, expected_sha)
    payload = wrapper["payload"]
    site = payload["site"]
    if Path(payload["model_path"]).resolve() != Path(site["base_model"]).resolve():
        inside(site["user_root"], payload["model_path"])
    os.environ.update(worker_environment(site))
    os.environ["LLMGO_RSI_WORKER"] = "1"
    os.chdir(inside(site["user_root"], payload["source"]["root"]))
    output = inside(site["user_root"], payload["output"])
    output.mkdir(parents=True, exist_ok=True)
    memory = MemoryStore(output / "worker")
    with memory.lock("execution"):
        started = time.monotonic()
        def progress(stage, message, **extra):
            atomic_json(output/"progress.json", {"stage": stage, "message": message, "updated": utc(),
                "job_id": payload["job_id"], "kind": payload["kind"], **extra})
        if (output / "result.json").exists():
            result = read_json(output / "result.json")
            if result["manifest_sha256"] != wrapper["sha256"] or file_hash(result["artifact"]) != result["artifact_sha256"]:
                raise ValueError("Completed worker artifact changed")
            return result
        versions = {}
        for package in ("torch", "transformers", "peft", "vllm", "verl", "python-chess"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        atomic_json(output / "environment.json", {"versions": versions, "manifest_sha256": wrapper["sha256"],
                                                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "utc": utc()})
        try:
            if payload["kind"] == "evaluate":
                from .evaluator import evaluate
                progress("evaluating", "加载候选服务并执行正式20局；完成局数从game_results读取")
                artifact = evaluate(payload)
            elif payload["kind"] == "train":
                from llmgo.harness.skills import validate_cards
                from llmgo.training.rollout import load_curriculum
                from llmgo.training.rewards import Lc0Reward
                from llmgo.training.ppo import training_config
                config = training_config(payload["training"])
                parent = model_identity(payload["model_path"])
                if parent["sha256"] != payload["expected_model_sha256"]:
                    raise ValueError("Training parent checkpoint changed")
                if file_hash(payload["curriculum_path"]) != payload["curriculum_sha256"]:
                    raise ValueError("Frozen curriculum changed")
                records = load_curriculum(payload["curriculum_path"])
                cards = validate_cards(read_json(payload["skills_path"]))
                from llmgo.training.ppo import train
                progress("training", "固定Lc0 critic的PPO；仅评分当前策略实际采样的走法", curriculum_records=len(records), backend=config["backend"])
                provider = Lc0Reward(site, payload["reward"])
                resume = None
                resume_root = inside(site["user_root"], payload.get("resume_from", output / "training"))
                checkpoints = sorted(resume_root.glob("checkpoint-*/state.json"))
                if checkpoints:
                    resume = checkpoints[-1].parent
                try:
                    exported = train(payload["model_path"], records, cards, provider, payload["reward"],
                                     config, output / "training", resume=resume, progress=progress)
                finally:
                    provider.close()
                identity = model_identity(exported)
                progress("exported", "检查点已导出，核验模型身份", model_path=str(exported))
                if identity["sha256"] == parent["sha256"]:
                    raise RuntimeError("Exported weights are identical to the parent")
                artifact = output / "checkpoint-ref.json"
                atomic_json(artifact, {"model_path": str(exported), "identity": identity,
                                       "parent_identity": parent, "training": config, "sft_enabled": False})
            else:
                raise ValueError("Unknown Worker action")
            result = {"status": "succeeded", "manifest_sha256": wrapper["sha256"],
                      "artifact": str(artifact), "artifact_sha256": file_hash(artifact), "utc": utc(),
                      "worker_wall_seconds": time.monotonic()-started, "allocated_gpu_count": site["gpu_count"],
                      "cost_scope": "worker execution; excludes platform queue and container startup"}
            atomic_json(output / "result.json", result)
            progress("completed", "Worker计算完成，等待Executor收集与研究验收", artifact=str(artifact))
            return result
        except Exception as exc:
            progress("failed", "Worker异常终止，完整错误见failure.json", error_type=type(exc).__name__)
            atomic_json(output / "failure.json", {"error": str(exc), "type": type(exc).__name__,
                                                 "traceback": traceback.format_exc(), "utc": utc(),
                                                 "worker_wall_seconds": time.monotonic()-started,
                                                 "manifest_sha256": wrapper["sha256"]})
            raise
