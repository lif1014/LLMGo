"""Operator and Meta-Agent command line for the devspace controller."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from .contracts import REPO, inside, load_site, read_json
from .codex_auth import resolve_auth
from .controller import Controller
from .memory import MemoryStore
from llmgo.remote.cctl_env import cctl_environment, cctl_failure_detail
from llmgo.remote.pools import own_pool_rows, authorized_pool, pool_summary


def doctor(site, *, worker=False, platform=False, codex_auth=False, ppo=False):
    checks = []
    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    check("user_root", Path(site["user_root"]).is_dir(), site["user_root"])
    check("repository", (Path(site["repo"])/"scripts/rsi.py").is_file(), site["repo"])
    check("base_model", (Path(site["base_model"])/"config.json").is_file(), site["base_model"])
    check("fairy_stockfish", os.access(site["engine"], os.X_OK), site["engine"])
    if worker:
        for name in ("torch", "transformers", "peft", "vllm", "chess", "httpx", "tokenizers"):
            check(name, importlib.util.find_spec(name) is not None, "Worker Python dependency")
        if ppo:
            check("lc0", os.access(site["lc0_bin"], os.X_OK), site["lc0_bin"])
            check("reward_network", Path(site["reward_weights"]).is_file(), site["reward_weights"])
        if importlib.util.find_spec("torch"):
            import torch
            check("gpu", torch.cuda.is_available(), f"{torch.cuda.device_count()} visible devices")
    else:
        check("codex_config", (Path(site["codex_home"])/"config.toml").is_file(), site["codex_home"])
        check("codex_program", shutil.which(site["codex_bin"]), site["codex_bin"])
        check("cctl_program", shutil.which(site["cctl_bin"]), site["cctl_bin"])
        if codex_auth:
            try:
                config_path = Path(site["codex_home"])/"config.toml"
                config = tomllib.loads(config_path.read_text()) if config_path.is_file() else {}
                auth_env, auth, _ = resolve_auth(config, {**os.environ, "CODEX_HOME": site["codex_home"]})
                if auth["api_key_present"]:
                    # Inline/custom-provider credentials need no CLI OAuth login.
                    # Presence is a configuration check, not an authentication test.
                    check("codex_provider_credentials", True,
                          f"Configured via {auth['auth_source']}; run check_meta_connection.py to verify inference")
                else:
                    result = subprocess.run([site["codex_bin"], "login", "status"], capture_output=True, text=True,
                        timeout=30, cwd=site["user_root"], env=auth_env)
                    check("codex_login_status", result.returncode == 0,
                          f"CLI login status exited {result.returncode}; no model request tested; raw output withheld")
            except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
                check("codex_login_status", False, type(exc).__name__)
        if platform and shutil.which(site["cctl_bin"]):
            try:
                cctl_env = cctl_environment()
                check("cctl_server", True, cctl_env.get("CCTL_SERVER", "CLI default configuration"))
                result = subprocess.run([site["cctl_bin"], "pool", "list", "--own", "--no-input", "-o", "json"],
                                        capture_output=True, text=True, timeout=90,
                                        env=cctl_env)
                pools = json.loads(result.stdout) if result.returncode == 0 else []
                rows = own_pool_rows(pools)
                allowed = authorized_pool(rows)
                detail = ("Authenticated own-pool lookup" if allowed else
                          "model-autosft not found in own pools; returned: "+pool_summary(rows)) if result.returncode == 0 else cctl_failure_detail(result,cctl_env)
                if result.returncode == 0 and not rows:
                    detail = ("cctl returned [] for own pools. Check cctl profile current/list and the logged-in account's "
                              "pool membership on this server; do not bypass the model-autosft permission check.")
                check("cctl_auth_and_pool", result.returncode == 0 and allowed, detail)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                check("cctl_auth_and_pool", False, str(exc) if isinstance(exc,ValueError) else type(exc).__name__)
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "inference_tested": False,
            "note": "CLI login status is optional and does not verify model inference; custom providers may use their own credentials. No inference, training or submission performed. Remote paths are expected to be absent on the development machine."}


def parser():
    p = argparse.ArgumentParser(description="Chess autonomous research with fixed multi-level evaluation")
    p.add_argument("--site", default=str(REPO / "configs/rsi/site.beijing.json"))
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--worker", action="store_true")
    d.add_argument("--ppo", action="store_true", help="Also check optional Lc0 PPO assets")
    d.add_argument("--platform", action="store_true")
    d.add_argument("--codex-auth", action="store_true", help="Check provider credential configuration or CLI login status, without inference")
    init = sub.add_parser("init")
    init.add_argument("campaign")
    for name in ("status", "step", "run", "workspace", "prepare", "preview", "submit", "resume-failed", "resume", "recover-meta", "index-activity", "upgrade"):
        cmd = sub.add_parser(name)
        cmd.add_argument("campaign")
        if name == "prepare":
            cmd.add_argument("--proposal", required=True)
        if name in ("preview", "submit"):
            cmd.add_argument("--job")
        if name == "run":
            cmd.add_argument("--steps", type=int, help="Optional operator limit; omitted means continuous operation")
    mem = sub.add_parser("memory")
    mem.add_argument("campaign")
    mem.add_argument("--search")
    mem.add_argument("--kind")
    mem.add_argument("--limit", type=int, default=20)
    note = sub.add_parser("note")
    note.add_argument("campaign")
    note.add_argument("--text", required=True)
    curriculum = sub.add_parser("curriculum")
    curriculum.add_argument("--events", action="append", default=[])
    curriculum.add_argument("--output", required=True)
    curriculum.add_argument("--count", type=int, default=16)
    curriculum.add_argument("--seed", type=int, default=4170)
    exe = sub.add_parser("executor", help="Isolated job lifecycle; raw output stays outside Meta-Agent Memory")
    exe.add_argument("action", choices=("run", "step", "status"))
    exe.add_argument("campaign")
    exe.add_argument("--job", help="Job ID for step/status; run follows the Controller outbox")
    exe.add_argument("--once", action="store_true", help="run performs one service pass instead of running continuously")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "curriculum":
        from llmgo.training.rollout import from_game_events, seed_curriculum, write_curriculum
        import random
        if args.count < 1:
            raise ValueError("count must be positive")
        if args.events:
            rows = [row for path in args.events for row in from_game_events(path)]
            random.Random(args.seed).shuffle(rows)
            rows = rows[:args.count]
        else:
            rows = seed_curriculum(args.count, args.seed)
        result = {"records": write_curriculum(args.output, rows), "output": args.output}
    else:
        site = load_site(args.site)
        if args.command == "doctor":
            result = doctor(site, worker=args.worker, platform=args.platform, codex_auth=args.codex_auth, ppo=args.ppo)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 1
        from .contracts import identifier
        root = inside(site["user_root"], Path(site["campaign_root"]) / identifier(args.campaign))
        if args.command == "init":
            controller = Controller.create(site, args.campaign)
            result = {"campaign": str(root), "state": controller.state}
        else:
            if not root.is_dir():
                raise FileNotFoundError(f"Unknown campaign: {root}")
            controller = Controller(root)
            if site != controller.site and args.command != "upgrade":
                raise ValueError("Campaign site is frozen; use its original site profile")
            if args.command == "index-activity":
                # Constructor replay indexes legacy records once, outside HTTP.
                MemoryStore(root/"executor")
                result = {"indexed": True, "campaign": str(root), "raw_journals_changed": False}
            elif args.command == "executor":
                if args.action == "run":
                    if args.job:
                        raise ValueError("Executor run follows active_job, not a manually selected old task")
                    if site.get("executor_mode", "embedded") != "external":
                        raise ValueError("Set executor_mode=external before campaign init to use a separate Executor service")
                    result = controller.lifecycle.run(once=args.once)
                else:
                    job_id = args.job or controller.state["active_job"]
                    if not job_id and args.action == "status":
                        job_id = next(reversed(controller.state["jobs"]), None)
                    if not job_id:
                        result = {"status": "idle"}
                    elif job_id not in controller.state["jobs"]:
                        raise ValueError("Unknown job")
                    elif args.action == "step":
                        if job_id != controller.state["active_job"]:
                            raise ValueError("Executor can advance only the current active job")
                        result = controller.lifecycle.advance(controller.state["jobs"][job_id])
                    else:
                        result = controller.lifecycle.report(controller.state["jobs"][job_id])
            elif args.command == "status":
                result = controller.memory.context()
            elif args.command == "memory":
                result = (controller.memory.search(args.search, args.limit) if args.search
                          else controller.memory.recent(args.limit, args.kind))
            elif args.command == "note":
                result = controller.memory.append("research_note", {"text": args.text})
            elif args.command == "run":
                if args.steps is not None and args.steps < 1:
                    raise ValueError("steps must be positive")
                result = controller.run(max_steps=args.steps)
            else:
                with controller.memory.lock():
                    if args.command == "upgrade":
                        with controller.lifecycle.memory.lock("service"):
                            result = controller.upgrade(site)
                    elif args.command == "step":
                        result = controller.step()
                    elif args.command == "workspace":
                        if controller.state["phase"] != "PLANNING":
                            raise ValueError("A planning workspace requires a completed baseline")
                        result = {"workspace": str(controller.workspace())}
                    elif args.command == "prepare":
                        result = controller.prepare(args.proposal)
                    elif args.command in ("preview", "submit"):
                        job_id = args.job or controller.state["active_job"]
                        if not job_id or job_id not in controller.state["jobs"]:
                            raise ValueError("No prepared job")
                        job = controller.state["jobs"][job_id]
                        if args.command == "submit" and job_id != controller.state["active_job"]:
                            raise ValueError("Only the current active job may be submitted")
                        result = (controller.executor.preview(job["manifest"]) if args.command == "preview"
                                  else controller.lifecycle.advance(job))
                    elif args.command == "resume-failed":
                        controller.resume_failed()
                        result = {"phase": controller.state["phase"], "job_id": controller.state["active_job"]}
                    elif args.command == "resume":
                        result = controller.resume()
                    elif args.command == "recover-meta":
                        result = controller.recover_meta()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.command == "doctor" and not result["ok"] else 0
