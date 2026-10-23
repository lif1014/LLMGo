"""cctl transport with durable submission intents and conservative reconciliation."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from llmgo.rsi.artifacts import load_manifest
from llmgo.rsi.contracts import atomic_json, digest, inside, read_json, utc
from .cctl_env import cctl_environment, cctl_failure_detail
from .pools import own_pool_rows, authorized_pool

TERMINAL_FAILURE = {"Failed", "OOMKilled", "ImagePullBackOff", "Invalid", "Killed", "Stopped"}


class SubmissionUncertain(RuntimeError):
    """Do not resubmit: a server-side task may already exist."""


class CctlCommandError(RuntimeError):
    """A concise transport failure; raw command output stays in the executor log."""


def _task_id(task):
    """v2 list entries use name=tasks/ID; get/create may also include id."""
    raw, name = task.get("id"), task.get("name")
    numeric = None
    if raw is not None:
        if type(raw) is int and raw > 0:
            numeric = raw
        elif isinstance(raw, str) and re.fullmatch(r"[1-9][0-9]*", raw):
            numeric = int(raw)
        else:
            raise ValueError("Invalid numeric task ID")
    if name is not None:
        match = re.fullmatch(r"tasks/([1-9][0-9]*)", name) if isinstance(name, str) else None
        if match is None or numeric is not None and int(match[1]) != numeric:
            raise ValueError("Invalid or conflicting task resource name")
        numeric = int(match[1])
    if numeric is None:
        raise ValueError("Task response has neither an ID nor a tasks/ID resource name")
    return numeric


class CctlExecutor:
    def __init__(self, site, memory, runner=subprocess.run, *, record_commands=False):
        self.site, self.memory, self.runner = site, memory, runner
        self.record_commands = record_commands

    def command(self, args, *, timeout=90, json_output=True):
        command = [self.site["cctl_bin"], *args]
        if json_output:
            command += ["--no-input", "-o", "json"]
        started = time.monotonic()
        operation_id = uuid.uuid4().hex
        if self.record_commands:
            self.memory.append("command_started", {"argv": command, "operation_id": operation_id, "timeout_sec": timeout})
        try:
            environment = cctl_environment()
            result = self.runner(command, capture_output=True, text=True, timeout=timeout,
                                 env=environment)
        except (OSError, subprocess.TimeoutExpired) as exc:
            if self.record_commands:
                self.memory.append("command", {"argv": command, "operation_id": operation_id, "error_type": type(exc).__name__,
                    "stdout": _text(getattr(exc, "stdout", None)), "stderr": _text(getattr(exc, "stderr", None)),
                    "seconds": time.monotonic()-started})
            raise
        if self.record_commands:
            self.memory.append("command", {"argv": command, "operation_id": operation_id, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "seconds": time.monotonic()-started})
        if result.returncode:
            raise CctlCommandError(f"{cctl_failure_detail(result,environment)}; see executor journal")
        return json.loads(result.stdout) if json_output else result.stdout

    def request(self, manifest_path):
        manifest = load_manifest(manifest_path)
        payload, site = manifest["payload"], self.site
        if payload["site"] != site:
            raise ValueError("Manifest and executor site differ")
        if (site["cluster"], site["resource_pool"], site["priority"]) != ("beijing", "model-autosft", "NORMAL"):
            raise ValueError("Unauthorized resource selection")
        if site["user_root"] != "/user/lyh4170":
            raise ValueError("Only /user/lyh4170 can submit remote experiments")
        inside(site["user_root"], manifest_path)
        resources = {k: site[k] for k in ("gpu_count", "cpu", "memory_gib")}
        resources.update(payload.get("resources", {}))
        gpu = resources["gpu_count"]
        if type(gpu) is not int or not 0 <= gpu <= 8:
            raise ValueError("gpu_count must be an integer from 0 to 8")
        for key in ("cpu", "memory_gib"):
            if type(resources[key]) is not int or resources[key] < 1:
                raise ValueError("Worker CPU/memory must be positive integers")
        script = Path(payload["source"]["root"]) / ("scripts/worker_entry.py" if site.get("worker_venv") else "scripts/rsi_worker.py")
        entry = shlex.join([site["python"], str(script), "--manifest", str(manifest_path),
                            "--manifest-sha256", manifest["sha256"]])
        description = f"llmgo-rsi:{payload['campaign_id']}:{payload['job_id']}:{manifest['sha256']}"
        return ["job", "create", "--project", site["project"], "--cluster", "beijing",
                "--resource-pool", "model-autosft", "--billing-account-id", site["billing_account_id"],
                "--priority", "NORMAL", "--image", site["image"], "--gpu", str(gpu),
                *(["--gpu-model", site["gpu_model"]] if gpu else []),
                "--cpu", str(resources["cpu"]), "--memory", str(resources["memory_gib"]),
                "--description", description, "--entry", entry], description

    def preview(self, manifest_path):
        args, description = self.request(manifest_path)
        return {"argv": [self.site["cctl_bin"], *args, "--no-input", "-o", "json"],
                "description": description, "submitted": False}

    def submit(self, manifest_path):
        args, description = self.request(manifest_path)
        intent_path = Path(manifest_path).with_name("submission.json")
        if intent_path.exists():
            intent = read_json(intent_path)
            if intent["description"] != description:
                raise ValueError("Existing submission belongs to a different manifest")
            if intent.get("task_id"):
                return intent
            return self.reconcile(intent_path)
        pools = self.command(["pool", "list", "--own"])
        if not authorized_pool(own_pool_rows(pools)):
            raise ValueError("Authorized model-autosft pool is unavailable")
        intent = {"description": description, "request_hash": digest(args), "status": "SUBMITTING",
                  "manifest": str(manifest_path), "created": utc()}
        atomic_json(intent_path, intent)
        self.memory.append("submission_intent", intent)
        try:
            task = self.command(args)
            task_id = _task_id(task)
        except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
            intent["status"] = "UNKNOWN"
            atomic_json(intent_path, intent)
            self.memory.append("submission_uncertain", {"description": description, "error": str(exc)})
            raise SubmissionUncertain("Submission response is uncertain; reconcile before any retry") from exc
        intent.update(task_id=task_id, status=task["status"])
        atomic_json(intent_path, intent)
        self.memory.append("task_submitted", intent)
        return intent

    def reconcile(self, intent_path):
        intent = read_json(intent_path)
        found = {}
        # Enumerate own tasks; no blind retry if listings are incomplete or ambiguous.
        for page in range(1, 101):
            response = self.command(["job", "list", "--own", "--cluster", "beijing", "--project",
                                     self.site["project"], "--pool", "model-autosft",
                                     "--page", str(page), "--limit", "100"])
            rows = response if isinstance(response, list) else response.get("tasks", response.get("items", []))
            for row in rows:
                try:
                    task_id = _task_id(row)
                except ValueError as exc:
                    raise SubmissionUncertain("Invalid task handle in listing; retain intent, do not resubmit") from exc
                if row.get("description") == intent["description"]:
                    found[task_id] = row
                elif row.get("description") is None:
                    detail = self.command(["job", "get", str(task_id)])
                    try:
                        if _task_id(detail) != task_id:
                            raise ValueError("Task detail returned a different ID")
                    except ValueError as exc:
                        raise SubmissionUncertain("Task detail identity is uncertain; retain intent, do not resubmit") from exc
                    if detail.get("description") == intent["description"]:
                        found[task_id] = detail
            if len(rows) < 100:
                break
        else:
            raise SubmissionUncertain("Task listing limit reached; retain intent, do not resubmit")
        if len(found) != 1:
            raise SubmissionUncertain(f"Found {len(found)} matching tasks. Retain intent; do not automatically resubmit.")
        task_id, task = next(iter(found.items()))
        intent.update(task_id=task_id, status=task["status"])
        atomic_json(intent_path, intent)
        self.memory.append("task_reconciled", intent)
        return intent

    def get(self, intent):
        task = self.command(["job", "get", str(intent["task_id"])])
        if (_task_id(task) != int(intent["task_id"])
                or task.get("description") != intent["description"] or task.get("project") != self.site["project"]
                or task.get("cluster") != "beijing" or task.get("resourcePoolId") != 525
                or task.get("priority") != "NORMAL"):
            raise ValueError("Task identity or authorized resource envelope changed")
        return task

    def logs(self, intent, pod):
        self.get(intent)  # Verify the recorded identity before reading logs.
        return self.command(["job", "logs", str(intent["task_id"]), "--pod", pod, "--page-size", "200"],
                            json_output=False)


def _text(value):
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value
