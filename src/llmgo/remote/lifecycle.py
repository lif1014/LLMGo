"""Job lifecycle without an LLM: isolated audit journal and small durable reports.

The Controller can host this component or only read its reports while a separate
devspace process runs it. Neither mode feeds status polls/raw logs to Meta-Agent.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from llmgo.rsi.contracts import atomic_json, digest, file_hash, identifier, inside, read_json, utc
from llmgo.rsi.memory import MemoryStore, redact
from .cctl import CctlExecutor, CctlCommandError, SubmissionUncertain, TERMINAL_FAILURE

FINAL = {"collected", "failed", "artifact_invalid"}


class JobExecutor:
    def __init__(self, root, site, *, transport=None, clock=time.time):
        self.root = inside(site["user_root"], root)
        self.site = site
        self.memory = MemoryStore(self.root / "executor")
        self.transport = transport or CctlExecutor(site, self.memory, record_commands=True)
        self.clock = clock
        self.interval = max(1., float(site.get("poll_seconds", 120)))
        self.max_interval = max(self.interval, float(site.get("executor_max_poll_seconds", 600)))

    def poll_bounds(self, payload):
        gpu_count = payload.get("resources", {}).get("gpu_count", self.site.get("gpu_count", 0))
        if gpu_count > 0:
            base = min(600., max(300., float(self.site.get("gpu_poll_seconds", 300))))
            ceiling = min(600., max(base, float(self.site.get("gpu_max_poll_seconds", 600))))
            return base, ceiling
        return self.interval, self.max_interval

    def report_path(self, job):
        return inside(self.root, self.root / "jobs" / identifier(job["id"]) / "executor/report.json")

    def _manifest(self, job):
        path = inside(self.root, job["manifest"])
        if path != self.report_path(job).parent.parent / "manifest.json":
            raise ValueError("Executor manifest is not in the registered job directory")
        wrapper = read_json(path)
        payload = wrapper["payload"]
        if (wrapper["sha256"] != job["sha256"] or digest(payload) != job["sha256"]
                or payload["job_id"] != job["id"] or payload["site"] != self.site
                or payload["campaign_id"] != self.root.name):
            raise ValueError("Executor manifest identity differs from registered job")
        inside(self.root, payload["output"])
        return payload

    def report(self, job):
        path = self.report_path(job)
        if not path.exists():
            return {"job_id": job["id"], "manifest_sha256": job["sha256"], "status": "waiting_for_executor",
                    "task_id": None, "platform_status": None, "report_path": str(path),
                    "summary": "Waiting for an Executor report; no platform call made by Controller"}
        report = read_json(path)
        if report["job_id"] != job["id"] or report["manifest_sha256"] != job["sha256"]:
            raise ValueError("Executor report belongs to a different job/manifest")
        return report

    def advance(self, job, *, force=False):
        # Separate from the campaign writer lock: an external Executor may run
        # while Controller owns its lock, including across long Codex turns.
        with self.memory.lock("lifecycle"):
            payload = self._manifest(job)
            previous = self.report(job)
            if previous["status"] in FINAL:
                return previous
            now = self.clock()
            interval, max_interval = self.poll_bounds(payload)
            due = previous.get("next_check_at", 0)
            prior_delay = previous.get("interval_seconds", interval)
            if previous.get("checks") and prior_delay < interval:
                # Honor the new GPU minimum even when resuming an older report.
                due = max(due, due-prior_delay+interval)
            if not force and now < due <= now + max_interval:
                return previous
            report = {"schema_version": 1, "job_id": job["id"], "manifest_sha256": job["sha256"],
                      "task_id": previous.get("task_id"), "platform_status": previous.get("platform_status"),
                      "updated": utc(), "checks": previous.get("checks", 0)+1,
                      "report_path": str(self.report_path(job)),
                      "evidence": {"executor_events": str(self.memory.events),
                                   "worker_output": payload["output"]}}
            self.memory.append("poll_started", {"job_id": job["id"], "task_id": report["task_id"],
                                                "checks": report["checks"], "previous_status": report["platform_status"]})
            try:
                intent_path = Path(job["manifest"]).with_name("submission.json")
                intent = read_json(intent_path) if intent_path.exists() else None
                if intent and intent.get("task_id"):
                    expected = f"llmgo-rsi:{payload['campaign_id']}:{job['id']}:{job['sha256']}"
                    if intent["description"] != expected:
                        raise ValueError("Submission intent belongs to a different manifest")
                else:
                    intent = self.transport.submit(job["manifest"])
                report["task_id"] = int(intent["task_id"])
                task = self.transport.get(intent)
                self.memory.append("platform_observation", {"job_id": job["id"], "task": task})
                report["platform_status"] = task["status"]
                if task["status"] in TERMINAL_FAILURE:
                    report.update(status="failed", summary=f"Task {report['task_id']} ended: {task['status']}")
                    self._collect_logs(job, intent, report)
                elif task["status"] != "Succeeded":
                    report.update(status="waiting", summary=f"Task {report['task_id']}: {task['status']}")
                else:
                    self._collect_result(payload, report)
                    if report["status"] in FINAL:
                        self._collect_logs(job, intent, report)
            except SubmissionUncertain as exc:
                self.memory.append("reconciliation_required", {"job_id": job["id"], "error": str(exc)})
                report.update(status="submission_uncertain", summary="Submission uncertain; reconcile existing intent, never blindly resubmit")
            except (CctlCommandError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                # A failed observation is not a terminal task. Keep its existing
                # intent/ID, back off, and poll the same handle next time.
                self.memory.append("observation_error", {"job_id": job["id"], "error_type": type(exc).__name__, "error": str(exc)})
                report.update(status="retrying", summary=f"Observation unavailable ({type(exc).__name__}); retain existing task and retry")
            stable = (previous.get("status"), previous.get("platform_status")) == (report["status"], report["platform_status"])
            delay = min(max_interval, max(interval, previous.get("interval_seconds", interval))*1.5) if stable else interval
            report.update(updated=utc(), interval_seconds=delay, next_check_at=self.clock()+delay)
            atomic_json(self.report_path(job), report)
            self.memory.append("poll_finished", {"job_id": job["id"], "task_id": report["task_id"],
                "checks": report["checks"], "status": report["status"], "platform_status": report["platform_status"],
                "interval_seconds": delay, "next_check_at": report["next_check_at"], "report_path": report["report_path"]})
            if not stable:
                self.memory.append("lifecycle_transition", {"job_id": job["id"], "status": report["status"],
                                                           "platform_status": report["platform_status"]})
            return report

    def _collect_result(self, payload, report):
        path = inside(self.root, Path(payload["output"]) / "result.json")
        if not path.is_file():
            report.update(status="waiting_for_artifact", summary="Platform succeeded; waiting for worker result on shared storage")
            return
        result = read_json(path)
        artifact = inside(self.root, result["artifact"])
        if result["manifest_sha256"] != report["manifest_sha256"] or file_hash(artifact) != result["artifact_sha256"]:
            self.memory.append("artifact_invalid", {"job_id": report["job_id"], "result_path": str(path)})
            report.update(status="artifact_invalid", summary="Worker result identity/hash mismatch; retain evidence and do not score it")
            return
        report.update(status="collected", summary="Platform succeeded and worker artifact hash verified",
                      artifact={"path": str(artifact), "sha256": result["artifact_sha256"], "job_id": payload["job_id"]},
                      result_path=str(path), worker_wall_seconds=result.get("worker_wall_seconds"),
                      allocated_gpu_count=result.get("allocated_gpu_count"))

    def _collect_logs(self, job, intent, report):
        # Worker full logs already live on the shared filesystem. Platform tail
        # collection is best effort once at termination, never every poll.
        if not hasattr(self.transport, "logs"):
            return
        path = self.report_path(job).with_name("platform.log")
        try:
            text = self.transport.logs(intent, f"job-rsi-{intent['task_id']}-main-0")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(redact(text), encoding="utf-8")
            report["evidence"]["platform_log"] = str(path)
            report["evidence"]["platform_log_scope"] = "cctl page-size 200; worker files retain full application logs"
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            self.memory.append("log_collection_error", {"job_id": job["id"], "error": str(exc)})
            report["evidence"]["platform_log_unavailable"] = True

    def run(self, *, once=False, max_checks=None, sleeper=time.sleep):
        control = MemoryStore(self.root)
        checks = 0
        with self.memory.lock("service"):
            while True:
                state = control.state()
                active = state.get("active_job")
                heartbeat = {"pid": os.getpid(), "host": socket.gethostname(), "updated": utc(),
                             "active_job": active, "role": "executor", "gpu_count": 0}
                atomic_json(self.memory.root / "heartbeat.json", heartbeat)
                if active:
                    self.advance(state["jobs"][active])
                checks += 1
                if once or (max_checks is not None and checks >= max_checks):
                    return self.report(state["jobs"][active]) if active else {"status": "idle"}
                sleeper(min(60., self.interval))

    def confirm_failed(self, job):
        """Explicit retry guard, independent of stale status in campaign.json."""
        with self.memory.lock("lifecycle"):
            self._manifest(job)
            intent = read_json(Path(job["manifest"]).with_name("submission.json"))
            task = self.transport.get(intent)
            self.memory.append("retry_guard", {"job_id": job["id"], "task": task})
            if task["status"] not in TERMINAL_FAILURE:
                raise ValueError("Original task is not confirmed terminal")
