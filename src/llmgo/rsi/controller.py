"""Persistent experiment state machine. All expensive work is a GPU job."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from llmgo.harness.skills import validate_cards
from llmgo.remote.lifecycle import JobExecutor
from llmgo.training.ppo import training_config
from llmgo.training.rewards import validate_reward
from llmgo.training.rollout import load_curriculum, seed_curriculum, write_curriculum
from .artifacts import source_snapshot, verify_source, write_manifest
from .contracts import SCHEMA, REPO, atomic_json, digest, file_hash, identifier, inside, protocol, read_json, utc
from .evaluator import engine_identity
from .ladder import batch_stats, compare, mastered, opponent
from .memory import CampaignLocked, MemoryStore


class Controller:
    def __init__(self, root, *, executor=None, engine_probe=engine_identity):
        self.root = Path(root)
        self.memory = MemoryStore(self.root)
        self.state = self.memory.state()
        self.site = self.state["site"]
        inside(self.site["user_root"], self.root)
        self.lifecycle = JobExecutor(self.root, self.site, transport=executor)
        self.engine_probe = engine_probe

    @property
    def executor(self):
        """Transport injection compatibility; research code consumes reports."""
        return self.lifecycle.transport

    @executor.setter
    def executor(self, value):
        self.lifecycle.transport = value

    @classmethod
    def create(cls, site, campaign_id, *, repo=None):
        campaign_id = identifier(campaign_id)
        root = inside(site["user_root"], Path(site["campaign_root"]) / campaign_id)
        if root.exists():
            raise FileExistsError("Campaign already exists; use run/status to resume")
        root.mkdir(parents=True)
        repo = Path(repo or site["repo"])
        source = source_snapshot(repo, root / "source")
        cards = validate_cards(read_json(repo / "src/llmgo/harness/skills/cards.json"))
        skills = root / "candidates/initial/skills.json"
        atomic_json(skills, list(cards))
        state = {
            "schema_version": SCHEMA, "campaign_id": campaign_id, "created": utc(), "site": site,
            "source": source, "phase": "BASELINE", "round": 0, "current_level": 1,
            "highest_cleared_level": 0, "nodes_base": None, "active_job": None, "jobs": {},
            "baselines": [], "candidate": None, "workspace": None, "attempts": [],
            "incumbent": {"id": "initial", "model_path": site["base_model"], "model_sha256": None,
                          "skills_path": str(skills), "skills_sha256": file_hash(skills)},
            "acceptance": {"min_improvement": .05, "confidence": .95, "max_comparison_batches": 2},
            "level_certificates": [], "meta_stalls": 0,
        }
        memory = MemoryStore(root)
        memory.save_state(state)
        memory.append("campaign_created", {"campaign_id": campaign_id, "source": source["sha256"],
                                           "constraints": "RL/skills only; 20 games; thinking; 200 plies; no runtime oracle"})
        return cls(root)

    def save(self):
        self.state["updated"] = utc()
        self.memory.save_state(self.state)

    def metrics(self, ref):
        if file_hash(ref["path"]) != ref["sha256"]:
            raise ValueError("Stored evaluation evidence changed")
        return read_json(ref["path"])

    def _candidate(self):
        return self.state["candidate"] or self.state["incumbent"]

    def _new_job(self, kind, role, candidate, *, batch_index=0, level=None, comparison=None, resume_from=None):
        level = level or self.state["current_level"]
        spec = opponent(level, nodes_base=self.state["nodes_base"])
        serial = len(self.state["jobs"]) + 1
        job_id = f"j{serial:06d}"
        folder = inside(self.site["user_root"], self.root / "jobs" / job_id)
        payload = {
            "schema_version": SCHEMA, "campaign_id": self.state["campaign_id"], "job_id": job_id,
            "kind": kind, "role": role, "candidate_id": candidate["id"],
            "site": self.site, "source": self.state["source"],
            "output": str(folder / "output"), "model_path": candidate["model_path"],
            "expected_model_sha256": candidate.get("model_sha256"),
            "skills_path": candidate["skills_path"], "skills_sha256": candidate["skills_sha256"],
        }
        if kind == "evaluate":
            seed_schedule = comparison["seed_schedule"] if comparison else [100000*level+100*batch_index+i for i in range(10)]
            comparison_id = comparison["comparison_id"] if comparison else f"{self.state['incumbent']['id']}:L{level}:b{batch_index}"
            payload.update(protocol=protocol(spec, self.site["engine"]),
                           engine_identity=self.engine_probe(self.site["engine"], spec),
                           seed_schedule=seed_schedule, comparison_id=comparison_id)
        else:
            payload.update(training=candidate["training"], reward=candidate["reward"],
                           curriculum_path=candidate["curriculum_path"],
                           curriculum_sha256=candidate["curriculum_sha256"])
            if resume_from:
                payload["resume_from"] = str(inside(self.site["user_root"], resume_from))
        manifest_path = folder / "manifest.json"
        manifest = write_manifest(manifest_path, payload)
        self.state["jobs"][job_id] = {"id": job_id, "manifest": str(manifest_path),
                                      "sha256": manifest["sha256"], "role": role, "batch_index": batch_index,
                                      "level": level, "candidate_id": candidate["id"], "status": "PREPARED"}
        self.state.update(active_job=job_id, phase="WAITING")
        self.save()  # durable before the external submission
        return self.state["jobs"][job_id]

    def _poll(self):
        job = self.state["jobs"][self.state["active_job"]]
        try:
            report = (self.lifecycle.report(job) if self.site.get("executor_mode") == "external"
                      else self.lifecycle.advance(job))
        except CampaignLocked:
            return {"status": "waiting", "job_id": job["id"], "detail": "Executor is checking this campaign"}
        old = (job.get("task_id"), job["status"], job.get("executor_status"))
        job.update(task_id=report.get("task_id"), status=report.get("platform_status") or job["status"],
                   executor_status=report["status"], executor_report=report["report_path"])
        changed = old != (job["task_id"], job["status"], job["executor_status"])
        if changed:
            # Only semantic transitions enter research Memory, not check counts,
            # timestamps, shell output, or repeated Running/Queued observations.
            self.memory.append("executor_transition", {"job_id": job["id"], "task_id": job["task_id"],
                "status": report["status"], "platform_status": job["status"], "report_path": report["report_path"]})
            self.save()
        if report["status"] == "submission_uncertain":
            return {"status": "submission_uncertain", "job_id": job["id"], "detail": report["summary"],
                    "report_path": report["report_path"]}
        if report["status"] == "artifact_invalid":
            raise ValueError("Worker result identity/hash mismatch; see Executor report")
        if report["status"] == "failed":
            self.state.update(phase="FAILED", failure={"job_id": job["id"], "task_id": job["task_id"],
                                                     "status": job["status"], "executor_report": report["report_path"]})
            self.memory.append("experiment_failed", self.state["failure"])
            self.memory.handoff({"completed": [], "waiting": [], "next": "Inspect failure and resume-failed explicitly",
                                 "failure": self.state["failure"]})
            self.save()
            return {**self.state["failure"], "status": "failed", "platform_status": job["status"]}
        if report["status"] != "collected":
            return {"status": "waiting_for_artifact" if report["status"] == "waiting_for_artifact" else "waiting",
                    "task_id": job["task_id"], "platform_status": job["status"], "report_path": report["report_path"]}
        wrapper = read_json(job["manifest"])
        ref = report["artifact"]
        inside(self.site["user_root"], ref["path"])
        if digest(wrapper["payload"]) != job["sha256"] or wrapper["sha256"] != job["sha256"]:
            raise ValueError("Manifest changed before collection")
        job["worker_wall_seconds"] = report.get("worker_wall_seconds")
        job["allocated_gpu_count"] = report.get("allocated_gpu_count")
        self._consume(job, ref, wrapper["payload"])
        job["handled"] = True
        self.state["active_job"] = None
        self.save()
        return {"status": "collected", "job_id": job["id"], "phase": self.state["phase"]}

    def _consume(self, job, ref, payload):
        role = job["role"]
        if role == "train":
            result = self.metrics(ref)
            candidate = self.state["candidate"]
            inside(self.site["user_root"], result["model_path"])
            if (result.get("training") != payload["training"] or result.get("sft_enabled") is not False
                    or result.get("parent_identity", {}).get("sha256") != payload["expected_model_sha256"]
                    or not result.get("identity", {}).get("sha256")
                    or result["identity"]["sha256"] == payload["expected_model_sha256"]):
                raise ValueError("Training result does not match the frozen parent/PPO configuration")
            candidate.update(model_path=result["model_path"], model_sha256=result["identity"]["sha256"],
                             checkpoint_ref=ref)
            self.state["phase"] = "CANDIDATE_EVAL"
            return
        metrics = self.metrics(ref)
        batch_stats(metrics)
        expected = {"candidate_id": payload["candidate_id"], "skills_sha256": payload["skills_sha256"],
                    "source_identity": payload["source"]["sha256"], "comparison_id": payload["comparison_id"],
                    "protocol_identity": digest(payload["protocol"]), "opponent_identity": digest(payload["engine_identity"]),
                    "seed_schedule": payload["seed_schedule"]}
        if any(metrics.get(k) != v for k, v in expected.items()):
            raise ValueError("Evaluation result does not match the frozen manifest")
        if payload.get("expected_model_sha256") and metrics["model_identity"]["sha256"] != payload["expected_model_sha256"]:
            raise ValueError("Evaluated the wrong checkpoint")
        self.memory.append("evaluation", {"ref": ref, "role": role, "level": job["level"],
                                          "candidate_id": job["candidate_id"], "statistics": metrics["statistics"]})
        if role in ("baseline", "comparison_baseline"):
            self.state["incumbent"]["model_sha256"] = metrics["model_identity"]["sha256"]
            self.state["baselines"].append(ref)
            self.state["phase"] = "CANDIDATE_EVAL" if role == "comparison_baseline" else "LEVEL_CHECK"
        elif role == "candidate":
            self.state["candidate"]["evaluations"].append(ref)
            self.state["phase"] = "REVIEW"
        elif role == "regression_baseline":
            self.state["candidate"]["regression_baseline"] = ref
            self.state["phase"] = "REGRESSION_EVAL"
        elif role == "regression_candidate":
            self.state["candidate"]["regression_candidate"] = ref
            self.state["phase"] = "REGRESSION_REVIEW"
        else:
            raise ValueError("Unknown evaluation role")

    def _level_check(self):
        refs = self.state["baselines"]
        if not refs:
            self.state["phase"] = "BASELINE"
        elif mastered(self.metrics(refs[-1])):
            if len(refs) >= 2 and mastered(self.metrics(refs[-2])):
                level = self.state["current_level"]
                if level == 8 and self.state["nodes_base"] is None:
                    nodes = [n for ref in refs[-2:] for game in self.metrics(ref)["games"]
                             for n in game.get("engine_nodes", []) if isinstance(n, int)]
                    if not nodes:
                        self.state.update(phase="NEEDS_INPUT", needs_input="L8 node telemetry missing", resume_phase="LEVEL_CHECK")
                        self.save()
                        return
                    self.state["nodes_base"] = max(1000, 2*max(nodes))
                certificate = {"level": level, "incumbent": self.state["incumbent"].copy(), "batches": refs[-2:]}
                self.state["level_certificates"].append(certificate)
                self.state["highest_cleared_level"] = max(level, self.state["highest_cleared_level"])
                self.state["current_level"] += 1
                self.state["baselines"] = []
                self.state["phase"] = "BASELINE"
                self.memory.append("opponent_level_promoted", certificate)
            else:
                self.state["phase"] = "BASELINE"
        else:
            self.state["phase"] = "PLANNING"
        self.save()

    def workspace(self):
        if self.state.get("workspace"):
            return Path(self.state["workspace"])
        self.state["round"] += 1
        folder = inside(self.site["user_root"], Path(self.site["worktree_root"]) /
                        self.state["campaign_id"] / f"round-{self.state['round']:06d}")
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copy2(self.state["incumbent"]["skills_path"], folder / "skills.json")
        write_curriculum(folder / "curriculum.jsonl", seed_curriculum())
        for name in (".agents", "docs"):
            source = Path(self.state["source"]["root"]) / name
            if source.exists():
                shutil.copytree(source, folder / name)
        atomic_json(folder / "context.json", self.memory.context())
        self.state["workspace"] = str(folder)
        self.save()
        return folder

    def prepare(self, proposal_path):
        if self.state["phase"] != "PLANNING":
            raise ValueError("A proposal can only be prepared after a completed incumbent baseline")
        proposal_path = inside(self.site["user_root"], proposal_path)
        proposal = read_json(proposal_path)
        allowed = {"mode", "hypothesis", "evidence_refs", "training", "reward", "repetition_reason"}
        if set(proposal) - allowed or proposal.get("mode") not in ("skills", "rl", "rl_and_skills"):
            raise ValueError("Proposal allows only skills, rl or rl_and_skills")
        if not isinstance(proposal.get("hypothesis"), str) or len(proposal["hypothesis"].strip()) < 5:
            raise ValueError("A concrete experiment hypothesis is required")
        if not isinstance(proposal.get("evidence_refs"), list):
            raise ValueError("evidence_refs must be an array (empty explicitly means no prior evidence)")
        workspace = proposal_path.parent
        if self.state.get("workspace") and workspace != Path(self.state["workspace"]):
            raise ValueError("Use the current registered proposal workspace")
        cards = list(validate_cards(read_json(workspace / "skills.json")))
        mode = proposal["mode"]
        if mode == "rl" and cards != read_json(self.state["incumbent"]["skills_path"]):
            raise ValueError("rl-only cannot change student skills")
        if mode == "skills" and cards == read_json(self.state["incumbent"]["skills_path"]):
            raise ValueError("skills-only proposal made no change")
        if mode == "skills" and (proposal.get("training") or proposal.get("reward")):
            raise ValueError("skills-only cannot submit a training configuration")
        cfg = training_config(proposal.get("training", {})) if mode != "skills" else None
        reward = validate_reward(proposal.get("reward", {})) if mode != "skills" else None
        records = load_curriculum(workspace / "curriculum.jsonl") if mode != "skills" else []
        fingerprint = digest({"parent": self.state["incumbent"]["model_sha256"], "skills": cards,
                              "training": cfg, "reward": reward, "curriculum": records})
        if fingerprint in [a["fingerprint"] for a in self.state["attempts"]] and not proposal.get("repetition_reason"):
            raise ValueError("Repeated experiment; consult Memory and declare a reason to repeat")
        cid = f"r{self.state['round']:06d}-{fingerprint[:10]}"
        folder = inside(self.site["user_root"], self.root / "candidates" / cid)
        if folder.exists():
            raise FileExistsError("Candidate already frozen; resume the existing state")
        folder.mkdir(parents=True)
        atomic_json(folder / "skills.json", cards)
        atomic_json(folder / "proposal.json", proposal)
        candidate = {**self.state["incumbent"], "id": cid, "parent_id": self.state["incumbent"]["id"],
                     "skills_path": str(folder / "skills.json"), "skills_sha256": file_hash(folder / "skills.json"),
                     "mode": mode, "fingerprint": fingerprint, "hypothesis": proposal["hypothesis"],
                     "evidence_refs": proposal["evidence_refs"], "evaluations": []}
        if mode != "skills":
            write_curriculum(folder / "curriculum.jsonl", records)
            candidate.update(training=cfg, reward=reward, curriculum_path=str(folder / "curriculum.jsonl"),
                             curriculum_sha256=file_hash(folder / "curriculum.jsonl"))
        self.state["candidate"] = candidate
        self.state["phase"] = "TRAIN" if mode != "skills" else "CANDIDATE_EVAL"
        self.memory.append("hypothesis", {"candidate": candidate, "proposal": proposal})
        self.save()
        return candidate

    def _decide(self, decision):
        candidate = self.state["candidate"]
        if decision["verdict"] == "accept":
            self.state["incumbent"] = {key: candidate[key] for key in
                                       ("id", "model_path", "model_sha256", "skills_path", "skills_sha256")}
            self.state["baselines"] = candidate["evaluations"][:]
        record = {"candidate_id": candidate["id"], "fingerprint": candidate["fingerprint"],
                  "hypothesis": candidate["hypothesis"], "decision": decision,
                  "evaluations": candidate["evaluations"], "parent_id": candidate["parent_id"]}
        self.state["attempts"].append(record)
        self.memory.append("decision", record)
        if decision["verdict"] == "accept":
            self.memory.append("incumbent_updated", {"incumbent": self.state["incumbent"], "evidence": record})
        self.memory.handoff({"completed": [record], "waiting": [], "next": "level check, then plan next hypothesis",
                             "incumbent": self.state["incumbent"], "current_level": self.state["current_level"]})
        self.state.update(candidate=None, workspace=None, phase="LEVEL_CHECK")
        self.save()

    def _review(self):
        candidate = self.state["candidate"]
        n = len(candidate["evaluations"])
        c = [self.metrics(ref) for ref in candidate["evaluations"]]
        b = [self.metrics(ref) for ref in self.state["baselines"][:n]]
        decision = compare(c, b, min_improvement=self.state["acceptance"]["min_improvement"],
                           confidence=self.state["acceptance"]["confidence"])
        if decision["verdict"] == "inconclusive" and n < self.state["acceptance"]["max_comparison_batches"]:
            self.state["phase"] = "CANDIDATE_EVAL"
            self.save()
        elif decision["verdict"] == "accept" and self.state["highest_cleared_level"] > 0:
            candidate["pending_decision"] = decision
            self.state["phase"] = "REGRESSION_BASELINE"
            self.save()
        else:
            self._decide(decision)

    def step(self, *, proposal=None):
        self.state = self.memory.state()
        verify_source(self.state["source"])
        if self.state.get("active_job") and self.state["phase"] != "FAILED":
            return self._poll()
        phase = self.state["phase"]
        incumbent, candidate = self.state["incumbent"], self.state["candidate"]
        if phase == "BASELINE":
            self._new_job("evaluate", "baseline", incumbent, batch_index=len(self.state["baselines"]))
        elif phase == "LEVEL_CHECK":
            self._level_check()
        elif phase == "PLANNING":
            if proposal:
                self.prepare(proposal)
            else:
                return {"status": "needs_meta", "workspace": str(self.workspace())}
        elif phase == "TRAIN":
            self._new_job("train", "train", candidate)
        elif phase == "CANDIDATE_EVAL":
            index = len(candidate["evaluations"])
            if index >= len(self.state["baselines"]):
                self._new_job("evaluate", "comparison_baseline", incumbent, batch_index=index)
            else:
                comparison = self.metrics(self.state["baselines"][index])
                self._new_job("evaluate", "candidate", candidate, batch_index=index, comparison=comparison)
        elif phase == "REVIEW":
            self._review()
        elif phase == "REGRESSION_BASELINE":
            self._new_job("evaluate", "regression_baseline", incumbent, level=self.state["highest_cleared_level"])
        elif phase == "REGRESSION_EVAL":
            self._new_job("evaluate", "regression_candidate", candidate, level=self.state["highest_cleared_level"],
                          comparison=self.metrics(candidate["regression_baseline"]))
        elif phase == "REGRESSION_REVIEW":
            baseline = batch_stats(self.metrics(candidate["regression_baseline"]))
            tested = batch_stats(self.metrics(candidate["regression_candidate"]))
            decision = {**candidate["pending_decision"], "regression": {"baseline": baseline, "candidate": tested}}
            if tested["score"] < baseline["score"]:
                decision.update(verdict="reject", reason="old_level_regression")
            self._decide(decision)
        elif phase in ("FAILED", "NEEDS_INPUT"):
            return {"status": phase.lower(), "detail": self.state.get("failure") or self.state.get("needs_input")}
        else:
            raise ValueError("Unknown controller phase")
        return {"status": "advanced", "phase": self.state["phase"]}

    def resume_failed(self):
        if self.state["phase"] != "FAILED":
            raise ValueError("Campaign is not in a failed-task state")
        old = self.state["jobs"][self.state["active_job"]]
        self.lifecycle.confirm_failed(old)
        payload = read_json(old["manifest"])["payload"]
        candidate = self.state["incumbent"] if old["role"] in ("baseline", "comparison_baseline", "regression_baseline") else self.state["candidate"]
        comparison = payload if payload["kind"] == "evaluate" else None
        resume = None
        if payload["kind"] == "train":
            fresh = Path(payload["output"]) / "training"
            # A retry can fail before creating its own checkpoint. Keep the prior
            # resume root instead of silently restarting an already trained batch.
            resume = fresh if any(fresh.glob("checkpoint-*/state.json")) else payload.get("resume_from", fresh)
        self.memory.append("explicit_retry", {"previous_job": old, "partial_artifacts_retained": payload["output"]})
        self._new_job(payload["kind"], old["role"], candidate, batch_index=old["batch_index"],
                      level=old["level"], comparison=comparison, resume_from=resume)

    def run(self, *, backend=None, max_steps=None, poll_seconds=None):
        with self.memory.lock():
            iterations = 0
            while max_steps is None or iterations < max_steps:
                result = self.step()
                iterations += 1
                if result["status"] == "needs_meta":
                    if backend is None:
                        from .meta import CodexBackend
                        backend = CodexBackend(self.site, self.memory)
                    workspace = Path(result["workspace"])
                    prompt = (
                        f"读取 {workspace / 'context.json'} 与 Memory，诊断并准备下一次实验。"
                        f"完整证据在 {self.root}。先读 .agents/skills/rsi-memory/SKILL.md。"
                        "在本工作目录编辑 skills.json、curriculum.jsonl，并写 proposal.json。"
                        "proposal 格式见 docs/rsi-operations.md。只允许 skills/rl/rl_and_skills。"
                        "RL仅用hf_ppo与lc0_bootstrap，固定Lc0 critic；细节先读rsi-rl-review技能。"
                        "复查失败指纹，说明主要假设和证据。没有真实进步不能扶正。"
                        "最后按指定 JSON 返回 ready 和 proposal 路径，或 needs_input 与具体缺失项。"
                    )
                    answer = backend.turn(workspace, prompt)
                    self.memory.append("meta_result", answer)
                    if answer["status"] == "needs_input":
                        self.state.update(phase="NEEDS_INPUT", needs_input=answer["summary"], resume_phase="PLANNING")
                        self.save()
                        return answer
                    try:
                        if not isinstance(answer.get("proposal_path"), str):
                            raise ValueError("ready requires a proposal_path string")
                        self.prepare(answer["proposal_path"])
                        self.state["meta_stalls"] = 0
                        self.save()
                    except (ValueError, FileNotFoundError) as exc:
                        self.memory.append("proposal_invalid", {"error": str(exc)})
                        self.state["meta_stalls"] += 1
                        atomic_json(workspace / "context.json", self.memory.context())
                        if self.state["meta_stalls"] >= 3:
                            self.state.update(phase="NEEDS_INPUT", needs_input=f"Three invalid proposals: {exc}", resume_phase="PLANNING")
                        self.save()
                elif result["status"] in ("waiting", "waiting_for_artifact"):
                    time.sleep(min(60, poll_seconds or self.site.get("poll_seconds", 30)))
                elif result["status"] in ("submission_uncertain", "failed", "needs_input"):
                    return result
            return {"status": "step_limit", "phase": self.state["phase"]}
