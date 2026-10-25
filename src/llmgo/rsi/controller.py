"""Persistent, method-agnostic research coordination with an external evaluator."""
from __future__ import annotations
import shutil
import time
from pathlib import Path
from llmgo.remote.lifecycle import JobExecutor
from .artifacts import source_snapshot, verify_source, write_manifest
from .contracts import SCHEMA, atomic_json, digest, file_hash, identifier, inside, model_identity, protocol, read_json, utc
from .evaluator import engine_identity
from .ladder import evaluation_plan, panel_summary, opponent
from .memory import CampaignLocked, MemoryStore

RESEARCH_VERSION = "autonomous-chess-v2"


class Controller:
    def __init__(self, root, *, executor=None, engine_probe=engine_identity):
        self.root = Path(root)
        self.memory = MemoryStore(self.root)
        self.state = self.memory.state()
        if self.state.get("research_version") != RESEARCH_VERSION:
            raise ValueError("Legacy ladder campaign: retain it read-only and init a new autonomous campaign")
        self.site = self.state["site"]
        inside(self.site["user_root"], self.root)
        self.lifecycle = JobExecutor(self.root, self.site, transport=executor)
        self.engine_probe = engine_probe

    @property
    def executor(self):
        return self.lifecycle.transport

    @executor.setter
    def executor(self, value):
        self.lifecycle.transport = value

    @classmethod
    def create(cls, site, campaign_id, *, repo=None):
        campaign_id = identifier(campaign_id)
        root = inside(site["user_root"], Path(site["campaign_root"])/campaign_id)
        if root.exists():
            raise FileExistsError("Campaign already exists; use run/status to resume")
        repo = Path(repo or site["repo"])
        root.mkdir(parents=True)
        source = source_snapshot(repo, root/"source")
        # A reference harness is frozen once for the model-only control track.
        initial_source = source_snapshot(repo, root/"candidates/initial/source")
        cards = root/"candidates/initial/skills.json"
        shutil.copy2(repo/"src/llmgo/harness/skills/cards.json", cards)
        initial = {"id": "initial", "model_path": site["base_model"], "model_sha256": None,
                   "source": initial_source, "skills_path": str(cards), "skills_sha256": file_hash(cards)}
        state = {"schema_version": SCHEMA, "research_version": RESEARCH_VERSION,
                 "campaign_id": campaign_id, "created": utc(), "site": site, "source": source,
                 "phase": "BASELINE", "baseline_games_per_level": 10, "baseline_levels": [1,2,3,4],
                 "round": 0, "active_job": None, "jobs": {},
                 "candidate": None, "workspace": None, "attempts": [], "meta_stalls": 0,
                 "initial": initial, "incumbent": initial.copy(), "working": initial.copy(),
                 "best": None, "recommended": None, "candidate_step": 0, "evaluation_cursor": 0,
                 "evaluation_plan": evaluation_plan(), "pending_evaluations": [], "baseline": []}
        memory = MemoryStore(root)
        memory.save_state(state)
        memory.append("campaign_created", {"campaign_id": campaign_id, "source": source["sha256"],
            "objective": "Open research; L1-L4 baseline 10 games/level, candidate evaluation 20 games/level; final panel L1-L8"})
        return cls(root)

    def save(self):
        self.state["updated"] = utc()
        self.memory.save_state(self.state)

    def upgrade(self, site, repo=None):
        """Explicit operator migration: keep old evidence and initial reference immutable."""
        if self.state["phase"] not in {"NEEDS_INPUT", "BASELINE", "PLANNING"}:
            raise ValueError("Upgrade requires an idle controller at a research/baseline boundary")
        active = self.state.get("active_job")
        if active:
            self.lifecycle.confirm_failed(self.state["jobs"][active])
        if self.state.get("workspace"):
            raise ValueError("Finish or archive the existing research workspace before upgrading")
        if site["base_model"] != self.site["base_model"] or site["engine"] != self.site["engine"]:
            raise ValueError("Upgrade cannot change the benchmark model/engine paths")
        number = len(self.state.get("upgrades", []))+1
        folder = self.root/"upgrades"/f"v{number:04d}"
        atomic_json(folder/"previous-state.json",self.state)
        snapshot = source_snapshot(Path(repo or site["repo"]),folder/"source")
        old_source = self.state["source"]
        working = self.state["working"].copy()
        working.update(id=f"upgrade-{number}", source=snapshot, parent_id=working["id"])
        self.state.setdefault("upgrades", []).append({"utc":utc(),"previous_source":old_source,
            "source":snapshot,"previous_site":self.site,"site":site})
        self.state.update(site=site,source=snapshot,working=working)
        self.site = site
        self.lifecycle = JobExecutor(self.root,site)
        self.memory.append("pipeline_upgraded", {"source":snapshot["sha256"],"gpu_count":site["gpu_count"],
            "previous_state":str(folder/"previous-state.json"),"initial_reference_unchanged":True})
        self.save()
        if active:
            self.resume_failed()
        return {"upgraded":True,"source":snapshot["sha256"],"active_job":self.state.get("active_job")}

    def metrics(self, ref):
        if file_hash(ref["path"]) != ref["sha256"]:
            raise ValueError("Stored evaluation evidence changed")
        return read_json(ref["path"])

    def _new_job(self, kind, role, candidate, *, level=1, track="player", command=None,
                 batch_index=0, comparison=None, resume_from=None):
        job_id = f"j{len(self.state['jobs'])+1:06d}"
        folder = inside(self.site["user_root"], self.root/"jobs"/job_id)
        payload = {"schema_version": SCHEMA, "campaign_id": self.state["campaign_id"], "job_id": job_id,
                   "kind": kind, "role": role, "candidate_id": candidate["id"], "site": self.site,
                   "source": self.state["source"], "candidate_source": candidate["source"],
                   "output": str(folder/"output"), "model_path": candidate["model_path"],
                   "expected_model_sha256": candidate.get("model_sha256"),
                   "skills_path": candidate["skills_path"], "skills_sha256": candidate["skills_sha256"]}
        if kind == "evaluate":
            spec = opponent(level)
            # Keep already-created campaigns on their frozen 20-game protocol.
            short_baseline = role == "baseline" and self.state.get("baseline_games_per_level", 20) == 10
            batch_protocol = protocol(spec, self.site["engine"], baseline=short_baseline)
            # A panel has matched schedules across candidates; repeats use a new batch index.
            actual_engine = self.engine_probe(self.site["engine"], spec)
            pin = {k:v for k,v in actual_engine.items() if k != "spec"}
            if self.state.get("engine_pin") is not None and self.state["engine_pin"] != pin:
                raise ValueError("Official opponent binary/NNUE identity changed since campaign baseline")
            self.state["engine_pin"] = pin
            payload.update(protocol=batch_protocol, track=track, resources={"gpu_count":candidate.get("proposal", {}).get("evaluation_gpus", self.site["gpu_count"])},
                engine_identity=actual_engine,
                seed_schedule=[100000*level+100*batch_index+i for i in range(batch_protocol["games"]//2)],
                comparison_id=f"{'baseline10-v2' if short_baseline else 'panel-v2'}:L{level}:b{batch_index}",
                reference_source=self.state["initial"]["source"])
        elif kind == "research":
            payload["command"] = command
            resources = command.get("resources", {})
            if set(resources) - {"gpu_count", "cpu", "memory_gib"}:
                raise ValueError("Unknown job resource field")
            payload["resources"] = resources
            if resume_from:
                payload["resume_from"] = str(inside(self.site["user_root"], resume_from))
        else:
            raise ValueError("Use research jobs for arbitrary training/labeling/conversion")
        if resume_from:
            payload["resume_from"] = str(inside(self.site["user_root"], resume_from))
        path = folder/"manifest.json"
        manifest = write_manifest(path, payload)
        job = {"id": job_id, "manifest": str(path), "sha256": manifest["sha256"], "role": role,
               "kind": kind, "batch_index": batch_index, "level": level if kind == "evaluate" else None,
               "track": track if kind == "evaluate" else None, "candidate_id": candidate["id"], "status": "PREPARED"}
        self.state["jobs"][job_id] = job
        self.state.update(active_job=job_id, phase="WAITING")
        self.save()
        return job

    def _poll(self):
        job = self.state["jobs"][self.state["active_job"]]
        try:
            report = (self.lifecycle.report(job) if self.site.get("executor_mode") == "external"
                      else self.lifecycle.advance(job))
        except CampaignLocked:
            return {"status": "waiting", "job_id": job["id"]}
        old = (job.get("task_id"), job["status"], job.get("executor_status"))
        job.update(task_id=report.get("task_id"), status=report.get("platform_status") or job["status"],
                   executor_status=report["status"], executor_report=report["report_path"])
        if old != (job["task_id"], job["status"], job["executor_status"]):
            self.memory.append("executor_transition", {"job_id": job["id"], "task_id": job["task_id"],
                "status": report["status"], "platform_status": job["status"], "report_path": report["report_path"]})
            self.save()
        if report["status"] == "submission_uncertain":
            return {"status": "submission_uncertain", "report_path": report["report_path"]}
        if report["status"] == "artifact_invalid":
            raise ValueError("Worker artifact identity/hash mismatch")
        if report["status"] == "failed":
            failure = {"job_id": job["id"], "task_id": job["task_id"], "report_path": report["report_path"]}
            self.memory.append("experiment_failed", failure)
            # Research failure is evidence for the next turn, not a permanent campaign stop.
            self.state["failure"] = failure
            if job["role"] == "baseline":
                self.state.update(phase="NEEDS_INPUT", resume_phase="BASELINE", needs_input="Baseline infrastructure failed; inspect and resume or resume-failed")
            else:
                self._finish({"status": "failed", "failure": failure}, persist=False)
                self.state["active_job"] = None
            self.save()
            return {"status": "needs_input" if job["role"] == "baseline" else "advanced", "phase": self.state["phase"]}
        if report["status"] != "collected":
            return {"status": "waiting", "report_path": report["report_path"]}
        wrapper = read_json(job["manifest"])
        if wrapper["sha256"] != job["sha256"] or digest(wrapper["payload"]) != job["sha256"]:
            raise ValueError("Manifest changed before collection")
        job.update(worker_wall_seconds=report.get("worker_wall_seconds"), allocated_gpu_count=report.get("allocated_gpu_count"))
        self._consume(job, report["artifact"], wrapper["payload"])
        job["handled"] = True
        self.state["active_job"] = None
        self.save()
        return {"status": "collected", "job_id": job["id"], "phase": self.state["phase"]}

    def _consume(self, job, ref, payload):
        result = self.metrics(ref)
        candidate = self.state["initial"] if job["role"] == "baseline" else self.state["candidate"]
        if payload["kind"] == "research":
            candidate.setdefault("artifacts", []).append(ref)
            if result.get("model_path"):
                model_path = str(inside(self.site["user_root"], result["model_path"]))
                identity = model_identity(model_path)
                if result.get("model_identity") and result["model_identity"] != identity:
                    raise ValueError("Research export identity changed before collection")
                candidate.update(model_path=model_path, model_sha256=identity["sha256"])
            self.state["candidate_step"] += 1
            self.state["phase"] = "EXPERIMENT"
            return
        from .ladder import batch_stats
        statistics = batch_stats(result, expected_games=payload["protocol"]["games"])
        expected = {"candidate_id": payload["candidate_id"], "track": payload["track"],
                    "source_identity": payload["source"]["sha256"], "candidate_source_identity": payload["candidate_source"]["sha256"],
                    "comparison_id": payload["comparison_id"], "protocol_identity": digest(payload["protocol"]),
                    "opponent_identity": digest(payload["engine_identity"]), "seed_schedule": payload["seed_schedule"]}
        if any(result.get(k) != v for k,v in expected.items()):
            raise ValueError("Evaluation does not match its registered benchmark/candidate")
        if payload.get("expected_model_sha256") and result["model_identity"]["sha256"] != payload["expected_model_sha256"]:
            raise ValueError("Evaluated the wrong checkpoint")
        candidate["model_sha256"] = result["model_identity"]["sha256"]
        row = {"ref": ref, "role": job["role"], "candidate_id": candidate["id"], "track": job["track"],
               "level": job["level"], "statistics": statistics}
        self.memory.append("evaluation", row, event_id="evaluation-"+job["id"])
        self.state["pending_evaluations"].append(row)
        self.state["evaluation_cursor"] += 1
        self.state["phase"] = "BASELINE" if job["role"] == "baseline" else "EVALUATING"

    def workspace(self):
        if self.state.get("workspace"):
            return Path(self.state["workspace"])
        self.state["round"] += 1
        folder = inside(self.site["user_root"], Path(self.site["worktree_root"])/self.state["campaign_id"]/f"round-{self.state['round']:06d}")
        parent = self.state["working"]
        source_snapshot(Path(parent["source"]["root"]), folder)
        # Editable package + top-level skill override for the initial/reference harness.
        if parent.get("skills_path"):
            shutil.copy2(parent["skills_path"], folder/"skills.json")
        atomic_json(folder/"context.json", self.memory.context())
        atomic_json(folder/"proposal.json", {"hypothesis": "Describe the next experiment", "evidence_refs": [],
            "jobs": [], "evaluation": evaluation_plan({"levels":[1]}), "recommend": False})
        self.state["workspace"] = str(folder)
        self.save()
        return folder

    def prepare(self, proposal_path):
        if self.state["phase"] != "PLANNING":
            raise ValueError("Prepare proposals during PLANNING")
        path = Path(proposal_path)
        if not path.is_absolute() and self.state.get("workspace"):
            path = Path(self.state["workspace"])/path
        path = inside(self.site["user_root"], path)
        workspace = Path(self.state["workspace"]) if self.state.get("workspace") else path.parent
        if path.parent != workspace:
            raise ValueError("Use the registered research workspace")
        proposal = read_json(path)
        digest(proposal)  # Reject non-JSON/nonfinite metadata before creating frozen artifacts.
        if not isinstance(proposal.get("hypothesis"), str) or not proposal["hypothesis"].strip():
            raise ValueError("Describe an experiment hypothesis")
        jobs = proposal.get("jobs", [])
        if not isinstance(jobs, list):
            raise ValueError("jobs must be a list of commands")
        for command in jobs:
            resources = command.get("resources", {}) if isinstance(command,dict) else {}
            if not isinstance(resources,dict) or set(resources)-{"gpu_count","cpu","memory_gib"}:
                raise ValueError("Invalid job resources")
            for key,value in resources.items():
                if type(value) is not int or value < (0 if key=="gpu_count" else 1) or (key=="gpu_count" and value > 8):
                    raise ValueError("Invalid job resources")
            argv = command.get("argv") if isinstance(command, dict) else None
            if not isinstance(argv, list) or not argv or any(not isinstance(x,str) or not x or "\0" in x for x in argv):
                raise ValueError("Each job requires a nonempty argv string array")
        evaluation_gpus = proposal.get("evaluation_gpus", self.site["gpu_count"])
        if type(evaluation_gpus) is not int or not 1 <= evaluation_gpus <= 8:
            raise ValueError("evaluation_gpus must be an integer from 1 to 8")
        batch = proposal.get("evaluation_batch", 0)
        if type(batch) is not int or batch < 0:
            raise ValueError("evaluation_batch must be a nonnegative integer")
        plan = evaluation_plan(proposal.get("evaluation")) if proposal.get("evaluation", True) is not False else None
        model = proposal.get("model_path", self.state["working"]["model_path"])
        if Path(model).resolve() != Path(self.site["base_model"]).resolve():
            model = str(inside(self.site["user_root"], model))
        folder = self.root/"candidates"/f"r{self.state['round']:06d}"
        snapshot = source_snapshot(workspace, folder/"source")
        cards = None
        if (workspace/"skills.json").is_file():
            cards = folder/"skills.json"
            shutil.copy2(workspace/"skills.json", cards)
        candidate = {"id": folder.name, "parent_id": self.state["working"]["id"], "source": snapshot,
                     "model_path": model, "model_sha256": None, "skills_path": str(cards) if cards else None, "skills_sha256": file_hash(cards) if cards else None,
                     "hypothesis": proposal["hypothesis"], "proposal": proposal, "jobs": jobs,
                     "evaluation": plan, "evaluations": [], "fingerprint": digest({"source":snapshot["sha256"], "model":model})}
        atomic_json(folder/"proposal.json", proposal)
        self.state.update(candidate=candidate, candidate_step=0, evaluation_cursor=0,
                          pending_evaluations=[], phase="EXPERIMENT", meta_stalls=0)
        self.memory.append("hypothesis", {"candidate_id": candidate["id"], "hypothesis": candidate["hypothesis"],
                                          "source": snapshot["sha256"], "proposal": proposal})
        self.save()
        return candidate

    def _evaluate_next(self, candidate, plan, role):
        schedule = [(t,l) for t in plan["tracks"] for l in plan["levels"]]
        cursor = self.state["evaluation_cursor"]
        if cursor < len(schedule):
            track, level = schedule[cursor]
            self._new_job("evaluate", role, candidate, track=track, level=level,
                          batch_index=candidate.get("proposal", {}).get("evaluation_batch", 0))
            return True
        return False

    def _finish(self, extra=None, *, persist=True):
        candidate = self.state["candidate"]
        if not candidate:
            return
        rows = self.state["pending_evaluations"]
        panel = panel_summary(rows)
        previous = self.state.get("best")
        score = panel["player"]["score"]
        improved = score is not None and (previous is None or previous.get("score") is None or score > previous["score"])
        record = {"candidate_id": candidate["id"], "parent_id": candidate["parent_id"],
                  "hypothesis": candidate["hypothesis"], "fingerprint": candidate["fingerprint"],
                  "panel": panel, "evaluations": rows, "new_best": improved, **(extra or {})}
        if improved:
            self.state["best"] = {"candidate": candidate.copy(), "score": score, "panel": panel,
                                  "note": "Observed best; confirm with independent repeats before claiming improvement"}
            self.state["incumbent"] = candidate.copy()
        candidate["evaluations"] = rows
        # Research may branch from failures. A job failure can leave missing model paths;
        # retain its code but keep the last runnable checkpoint for the next turn.
        if record.get("status") == "failed":
            candidate["model_path"] = self.state["working"]["model_path"]
            candidate["model_sha256"] = self.state["working"].get("model_sha256")
        self.state["working"] = candidate.copy()
        if candidate["proposal"].get("recommend"):
            self.state["recommended"] = {"candidate_id": candidate["id"], "reason": candidate["proposal"].get("recommendation", candidate["hypothesis"])}
        self.state["attempts"].append(record)
        self.memory.append("decision", record, event_id="decision-"+candidate["id"])
        self.memory.handoff({"completed": [record], "next": "Read evidence; choose any next research method",
                             "working": candidate["id"], "best": self.state.get("best", {}).get("score") if self.state.get("best") else None})
        self.state.update(candidate=None, workspace=None, phase="PLANNING")
        if persist:
            self.save()

    def step(self, *, proposal=None):
        self.state = self.memory.state()
        verify_source(self.state["source"])
        if self.state.get("active_job") and self.state["phase"] != "NEEDS_INPUT":
            return self._poll()
        phase = self.state["phase"]
        if phase == "BASELINE":
            baseline_plan = evaluation_plan({"levels":self.state.get("baseline_levels", list(range(1,9)))})
            if not self._evaluate_next(self.state["initial"], baseline_plan, "baseline"):
                panel = panel_summary(self.state["pending_evaluations"])
                self.state["baseline"] = self.state["pending_evaluations"][:]
                self.state["baseline_panel"] = panel
                self.state["best"] = ({"candidate": self.state["initial"].copy(), "score": panel["player"]["score"], "panel": panel}
                                      if panel["player"]["complete"] else None)
                self.state.update(phase="PLANNING", pending_evaluations=[], evaluation_cursor=0)
                self.save()
        elif phase == "PLANNING":
            if proposal:
                self.prepare(proposal)
            else:
                return {"status": "needs_meta", "workspace": str(self.workspace())}
        elif phase == "EXPERIMENT":
            candidate = self.state["candidate"]
            i = self.state["candidate_step"]
            if i < len(candidate["jobs"]):
                self._new_job("research", "research", candidate, command=candidate["jobs"][i])
            else:
                self.state.update(phase="EVALUATING", evaluation_cursor=0)
                self.save()
        elif phase == "EVALUATING":
            candidate = self.state["candidate"]
            if not candidate["evaluation"] or not self._evaluate_next(candidate, candidate["evaluation"], "candidate"):
                self._finish()
        elif phase == "NEEDS_INPUT":
            return {"status": "needs_input", "detail": self.state.get("needs_input")}
        else:
            raise ValueError("Unknown controller phase")
        return {"status": "advanced", "phase": self.state["phase"]}

    def resume_failed(self):
        job_id = self.state.get("active_job") or next((k for k,v in reversed(self.state["jobs"].items()) if v.get("executor_status")=="failed"), None)
        if not job_id:
            raise ValueError("No failed task to resume")
        old = self.state["jobs"][job_id]
        payload = read_json(old["manifest"])["payload"]
        # After an explicit pipeline upgrade, old manifests still retain their
        # original site identity. Validate the terminal task using that identity.
        guard = self.lifecycle if payload["site"] == self.site else JobExecutor(
            self.root, payload["site"], transport=self.executor)
        guard.confirm_failed(old)
        if old["role"] != "baseline" and not self.state.get("candidate"):
            raise ValueError("Experiment was handed back to Meta-Agent; submit a new proposal with explicit resume paths")
        candidate = self.state["initial"] if old["role"] == "baseline" else self.state["candidate"]
        self.memory.append("explicit_retry", {"previous_job": old, "partial_artifacts_retained": payload["output"]})
        self._new_job(payload["kind"], old["role"], candidate, level=old.get("level") or 1,
                      track=old.get("track") or "player", command=payload.get("command"),
                      batch_index=old["batch_index"], resume_from=payload["output"])

    def recover_meta(self):
        if self.state["phase"] != "PLANNING" or self.state.get("active_job") or not self.state.get("workspace"):
            raise ValueError("Recover Meta only at an idle PLANNING boundary with a registered workspace")
        from .meta import CodexBackend
        backend = CodexBackend(self.site, self.memory)
        # Preserve both the prior state and the failing session before recovery.
        folder = self.root / "recoveries" / ("meta-" + str(self.state["round"]) + "-" + digest(backend.session)[:16])
        if not folder.exists():
            atomic_json(folder/"campaign.json", self.state)
            atomic_json(folder/"meta-session.json", backend.session)
        answer = backend.recover_completed(Path(self.state["workspace"]))
        candidate = self.prepare(answer["proposal_path"])
        self.memory.append("meta_proposal_recovered", {"candidate_id": candidate["id"],
            "proposal_path": answer["proposal_path"], "backup": str(folder)})
        return {"recovered": True, "candidate_id": candidate["id"], "phase": self.state["phase"],
                "evaluation": candidate["evaluation"], "inference_performed": False}

    def resume(self):
        """An explicit operator resume authorizes a new Meta turn, not cached output."""
        if self.state["phase"] != "NEEDS_INPUT":
            raise ValueError("resume only clears an explicitly resolved NEEDS_INPUT state")
        self.memory.append("operator_resume", {"previous_reason": self.state.get("needs_input"),
            "previous_phase": self.state.get("resume_phase", "PLANNING"),
            "meta_request_revision": self.state.get("meta_request_revision", 0) + 1})
        self.state.update(phase=self.state.get("resume_phase", "PLANNING"), active_job=None,
                          needs_input=None, meta_stalls=0,
                          meta_request_revision=self.state.get("meta_request_revision", 0) + 1)
        self.save()
        return {"phase": self.state["phase"]}

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
                    prompt = (f"读取 {workspace/'context.json'}、Memory 和 .agents/skills/rsi-memory/SKILL.md。"
                        f"完整证据在 {self.root}。只能通过RL和改进skills提升Qwen3.5-4B，可单独或组合使用。"
                        "禁止SFT（监督微调），包括预热、辅助SFT损失、行为克隆和监督蒸馏；使用SFT，最终得分为0分。"
                        "在此范围内自主修改本研究副本的RL、reward、数据、Harness、prompt、skills、工具或配置，现有 PPO/LC0 只是可用起点。"
                        "教师只能用于RL奖励/价值、分析和提炼skills，不能作为SFT答案目标。最终会对战固定 L1-L8 Fairy-Stockfish，"
                        "最低档仍是 skill -9/depth5/0.05秒；只刷简单档没有用。"
                        "新campaign只先跑L1-L4基线，每档10局、白黑各5局，共40局；之后进入研究。"
                        "研究每轮可只测一个难度，后续评测每档20局；不必每次全八档。"
                        "每任务最多8 GPU，默认8卡评测并行；训练用verl PPO+LC0（默认7卡actor加1卡教师），禁止SFT。可改RL配方。 在workspace运行 python scripts/request_evaluation.py --level L2 --gpus 8 可保留既有训练任务并把proposal.json设为只测L2；"
                        "再返回ready及proposal绝对路径，由Executor提交。--track model测模型对照，--full-panel登记L1-L8完整评测。"
                        "评测套件/裁判/正式记录不可修改。提交协议及任意计算任务格式见 docs/rsi-operations.md。"
                        "可以通过 research job 调用受保护 source/scripts/evaluate.py 做快速或完整测试。"
                        "不要求本轮成功才能继续。请现在实际调用工具读取文件并开展研究，不要仅宣布准备开始就结束。"
                        "保存实验结论，最后写 proposal.json 并返回JSON：status=ready、summary和proposal_path绝对路径；"
                        "只有确实需要用户补充信息或权限时返回needs_input，并明确说明实际障碍和需要回答的问题。"
                        f"操作员恢复版本：{self.state.get('meta_request_revision', 0)}。")
                    answer = backend.turn(workspace, prompt)
                    self.memory.append("meta_result", answer)
                    if answer["status"] == "needs_input":
                        self.state.update(phase="NEEDS_INPUT", needs_input=answer["summary"], resume_phase="PLANNING")
                        self.save()
                        return answer
                    try:
                        self.prepare(answer["proposal_path"])
                    except (ValueError, FileNotFoundError, TypeError) as exc:
                        self.memory.append("proposal_invalid", {"error": str(exc)})
                        self.state["meta_stalls"] += 1
                        atomic_json(workspace/"context.json", self.memory.context())
                        if self.state["meta_stalls"] >= 3:
                            self.state.update(phase="NEEDS_INPUT", needs_input=str(exc), resume_phase="PLANNING")
                        self.save()
                elif result["status"] == "waiting":
                    time.sleep(min(60, poll_seconds or self.site.get("poll_seconds", 120)))
                elif result["status"] in ("submission_uncertain", "needs_input"):
                    return result
            return {"status": "step_limit", "phase": self.state["phase"]}
