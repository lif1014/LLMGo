"""Persistent, method-agnostic research coordination with an external evaluator."""
from __future__ import annotations
import shutil
import tempfile
import time
from pathlib import Path
from llmgo.remote.lifecycle import JobExecutor
from .artifacts import source_snapshot, verify_source, write_manifest
from .contracts import SCHEMA, VALIDATION_GAMES, atomic_json, digest, file_hash, identifier, inside, model_identity, protocol, read_json, utc
from .evaluator import engine_identity
from .ladder import evaluation_plan, panel_summary, opponent
from .memory import CampaignLocked, MemoryStore
from .meta_protocol import MetaFailure
from .research_control import ResearchPaused, read_control, require_research

RESEARCH_VERSION = "chess-grpo-validation-v1"
PHASE1_SCOPE = "phase1_rl_only"


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
        # Freeze the same Chess-R1 baseline for model-only comparisons.
        initial_source = source_snapshot(repo, root/"candidates/initial/source")
        cards = root/"candidates/initial/skills.json"
        atomic_json(cards, [])
        initial = {"id": "initial", "model_path": site["base_model"], "model_sha256": None,
                   "source": initial_source, "skills_path": str(cards), "skills_sha256": file_hash(cards)}
        state = {"schema_version": SCHEMA, "research_version": RESEARCH_VERSION,
                 "research_scope": PHASE1_SCOPE,
                 "campaign_id": campaign_id, "created": utc(), "site": site, "source": source,
                 "phase": "BASELINE", "baseline_games_per_level": VALIDATION_GAMES, "baseline_levels": [2,3],
                 "round": 0, "active_job": None, "jobs": {},
                 "candidate": None, "workspace": None, "attempts": [], "meta_stalls": 0,
                 "initial": initial, "incumbent": initial.copy(), "working": initial.copy(),
                 "best": None, "recommended": None, "candidate_step": 0, "evaluation_cursor": 0,
                 "evaluation_plan": evaluation_plan(), "pending_evaluations": [], "baseline": []}
        memory = MemoryStore(root)
        memory.save_state(state)
        memory.append("campaign_created", {"campaign_id": campaign_id, "source": source["sha256"],
            "objective": "GRPO research; L2/L3 development validation, then one held-out DeepSeek Flash non-thinking final test"})
        return cls(root)

    def save(self):
        self.state["updated"] = utc()
        self.memory.save_state(self.state)
        self._publish_control_status()

    def _publish_control_status(self):
        # Small, read-only web compatibility artifact. Older Web processes can
        # show recovery without loading a multi-megabyte campaign or raw log.
        atomic_json(self.root/"controller-status.json", {"schema_version":1,
            **{key:self.state.get(key) for key in ("campaign_id","phase","round","active_job",
                                                  "controller_recovery","training_review","final_checkpoint","updated")},
            "research_control": read_control(self.root)})

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
        if not self.state['jobs'] and site.get('training_review'):
            self.state['baseline_games_per_level'] = VALIDATION_GAMES
        self.site = site
        self.lifecycle = JobExecutor(self.root,site)
        self.memory.append("pipeline_upgraded", {"source":snapshot["sha256"],"gpu_count":site["gpu_count"],
            "previous_state":str(folder/"previous-state.json"),"initial_reference_unchanged":True})
        self.save()
        if active:
            self.resume_failed()
        return {"upgraded":True,"source":snapshot["sha256"],"active_job":self.state.get("active_job")}

    def upgrade_validation(self, repo=None):
        """Migrate the protected referee while preserving paused RL research."""
        if (not read_control(self.root)['paused'] or self.state['phase']!='PLANNING'
                or self.state.get('active_job') or self.state.get('candidate') or self.state.get('training_review')):
            raise ValueError('Validation migration requires paused idle PLANNING')
        session=read_json(self.root/'meta-session.json') if (self.root/'meta-session.json').exists() else {}
        if session.get('process') or session.get('status') not in {None,'paused','completed'}:
            raise ValueError('Meta must be stopped before validation migration')
        workspace=Path(self.state['workspace']) if self.state.get('workspace') else None
        if workspace and session.get('request_key'):
            result=self.root/'meta-requests'/session['request_key']/'result.json'
            if result.exists():raise ValueError('Resolve completed handoff before migrating its context')
        folder=self.root/'upgrades'/f"v{len(self.state.get('upgrades',[]))+1:04d}"
        atomic_json(folder/'previous-state.json',self.state)
        if workspace: shutil.copy2(workspace/'context.json',folder/'previous-context.json')
        snapshot=source_snapshot(Path(repo or self.site['repo']),folder/'source')
        policy={'games_per_level':VALIDATION_GAMES,'white_per_level':20,'black_per_level':20,
                'vllm_max_num_seqs':20,'max_parallel_games':80}
        self.state.setdefault('upgrades',[]).append({'utc':utc(),'kind':'validation_policy',
            'previous_source':self.state['source'],'source':snapshot,'site':self.site,
            'workspace_preserved':str(workspace) if workspace else None})
        self.state.update(source=snapshot,validation_policy=policy,
            meta_request_revision=self.state.get('meta_request_revision',0)+1)
        self.memory.append('validation_policy_upgraded',{'policy':policy,'source_sha256':snapshot['sha256'],
            'previous_state':str(folder/'previous-state.json'),'research_paused':True,
            'historical_results_unchanged':True,'executor_site_unchanged':True})
        self.save()
        if workspace:atomic_json(workspace/'context.json',self.memory.context())
        return {'policy':policy,'source_sha256':snapshot['sha256'],'research_paused':True,
                'workspace':str(workspace) if workspace else None}

    def metrics(self, ref):
        if file_hash(ref["path"]) != ref["sha256"]:
            raise ValueError("Stored evaluation evidence changed")
        return read_json(ref["path"])

    def _new_job(self, kind, role, candidate, *, level=2, track="player", command=None,
                 batch_index=0, comparison=None, resume_from=None, panel_levels=None):
        require_research(self.root)
        job_id = f"j{len(self.state['jobs'])+1:06d}"
        folder = inside(self.site["user_root"], self.root/"jobs"/job_id)
        run_output = inside(self.site["user_root"],
                            Path(self.site["repo"])/"var/runs"/f"{self.state['campaign_id']}-{job_id}")
        payload = {"schema_version": SCHEMA, "campaign_id": self.state["campaign_id"], "job_id": job_id,
                   "research_scope": self.state.get("research_scope", PHASE1_SCOPE),
                   "kind": kind, "role": role, "candidate_id": candidate["id"], "site": self.site,
                   "source": self.state["source"], "candidate_source": candidate["source"],
                   "output": str(run_output), "model_path": candidate["model_path"],
                   "expected_model_sha256": candidate.get("model_sha256"),
                   "skills_path": candidate["skills_path"], "skills_sha256": candidate["skills_sha256"]}
        payload['lineage_steps'] = candidate.get('lineage_steps',0)
        payload['lineage_gpu_hours'] = candidate.get('lineage_gpu_hours',0.)
        payload['parent_curve'] = candidate.get('curve')
        payload['engine_pin'] = self.state.get('engine_pin')
        if kind in {'evaluate','evaluate_panel'}:
            levels=panel_levels if kind=='evaluate_panel' else [level]
            if not levels or len(set(levels))!=len(levels):raise ValueError('Invalid evaluation panel levels')
            batches=[]
            for selected in levels:
                spec=opponent(selected)
                short_baseline=role=='baseline' and self.state.get('baseline_games_per_level',20)==10
                games=self.state.get('baseline_games_per_level',VALIDATION_GAMES) if role=='baseline' else VALIDATION_GAMES
                batch_protocol=protocol(spec,self.site['engine'],baseline=short_baseline,games=games)
                actual_engine=self.engine_probe(self.site['engine'],spec)
                pin={k:v for k,v in actual_engine.items() if k!='spec'}
                if self.state.get('engine_pin') is not None and self.state['engine_pin']!=pin:
                    raise ValueError('Official opponent binary/NNUE identity changed since campaign baseline')
                self.state['engine_pin']=pin
                batches.append({'protocol':batch_protocol,'engine_identity':actual_engine,
                    'seed_schedule':[10000*int(selected*10)+100*batch_index+i for i in range(batch_protocol['games']//2)],
                    'comparison_id':f"{'baseline10' if short_baseline else 'validation'+str(games)}:L{selected}:b{batch_index}"})
            payload.update(**batches[0],track=track,
                resources={'gpu_count':candidate.get('proposal',{}).get('evaluation_gpus',self.site['gpu_count'])},
                reference_source=self.state['initial']['source'])
            if kind=='evaluate_panel':
                total=sum(row['protocol']['games'] for row in batches)
                payload['evaluation_panel']=batches
                payload['protocol']={'game':'chess_panel','games':total,'white':total//2,'black':total//2,
                    'levels':levels,'purpose':'development_panel'}
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
               "track": track if kind in {'evaluate','evaluate_panel'} else None, "candidate_id": candidate["id"], "status": "PREPARED"}
        if kind=='evaluate_panel':job['levels']=list(panel_levels)
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
        if payload['kind']=='evaluate_panel':
            from .evaluator import panel_payloads
            batches=panel_payloads(payload)
            refs=result.get('evaluations',[])
            levels=[p['protocol']['opponent']['level'] for p in batches]
            if [row.get('level') for row in refs]!=levels:
                raise ValueError('Evaluation panel does not match its registered levels')
            for child,row in zip(batches,refs,strict=True):
                inside(payload['output'],row['ref']['path'])
                subjob={**job,'kind':'evaluate','level':row['level'],'id':job['id']+'-L'+str(row['level'])}
                self._consume(subjob,row['ref'],child)
            return
        if payload["kind"] == "research":
            candidate.setdefault("artifacts", []).append(ref)
            if result.get("model_path"):
                model_path = str(inside(self.site["user_root"], result["model_path"]))
                identity = model_identity(model_path)
                if result.get("model_identity") and result["model_identity"] != identity:
                    raise ValueError("Research export identity changed before collection")
                candidate.update(model_path=model_path, model_sha256=identity["sha256"])
            action = result.get('training_loop_action')
            if action:
                candidate.update(training_loop_action=action,lineage_steps=result['lineage_steps'],
                                 lineage_gpu_hours=result.get('lineage_gpu_hours',0.),curve=result['curve'],evaluation=None)
                candidate['jobs'] = candidate['jobs'][:self.state['candidate_step']+1]
                for row in result['evaluations']:
                    metrics = self.metrics(row['ref'])
                    if metrics['model_identity']['sha256'] != candidate['model_sha256']:
                        raise ValueError('Selected loop model differs from its evaluation')
                    self.state['pending_evaluations'].append({**row,'role':'candidate','candidate_id':candidate['id']})
                if action == 'finish':
                    candidate['proposal'].update(recommend=True,recommendation=result['reason'])
                if action == 'review_timeout':
                    self._finish({'status':'review_timeout','checkpoint':result.get('checkpoint_receipt')},persist=False)
                    self.state.update(phase='NEEDS_INPUT',resume_phase='PLANNING',needs_input='Meta review timed out; native checkpoint and selected HF model retained. Resume after restoring Meta connectivity.')
                    return
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
        next_source = self.state.pop('next_replan_source',None)
        source_snapshot(Path((next_source or parent['source'])['root']), folder)
        # Preserve the parent cards so phase 1 can verify that they stay empty.
        if parent.get("skills_path"):
            shutil.copy2(parent["skills_path"], folder/"skills.json")
        atomic_json(folder/"context.json", self.memory.context())
        atomic_json(folder/"proposal.json", {"hypothesis": "Describe the next experiment", "evidence_refs": [],
            "jobs": [], "evaluation": evaluation_plan({"levels":[2,3]}), "recommend": False})
        self.state["workspace"] = str(folder)
        self.save()
        return folder

    def training_review(self,job):
        """Observe the shared disk, never issue a scheduler poll to find a review."""
        from .training_review import request, decision
        payload=read_json(job['manifest'])['payload']
        output=inside(self.site['user_root'],payload['output'])
        pending=output/'scaling/pending-review.json'
        if not pending.is_file():
            return None
        path=inside(output,read_json(pending)['path'])
        review=request(path)
        if path.with_name('expired.json').exists() or decision(path) is not None:
            return None
        p=review['payload']
        if p['job_id']!=job['id'] or p['campaign_id']!=self.root.name or p['candidate_source_sha256']!=payload['candidate_source']['sha256']:
            raise ValueError('Training review belongs to another job/source')
        workspace=inside(self.site['user_root'],Path(self.site['worktree_root'])/self.root.name/'reviews'/f"{job['id']}-{p['step']:08d}")
        if not workspace.exists():
            source_snapshot(Path(payload['candidate_source']['root']),workspace)
            atomic_json(workspace/'proposal.json',{'hypothesis':f"Review training step {p['step']}",'jobs':[]})
        atomic_json(workspace/'context.json',{'campaign':self.memory.context(),'training_review':review,
            'curve':read_json(p['curve_path'])})
        self.state['training_review']={'request':str(path),'workspace':str(workspace),'step':p['step'],'request_id':review['request_id']}
        self.save()
        return {'status':'needs_training_review','workspace':str(workspace),'request':str(path)}

    def finish_training_review(self,answer):
        from .training_review import commit, validate
        item=self.state['training_review']
        if Path(item['request']).with_name('expired.json').exists():
            self.memory.append('training_review_expired',{'request_id':item['request_id'],'step':item['step']})
            self.state['training_review']=None
            self.save()
            return {'status':'expired','checkpoint_retained':True}
        workspace=Path(item['workspace'])
        path=inside(workspace,answer['proposal_path'])
        if path.parent!=workspace:
            raise ValueError('Review proposal outside its registered workspace')
        proposal=read_json(path)
        value=validate(item['request'],proposal.get('training_review'))
        self._validate_phase1_workspace(workspace)
        if value['action']=='continue':
            from .artifacts import snapshot_files
            frozen=self.state['candidate']['source']['files']
            changed=[str(rel) for rel,file in snapshot_files(workspace)
                     if rel.parts[0] in {'src','scripts','configs'} and file_hash(file)!=frozen.get(rel.as_posix())]
            missing=[name for name in frozen if name.split('/')[0] in {'src','scripts','configs'} and not (workspace/name).is_file()]
            if changed or missing:
                raise ValueError('Continue keeps the frozen training plan; use replan to change code/configuration')
        if value['action']=='replan':
            folder=self.root/'review-sources'/item['request_id']
            if not folder.exists():
                source_snapshot(workspace,folder)
            snapshot=read_json(folder/'source-manifest.json');verify_source(snapshot)
            self.state['next_replan_source']=snapshot
        result=commit(item['request'],value)
        self.memory.append('training_review_decision',{**result,'job_id':self.state['active_job'],'step':item['step']},event_id='training-review-'+item['request_id'])
        self.state['training_review']=None
        self.save()
        return result

    def prepare(self, proposal_path):
        require_research(self.root)
        if self.state["phase"] != "PLANNING":
            raise ValueError("Prepare proposals during PLANNING")
        path = Path(proposal_path)
        if not path.is_absolute() and self.state.get("workspace"):
            path = Path(self.state["workspace"])/path
        path = inside(self.site["user_root"], path)
        workspace = Path(self.state["workspace"]) if self.state.get("workspace") else path.parent
        if path.parent != workspace:
            raise ValueError("Use the registered research workspace")
        if self.state.get("research_scope", PHASE1_SCOPE) == PHASE1_SCOPE:
            self._validate_phase1_workspace(workspace)
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
        cards = folder/"skills.json" if (workspace/"skills.json").is_file() else None
        if folder.exists():
            # Crash after publishing files but before saving campaign state:
            # reuse exactly the same prepared experiment, never overwrite it.
            from .meta_protocol import workspace_hash
            snapshot = read_json(folder/"source/source-manifest.json")
            verify_source(snapshot)
            if (snapshot["root"] != str(folder/"source") or read_json(folder/"proposal.json") != proposal
                    or snapshot["sha256"] != workspace_hash(workspace)
                    or cards is not None and file_hash(cards) != file_hash(workspace/"skills.json")
                    or cards is None and (folder/"skills.json").exists()):
                raise ValueError("Prepared candidate differs from this proposal; inspect preserved candidate before retrying")
        else:
            # Publish a complete snapshot atomically; partial copies are never
            # mistaken for a frozen candidate by a later Controller run.
            folder.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix="."+folder.name+"-prepare-", dir=folder.parent))
            try:
                snapshot = source_snapshot(workspace, temporary/"source")
                snapshot["root"] = str(folder/"source")
                atomic_json(temporary/"source/source-manifest.json", snapshot)
                if cards is not None:
                    shutil.copy2(workspace/"skills.json", temporary/"skills.json")
                atomic_json(temporary/"proposal.json", proposal)
                temporary.rename(folder)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        candidate = {"id": folder.name, "parent_id": self.state["working"]["id"], "source": snapshot,
                     "model_path": model, "model_sha256": None, "skills_path": str(cards) if cards else None, "skills_sha256": file_hash(cards) if cards else None,
                     "hypothesis": proposal["hypothesis"], "proposal": proposal, "jobs": jobs,
                     "evaluation": plan, "evaluations": [], "fingerprint": digest({"source":snapshot["sha256"], "model":model})}
        parent = self.state['working']
        if model == parent['model_path']:
            candidate.update(lineage_steps=parent.get('lineage_steps',0),lineage_gpu_hours=parent.get('lineage_gpu_hours',0.),curve=parent.get('curve'))
        self.state.update(candidate=candidate, candidate_step=0, evaluation_cursor=0,
                          pending_evaluations=[], phase="EXPERIMENT", meta_stalls=0)
        self.memory.append("hypothesis", {"candidate_id": candidate["id"], "hypothesis": candidate["hypothesis"],
                                          "source": snapshot["sha256"], "proposal": proposal},
                           event_id="hypothesis-"+candidate["id"])
        self.save()
        return candidate

    def _validate_phase1_workspace(self, workspace):
        """Keep the chess prompt and candidate prompt inputs fixed in phase 1."""
        if (workspace/"harness.py").exists():
            raise ValueError("Phase 1 forbids candidate harness.py")
        cards = workspace/"skills.json"
        if not cards.is_file() or read_json(cards) != []:
            raise ValueError("Phase 1 requires an empty skills.json")
        initial = self.state["initial"]["source"]["files"]
        for relative in ("src/llmgo/training/chess_r1.py",):
            original = initial.get(relative)
            if original is not None and file_hash(workspace/relative) != original:
                raise ValueError(f"Phase 1 freezes the Chess-R1 prompt and answer protocol: {relative}")
        config = workspace/"configs/grpo/default.json"
        if config.is_file():
            recipe = read_json(config)
            if recipe.get("enable_thinking") is not False or recipe.get("answer_protocol") != "chess-r1-boxed-san-v2":
                raise ValueError("Phase 1 requires non-thinking boxed SAN training")
            if recipe.get("model_path", self.site["base_model"]) != self.site["base_model"]:
                raise ValueError("Phase 1 freezes the initial Qwen3.5-4B checkpoint")

    def _evaluate_next(self, candidate, plan, role):
        schedule = [(t,l) for t in plan["tracks"] for l in plan["levels"]]
        cursor = self.state["evaluation_cursor"]
        if cursor < len(schedule):
            track, level = schedule[cursor]
            remaining=[l for t,l in schedule[cursor:] if t==track]
            shared=self.site.get('eval_shared_levels') and len(remaining)>1
            self._new_job('evaluate_panel' if shared else 'evaluate',role,candidate,track=track,level=level,
                          panel_levels=remaining if shared else None,
                          batch_index=candidate.get('proposal',{}).get('evaluation_batch',0))
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
        if candidate.get('training_loop_action')=='finish':
            self.state.update(phase='COMPLETE',final_checkpoint={'model_path':candidate['model_path'],
                'model_sha256':candidate['model_sha256'],'curve':candidate['curve'],'candidate_id':candidate['id']})
        if persist:
            self.save()

    def step(self, *, proposal=None):
        self.state = self.memory.state()
        verify_source(self.state["source"])
        control = read_control(self.root)
        if self.state.get("active_job") and self.state["phase"] != "NEEDS_INPUT":
            job = self.state["jobs"][self.state["active_job"]]
            if control["paused"]:
                intent_path = Path(job["manifest"]).with_name("submission.json")
                confirmed = read_json(intent_path).get("task_id") if intent_path.exists() else None
                if not confirmed:
                    return {"status": "research_paused", "control": control,
                            "pending_job": job["id"], "submission_held": True}
            if not control['paused']:
                review=self.training_review(job)
                if review:
                    return review
            return self._poll()
        phase = self.state["phase"]
        if control["paused"]:
            # Finalize already-collected evaluation evidence, without scheduling work.
            if phase == "EVALUATING" and self.state.get("candidate"):
                plan = self.state["candidate"].get("evaluation")
                count = len(plan["levels"])*len(plan["tracks"]) if plan else 0
                if self.state.get("evaluation_cursor", 0) >= count:
                    self._finish()
            return {"status": "research_paused", "control": control}
        if phase == "BASELINE":
            baseline_plan = evaluation_plan({"levels":self.state.get("baseline_levels", [2,3])})
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
        elif phase == 'COMPLETE':
            return {'status':'completed','checkpoint':self.state['final_checkpoint']}
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
        self._new_job(payload["kind"], old["role"], candidate, level=old.get("level") or 2,
                      track=old.get("track") or "player", command=payload.get("command"),
                      batch_index=old["batch_index"], resume_from=payload["output"],panel_levels=old.get('levels'))

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

    def recover_training_review(self, job_id):
        """Recover a completed replan after a legacy timed review released GPU.

        Caller holds the Controller lock. Keep the old expiry, job, grade and
        completion unchanged; validate the real request-bound Agent handoff and
        preserve its editable source for the next ordinary research round.
        """
        from .artifacts import load_manifest
        from .meta_protocol import ResearchHandshake
        from .training_review import request, validate_choice
        job=self.state['jobs'].get(job_id)
        if not job or job.get('kind')!='research':
            raise ValueError('Recovery requires a registered training job')
        payload=load_manifest(job['manifest'],job['sha256'])['payload']
        output=inside(self.site['user_root'],payload['output'])
        path=inside(output,read_json(output/'scaling/pending-review.json')['path'])
        review=request(path);p=review['payload'];key=review['request_id']
        recovered=self.state.get('recovered_training_reviews',{}).get(key)
        if recovered:
            return {**read_json(recovered),'already_recovered':True}
        if (self.state['phase']!='NEEDS_INPUT' or self.state.get('active_job') or self.state.get('workspace')
                or self.state.get('training_review') or self.state.get('resume_phase')!='PLANNING'):
            raise ValueError('Recovery requires an idle NEEDS_INPUT training-review boundary')
        if not job.get('handled') or job.get('status')!='Succeeded' or job.get('executor_status')!='collected':
            raise ValueError('Training Worker must be terminal and collected before recovery')
        report=read_json(job['executor_report'])
        if report.get('status')!='collected' or report.get('platform_status')!='Succeeded':
            raise ValueError('Missing terminal Worker collection receipt')
        result=self.metrics(report['artifact'])
        if result.get('training_loop_action')!='review_timeout':
            raise ValueError('This Worker did not exit through the legacy review deadline')
        if (p['job_id']!=job_id or p['campaign_id']!=self.root.name
                or p['candidate_source_sha256']!=payload['candidate_source']['sha256']
                or read_json(path.with_name('expired.json'))['request_id']!=key):
            raise ValueError('Expired review does not match the registered training job')
        workspace=inside(self.site['user_root'],Path(self.site['worktree_root'])/self.root.name/'reviews'/f"{job_id}-{p['step']:08d}")
        session=read_json(self.root/'meta-session.json')
        if session.get('status')!='completed' or session.get('process') is not None or session.get('cwd')!=str(workspace):
            raise ValueError('Recovery requires the completed Agent session in this review workspace')
        handshake=ResearchHandshake(self.root,workspace,session['request_key'])
        answer=handshake.read()
        if answer is None or answer['status']!='ready' or answer!=session.get('answer'):
            raise ValueError('Missing matching request-bound Agent completion')
        proposal_path=inside(workspace,answer['proposal_path'])
        if proposal_path.parent!=workspace:
            raise ValueError('Review proposal outside its registered workspace')
        proposal=read_json(proposal_path)
        choice=validate_choice(review,proposal.get('training_review'))
        if choice['action']!='replan':
            raise ValueError('This recovery accepts a completed replan; the exited Worker cannot continue')
        self._validate_phase1_workspace(workspace)
        point=next(row for row in read_json(inside(output,p['curve_path']))['points'] if row['step']==choice['selected_step'])
        identity=model_identity(inside(self.site['user_root'],point['model_path']))
        if identity!=point['model_identity']:
            raise ValueError('Selected checkpoint identity changed')
        working=self.state['working']
        if (working['id']!=payload['candidate_id'] or working['model_path']!=result['model_path']
                or working.get('training_loop_action')!='review_timeout'):
            raise ValueError('Campaign has moved beyond the timed-out training result')
        folder=self.root/'recoveries'/('training-review-'+key)
        if not (folder/'previous-state.json').exists():
            atomic_json(folder/'previous-state.json',self.state)
            atomic_json(folder/'meta-session.json',session)
        if not (folder/'source').exists():
            source_snapshot(workspace,folder/'source')
        snapshot=read_json(folder/'source/source-manifest.json');verify_source(snapshot)
        from .meta_protocol import workspace_hash
        if snapshot['sha256']!=workspace_hash(workspace):
            raise ValueError('Recovery snapshot differs from the completed research')
        record={'recovered':True,'job_id':job_id,'request_id':key,'action':'replan',
            'selected_step':choice['selected_step'],'model_path':point['model_path'],
            'model_sha256':identity['sha256'],'reason':choice['reason'],'review_source':snapshot,
            'completion_path':str(handshake.result_path),'completion_sha256':file_hash(handshake.result_path),
            'proposal_sha256':file_hash(proposal_path),'expired_request_preserved':True,
            'worker_decision_not_written':True,'inference_performed':False,'gpu_jobs_submitted':0,'utc':utc()}
        if (folder/'recovery.json').exists():
            previous=read_json(folder/'recovery.json')
            if any(previous.get(k)!=v for k,v in record.items() if k!='utc'):
                raise ValueError('Existing recovery receipt differs from this completed review')
            record=previous
        else:
            atomic_json(folder/'recovery.json',record)
        working.update(model_path=point['model_path'],model_sha256=identity['sha256'],
            lineage_steps=payload.get('lineage_steps',0)+point['step'],training_loop_action='replan')
        self.state.setdefault('recovered_training_reviews',{})[key]=str(folder/'recovery.json')
        self.state.update(next_replan_source=snapshot,phase='PLANNING',needs_input=None,
            meta_stalls=0,meta_request_revision=self.state.get('meta_request_revision',0)+1)
        self.state.pop('resume_phase',None)
        self.memory.append('expired_training_review_recovered',record,event_id='recover-training-review-'+key)
        self.memory.handoff({'completed':[record],'next':'Use the recovered review source and submit the next ordinary research proposal',
            'working':working['id'],'best':(self.state.get('best') or {}).get('score')})
        self.save()
        return record

    def resume(self):
        """An explicit operator resume authorizes a new Meta turn, not cached output."""
        if self.state["phase"] != "NEEDS_INPUT":
            raise ValueError("resume only clears an explicitly resolved NEEDS_INPUT state")
        self.memory.append("operator_resume", {"previous_reason": self.state.get("needs_input"),
            "previous_phase": self.state.get("resume_phase", "PLANNING"),
            "meta_request_revision": self.state.get("meta_request_revision", 0) + 1})
        self.state.update(phase=self.state.get("resume_phase", "PLANNING"),
                          active_job=self.state.pop("resume_active_job", None),
                          needs_input=None, meta_stalls=0,
                          meta_request_revision=self.state.get("meta_request_revision", 0) + 1)
        self.state.pop("controller_recovery", None)
        self.save()
        return {"phase": self.state["phase"]}

    def _clear_recovery(self, stage):
        recovery = self.state.get("controller_recovery")
        if recovery and recovery.get("stage") == stage:
            self.memory.append("controller_recovered", {"stage": stage, "attempts": recovery["attempt"],
                                                       "workspace": self.state.get("workspace")})
            self.state.pop("controller_recovery", None)
            self.save()

    def _control_failure(self, exc, stage):
        """Persist a bounded retry schedule, or stop clearly at an unsafe boundary.

        Never resubmit here: task submission/reconciliation remains owned by
        Executor. Unknown bugs and artifact integrity failures require inspection.
        """
        from .connection_check import scrub
        previous = self.state.get("controller_recovery") or {}
        attempt = previous.get("attempt", 0)+1 if previous.get("stage") == stage else 1
        retryable = exc.retryable if isinstance(exc, MetaFailure) else isinstance(exc, (OSError, TimeoutError))
        code = getattr(exc, "code", "io" if retryable else "unexpected")
        error = scrub(str(exc))[:4000]
        retry = retryable and attempt <= 5
        delay = min(600, 60*2**min(attempt-1, 4)) if retry else None
        recovery = {"status": "waiting" if retry else "blocked", "stage": stage,
            "code": code, "error_type": type(exc).__name__, "error": error,
            "attempt": attempt, "max_retries": 5, "retryable": retryable,
            "next_retry_at": time.time()+delay if retry else None, "updated": utc()}
        self.state["controller_recovery"] = recovery
        self.memory.append("controller_retry" if retry else "controller_blocked", recovery)
        if retry:
            self.save()
            return {**recovery, "status": "retry_wait"}
        self.state.update(resume_phase=self.state["phase"], resume_active_job=self.state.get("active_job"),
            phase="NEEDS_INPUT", needs_input=f"{stage} / {code}: {error}. "
                + ("自动重试5次后仍未恢复；" if retryable else "需要检查配置或完整性，未自动重试；")
                + "研究文件、任务与回执均保留。处理原因后用start_rsi.sh --resume --web恢复。")
        self.save()
        return {"status": "needs_input", "detail": self.state["needs_input"], "recovery": recovery}

    def run(self, *, backend=None, max_steps=None, poll_seconds=None):
        with self.memory.lock():
            self._publish_control_status()
            iterations = 0
            while max_steps is None or iterations < max_steps:
                iterations += 1
                self._publish_control_status()
                recovery = self.state.get("controller_recovery") or {}
                if (not read_control(self.root)["paused"] and recovery.get("status") == "waiting"
                        and recovery["next_retry_at"] > time.time()):
                    time.sleep(min(60, recovery["next_retry_at"]-time.time()))
                    continue
                try:
                    result = self.step()
                except ResearchPaused as exc:
                    result = {"status": "research_paused", "control": exc.control}
                except Exception as exc:
                    result = self._control_failure(exc, "advance")
                    if result["status"] == "needs_input":
                        return result
                    continue
                if result["status"] != "research_paused":
                    self._clear_recovery("advance")
                if result['status']=='needs_training_review':
                    from .meta import CodexBackend
                    from .training_review import ReviewExpired
                    active_backend=backend or CodexBackend(self.site,self.memory)
                    prompt=(f"读取context.json、.agents/skills/rsi-training-review/SKILL.md和完整曲线。当前训练在安全checkpoint等待你的决定；请求位于 {result['request']}。"
                        "用户已取消审查的固定截止时间；旧副本中900秒或15分钟的条款已失效，Worker将等待有效决定。"
                        "检查L2/L3结果和败局每一步。选择continue、replan或finish；在proposal.json的training_review写request_id、action、reason、selected_step。"
                        "continue保留当前冻结配方并原地续训；replan保留父模型并在本副本修改下一轮方案；finish选择一个已评测checkpoint交用户做独立DeepSeek测试。"
                        "不要提交第二个GPU任务，不修改运行中的冻结源。使用正常finish_research.py交接，status=ready。")
                    try:
                        answer=active_backend.turn(Path(result['workspace']),prompt)
                        if answer['status']!='ready':
                            raise MetaFailure(answer['summary'],code='review_needs_input',retryable=False)
                        self.finish_training_review(answer)
                        self._clear_recovery('training_review')
                    except ResearchPaused:
                        self._publish_control_status()
                    except ReviewExpired:
                        item=self.state.get('training_review') or {}
                        self.memory.append('training_review_expired',{'request_id':item.get('request_id'),'step':item.get('step')})
                        self.state['training_review']=None
                        self.state.pop('next_replan_source',None)
                        self.save()
                    except Exception as exc:
                        outcome=self._control_failure(exc,'training_review')
                        if outcome['status']=='needs_input':
                            return outcome
                    continue
                if result["status"] == "needs_meta":
                    workspace = Path(result["workspace"])
                    prompt = (f"读取 {workspace/'context.json'}、Memory 和 .agents/skills/rsi-memory/SKILL.md。"
                        f"完整实验、评测和逐步棋局证据在 {self.root}。先逐手检查输棋：打开每盘 raw_events 所指的 events.jsonl，"
                        "按 ply 还原双方走法、FEN、模型提示与原始回答、解析结果、随机后备和败因；不能只看胜负摘要。"
                        "Phase 1 默认训练使用固定的 Chess-R1 非思考 FEN、合法 SAN、boxed SAN prompt 与原生 verl GRPO。"
                        "完整提示和数据、Qwen3.5-4B、Searchless Chess 270M 身份见 docs/phase1-rl-scope.md。"
                        "禁止任何 SFT。可在研究副本中改进 RL 算法、reward、采样和课程；"
                        "不可修改 prompt 或答案协议，不可新增 harness.py，skills.json 必须保持空数组。"
                        "正式裁判、历史回执和冻结评测源码不可改。"
                        "初始Qwen3.5-4B checkpoint固定，可继续本研究派生checkpoint。默认仍用现有筛选数据；可自主决定是否使用新的完整Lichess训练池、如何筛选与混合。"
                        "完整池位于 assets/datasets/lichess/20261103/boxed-nonthinking-v2，先检查verification.json。详见docs/lichess-data.md。"
                        "允许修改超参和reward等；无需random seed或算力预算比较。每轮交付选样程序/查询、ID清单、分类统计、基于败局的选样假设；沿用默认数据也记录选择。"
                        "先做有证据的假设与资源计划，再提出训练或 L2/L2.5/L3 Fairy-Stockfish 开发验证。"
                        "L2/L3 后续基线和验证每档40局、白黑各20局，共80局共享队列；每个评测vLLM副本max_num_seqs=20。旧20局结果保留并标注样本数，不混为同规模实验。默认每5个optimizer step保存完整checkpoint并验证，验证后你决定continue、replan或finish；读取rsi-training-review技能。"
                        "保留全部中间checkpoint、原始得分与observed-best曲线；不把观测最优包络当作实际能力单调增长。DeepSeek non-thinking只留给用户做最终测试。每任务最多8 GPU，默认HIGH，同时最多一个HIGH任务。"
                        "按 docs/grpo-pipeline.md 的 recipe 与输出目录提交，训练和评测的 logs、JSON 与逐步棋谱全部写到远端项目 var/runs。"
                        "完成 proposal.json 与必要验证后调用 finish_research.py 交接；确有障碍时返回 needs_input。"
                        f"操作员恢复版本：{self.state.get('meta_request_revision', 0)}。")
                    try:
                        require_research(self.root)
                        if backend is None:
                            from .meta import CodexBackend
                            active_backend = CodexBackend(self.site, self.memory)
                        else:
                            active_backend = backend
                        answer = active_backend.turn(workspace, prompt)
                    except ResearchPaused as exc:
                        self.memory.append("research_paused", {"control": exc.control, "stage": "meta"})
                        self._publish_control_status()
                        continue
                    except Exception as exc:
                        result = self._control_failure(exc, "meta")
                        if result["status"] == "needs_input":
                            return result
                        continue
                    self._clear_recovery("meta")
                    self.memory.append("meta_result", answer)
                    if answer["status"] == "needs_input":
                        self.state.update(phase="NEEDS_INPUT", needs_input=answer["summary"], resume_phase="PLANNING")
                        self.save()
                        return answer
                    try:
                        self.prepare(answer["proposal_path"])
                    except ResearchPaused as exc:
                        self.memory.append("research_paused", {"control": exc.control, "stage": "prepare"})
                        self._publish_control_status()
                    except (ValueError, FileNotFoundError, TypeError) as exc:
                        self.memory.append("proposal_invalid", {"error": str(exc)})
                        self.state["meta_stalls"] += 1
                        atomic_json(workspace/"context.json", self.memory.context())
                        if self.state["meta_stalls"] >= 3:
                            self.state.update(phase="NEEDS_INPUT", needs_input=str(exc), resume_phase="PLANNING")
                        self.save()
                    except Exception as exc:
                        result = self._control_failure(exc, "prepare")
                        if result["status"] == "needs_input":
                            return result
                    else:
                        self._clear_recovery("prepare")
                elif result["status"] == "waiting":
                    time.sleep(min(60, poll_seconds or self.site.get("poll_seconds", 120)))
                elif result["status"] == "research_paused":
                    if max_steps is None:
                        time.sleep(5)
                elif result["status"] in ("submission_uncertain", "needs_input", 'completed'):
                    return result
            return {"status": "step_limit", "phase": self.state["phase"]}
