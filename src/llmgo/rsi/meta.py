"""Codex app-server backend over JSONL stdio, with persistent visible event logs."""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Protocol

from .contracts import atomic_json, digest, file_hash, inside, read_json, utc, worker_environment
from .codex_auth import runtime_auth, scrub_known
from .codex_sandbox import sandbox_mode, thread_sandbox, turn_sandbox
from .generation_guard import RepetitiveGeneration, StreamRepetitionGuard
from .meta_protocol import MetaFailure, ResearchHandshake, service_failure


class MetaAgentBackend(Protocol):
    def turn(self, workspace: Path, prompt: str) -> dict: ...


FINAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"status": {"enum": ["ready", "needs_input"]},
                   "summary": {"type": "string"}, "proposal_path": {"type": ["string", "null"]}},
    "required": ["status", "summary", "proposal_path"],
}


def _process_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        # comm may contain spaces: parse after the closing ')'.
        return {"pid": pid, "host": socket.gethostname(), "start_ticks": raw.rsplit(")", 1)[1].split()[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


class CodexBackend:
    def __init__(self, site, memory, popen=subprocess.Popen):
        self.site, self.memory, self.popen = site, memory, popen
        self.session_path = memory.root / "meta-session.json"
        self.session = read_json(self.session_path) if self.session_path.exists() else {}
        self.events = queue.Queue()
        self.sequence = 0
        self.proc = None
        self.responses = {}
        self.notifications = deque()
        self._secrets = ()

    def _save(self, **changes):
        old_status = self.session.get("status")
        self.session.update(changes)
        atomic_json(self.session_path, self.session)
        if "status" in changes and changes["status"] != old_status:
            self.memory.append("meta_status", {"status": changes["status"], "thread_id": self.session.get("thread_id"),
                "turn_id": self.session.get("turn_id"), "cwd": self.session.get("cwd")})

    def _reader(self, stream, channel, events):
        try:
            for line in stream:
                if channel == "stdout":
                    try:
                        value = json.loads(line)
                    except ValueError:
                        events.put(("stderr", {"text": line, "malformed_protocol": True}))
                    else:
                        events.put(("protocol", value))
                else:
                    events.put(("stderr", {"text": line}))
        finally:
            if channel == "stdout":
                events.put(("eof", {}))

    def _record(self, channel, value):
        """Persist a received message independently of active RPC dispatch."""
        self.memory.append("meta_event" if channel == "protocol" else "meta_stderr",
                           {"message": scrub_known(value, self._secrets), "thread_id": self.session.get("thread_id")})
        if channel == "protocol" and isinstance(value, dict) and value.get("method") == "thread/tokenUsage/updated":
            usage = value.get("params", {}).get("tokenUsage")
            if isinstance(usage, dict):
                self._save(token_usage=usage, updated=utc())

    def _send(self, value):
        self.memory.append("meta_send", {"message": scrub_known(value, self._secrets), "thread_id": self.session.get("thread_id")})
        self.proc.stdin.write(json.dumps(value) + "\n")
        self.proc.stdin.flush()

    def _pump(self, timeout=30):
        try:
            channel, value = self.events.get(timeout=timeout)
        except queue.Empty:
            if self.proc.poll() is not None:
                raise MetaFailure(f"Codex exited with status {self.proc.returncode}")
            self.memory.append("meta_heartbeat", {"thread_id": self.session.get("thread_id"),
                                                  "turn_id": self.session.get("turn_id")})
            return
        if channel == "eof":
            raise MetaFailure("Codex app-server stream ended")
        self._record(channel, value)
        if channel != "protocol":
            return
        if "id" in value and "method" not in value:
            self.responses[value["id"]] = value
        elif "id" in value:
            # Explicit never-approval policy normally avoids these. Do not silently
            # approve an unexpected permission request from a different protocol.
            self._send({"id": value["id"], "error": {"code": -32601, "message": "Interactive request unavailable; report needs_input"}})
        else:
            self.notifications.append(value)

    def _rpc(self, method, params, timeout=120):
        self.sequence += 1
        rid = self.sequence
        self._send({"id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while rid not in self.responses:
            if time.monotonic() >= deadline:
                raise MetaFailure(f"Codex protocol acknowledgement timed out: {method}", code="rpc_timeout")
            self._pump(timeout=min(30, max(.1, deadline-time.monotonic())))
        response = self.responses.pop(rid)
        if "error" in response:
            error = response["error"]
            message = f"Codex {method}: {scrub_known(error, self._secrets)}"
            if error.get("code") in {-32600, -32601, -32602}:
                raise MetaFailure(message, code="protocol_configuration", retryable=False)
            raise service_failure(message)
        return response["result"]

    def turn(self, workspace, prompt):
        isolation = sandbox_mode(self.site)
        root = self.site["user_root"]
        workspace = inside(root, workspace)
        context = workspace / "context.json"
        request_key = digest({"workspace": str(workspace), "prompt": prompt,
                              "context": file_hash(context) if context.exists() else None})
        handshake = ResearchHandshake(self.memory.root, workspace, request_key)
        previous = self.session.get("process")
        if previous and _process_identity(previous["pid"]) == previous:
            raise RuntimeError("Previous Codex process is still alive; inspect it before starting another turn")
        if self.session.get("request_key") == request_key and self.session.get("answer"):
            cached = self.session.get("result_source") != "file"
            if not cached:
                try:
                    cached = handshake.read() == self.session["answer"]
                except (ValueError, OSError, TypeError, KeyError) as exc:
                    self.memory.append("meta_completion_rejected", {"request_id": request_key,
                        "error": str(exc), "source": "cached_completion"})
            if cached:
                self.memory.append("meta_recovered", {"thread_id": self.session.get("thread_id"),
                                                      "turn_id": self.session.get("turn_id")})
                return self.session["answer"]
        finish_command = handshake.publish()
        # A deliberate file commit remains valid if the Controller died before
        # receiving the final model message. Reuse it without another API call.
        try:
            committed = handshake.read()
        except (ValueError, OSError, TypeError, KeyError):
            committed = None  # The next turn will receive the rejection details.
        if committed is not None:
            self._save(answer=committed, status="completed", result_source="file",
                       result_path=str(handshake.result_path), process=None, updated=utc())
            self.memory.append("meta_completion_committed", {"request_id": request_key,
                "result_path": str(handshake.result_path), "answer": committed, "recovered": True})
            return committed
        completion_prompt = ("\n明确交接接口：当前研究请求为 " + str(handshake.request_path) + "。"
            "完成代码修改、测试和proposal后，执行 " + shlex.join(finish_command) +
            " --summary '本轮完成内容与证据'。此命令保存绑定本次请求和当前源码哈希的完成记录，"
            "不提交任务；之后结束发言即可。需要用户补充时加 --status needs_input，summary写具体问题。"
            "只有proposal.json存在不代表完成；未完成就继续工具工作，不要提前调用完成命令。"
            "完成后又修改研究文件必须重新执行完成命令。也兼容原来的三字段最终JSON。")
        self.events = queue.Queue()
        self.responses = {}
        self.notifications = deque()
        env = worker_environment(self.site)
        env["CODEX_HOME"] = str(inside(root, self.site["codex_home"]))
        env, self._secrets = runtime_auth(self.site, env)
        instructions = (
            "你是国际象棋自主研究 Meta-Agent，在用户自己的 0 GPU devspace 工作。"
            "每轮先读持久 Memory、失败记录与跨难度评测。只能通过强化学习（RL）和改进skills这两种方法提升Qwen3.5-4B，可以单独使用或组合。"
            "禁止使用SFT（监督微调），包括SFT预热、辅助SFT损失、行为克隆和对教师答案的监督蒸馏；使用SFT，最终得分为0分。"
            "可自主修改RL算法、reward、数据、课程，以及为RL和skills服务的Harness、prompt、工具、skills内容/触发/选择、研究流程；默认训练入口scripts/train_ppo.py现为verl 0.8.0 PPO+LC0，可研究修改，不要求沿用现有PPO。"
            "离线教师可用于RL奖励/价值估计、局面分析和提炼skills，不可将教师走法或思考作为SFT目标。RL中的策略梯度和价值函数拟合仍可使用。"
            "最终会把成果对战固定L1-L8 Fairy-Stockfish，最低档skill -9/depth5/0.05秒；只刷简单档没有用。"
            "新campaign先做L1-L4基线，每档10局、白黑各5局，共40局，然后开始研究；初始基线不是完整八档成绩。"
            "每项计算任务最多8 GPU，默认8卡。评测每卡一个Qwen副本并行棋局；可用request_evaluation.py --level L2 --gpus 4改并行数。verl PPO默认7卡actor/rollout/ref加1卡LC0，单卡时需CPU LC0后端。不要在devspace使用GPU。每次研究可只测一个难度，不必每次全八档；后续单档评测20局，比较时注意样本量差异。"
            "在workspace运行 python scripts/request_evaluation.py --level L2 即可更新proposal.json为只测L2，保留已有训练任务；"
            "再返回ready和proposal绝对路径，由Executor提交及监控。--track model选择固定Harness模型对照，--full-panel请求L1-L8完整评测。"
            "可以调用受保护evaluate.py；正式裁判、协议、引擎和结果记录不得更改。正式走棋由Qwen玩家完成，不用另一引擎替它选步。"
            "只写 /user/lyh4170，来源 /user/liyaohu4170 只读。不要修改用户其他任务的环境；新增依赖用本研究专用或实验独立venv与缓存，不原地升级系统/他人环境。"
            "GPU任务提交后每5–10分钟查询一次，纯CPU任务可较频繁查询，常规监控交给Executor。只用 beijing/model-autosft/NORMAL；devspace 0 GPU，GPU工作通过job；不得停止他人任务。"
            "指定workspace是自由研究副本，任意修改其内容。用proposal.json登记任意计算命令和评测计划；"
            "Controller/Executor会快照和提交，避免重复提交同一实验。可以自行研究提交命令，遵守资源授权。"
            "没有研究总时间上限。失败也记录并继续，不要求每轮成功。"
            "收到研究请求就立即使用工具读取证据并开展工作，不要把‘我将开始读取/研究’作为最终回答。"
            "只有确实缺少用户才能提供的信息或权限时才返回needs_input，summary必须写清实际阻碍和具体问题。"
            "可自行阅读文件、运行CPU命令、修改研究副本或登记计算任务的事项，不是缺少用户输入。"
            "研究过程中正常使用工具；完成后优先执行本次请求的finish_research.py命令保存明确交接记录。"
            "也兼容最后输出JSON对象：status、summary、proposal_path；ready指向实际提案，needs_input时proposal_path=null。"
            "参阅本目录docs/rsi-operations.md、.agents/skills中的Memory、提交和Executor说明。"
        ) + completion_prompt

        command = [self.site["codex_bin"], "app-server"]
        try:
            self.proc = self.popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        except OSError as exc:
            raise MetaFailure(str(exc), code="launcher", retryable=False) from exc
        self._save(process=_process_identity(self.proc.pid), cwd=str(workspace), sandbox_mode=isolation)
        readers = []
        for stream, channel in ((self.proc.stdout, "stdout"), (self.proc.stderr, "stderr")):
            thread = threading.Thread(target=self._reader, args=(stream, channel, self.events), daemon=True)
            thread.start()
            readers.append(thread)
        try:
            self._rpc("initialize", {"clientInfo": {"name": "llmgo-rsi", "title": "LLMGo RSI", "version": "1.0.0"}})
            self._send({"method": "initialized", "params": {}})
            params = {"cwd": str(workspace), "approvalPolicy": "never", "sandbox": thread_sandbox(self.site),
                      "developerInstructions": instructions}
            if self.site.get("meta_model"):
                params["model"] = self.site["meta_model"]
            recovered = None
            same_request = self.session.get("request_key") == request_key
            if self.session.get("thread_id"):
                result = self._rpc("thread/resume", {**params, "threadId": self.session["thread_id"]})
                prior_turns = result.get("thread", {}).get("turns", [])
                if any(t.get("status") == "inProgress" for t in prior_turns):
                    raise RuntimeError("Resumed thread still has an in-progress turn; reconcile before continuing")
                if same_request:
                    for turn in prior_turns:
                        known_turn = turn.get("id") == self.session.get("turn_id")
                        ack_lost = (self.session.get("status") == "starting"
                                    and not self.session.get("turn_id")
                                    and turn.get("id") not in self.session.get("previous_turn_ids", []))
                        if (known_turn or ack_lost) and turn.get("status") == "completed":
                            recovered = turn
            else:
                result = self._rpc("thread/start", params)
                prior_turns = result.get("thread", {}).get("turns", [])
            thread_id = result["thread"]["id"]
            turn_ids = [t["id"] for t in prior_turns]
            review = self.session.get("review", {}) if same_request else {}
            next_prompt = self.session.get("pending_prompt", prompt) if same_request else prompt
            # Old failed sessions used a single format_retry bit. Migrate without
            # rethrowing their saved terminal reply forever after ordinary restart.
            review = {k:v for k,v in review.items() if k != "format_retry"}
            if completion_prompt not in next_prompt:
                next_prompt += completion_prompt
            self._save(thread_id=thread_id, request_key=request_key, answer=None, updated=utc())
            while True:
                if recovered is not None:
                    completed = recovered
                    texts = [i.get("text", "") for i in completed.get("items", []) if i.get("type") == "agentMessage"]
                    acted = any(self._is_action(i) for i in completed.get("items", []))
                    recovered = None
                    self.memory.append("meta_recovered", {"turn_id": completed["id"], "review_pending": True})
                else:
                    self._save(turn_id=None, status="starting", previous_turn_ids=turn_ids,
                               pending_prompt=next_prompt, review=review)
                    result = self._rpc("turn/start", {
                        "threadId": thread_id, "cwd": str(workspace),
                        "input": [{"type": "text", "text": next_prompt}],
                        "approvalPolicy": "never", "sandboxPolicy": turn_sandbox(self.site),
                    })
                    turn_id = result["turn"]["id"]
                    self._save(turn_id=turn_id, status="running")
                    try:
                        completed, texts, acted = self._wait_turn(turn_id)
                    except RepetitiveGeneration as exc:
                        turn_ids.append(turn_id)
                        retries = review.get("generation_retries", 0)
                        if retries >= 2:
                            answer = {"status": "needs_input", "proposal_path": None,
                                "summary": "Meta模型连续出现重复生成，已中断并更换研究会话重试两次仍未恢复。"
                                    "研究文件、原会话和已提交任务均保留；请检查上游模型服务或调整Meta模型配置后恢复。"}
                            self.memory.append("meta_generation_exhausted", {"thread_id": thread_id,
                                "turn_id": turn_id, "recoveries": retries})
                            self._save(answer=answer, status="recovery_exhausted", review=review)
                            return answer
                        review = {**review, "generation_retries": retries + 1}
                        prior_thread = thread_id
                        fresh = self._rpc("thread/start", params)
                        thread_id = fresh["thread"]["id"]
                        turn_ids = []
                        next_prompt = (prompt + "\n恢复说明：上一会话的助手文字出现长循环重复，已中断该turn，原始记录仍保留。"
                            "请先读取context.json、Memory摘要和workspace已有文件，从实际进度继续，不重做已完成工作。"
                            "先核对已有任务ID、提交intent与产物，不能重复提交训练或评测。不要读取或复述整段重复输出。"
                            "现在执行具体工具动作；完成研究后写proposal.json并明确交接。" + completion_prompt)
                        self.memory.append("meta_generation_recovery", {"previous_thread_id": prior_thread,
                            "previous_turn_id": turn_id, "thread_id": thread_id,
                            "attempt": retries + 1, "workspace": str(workspace)})
                        self._save(thread_id=thread_id, turn_id=None, status="recovering",
                                   previous_turn_ids=[], pending_prompt=next_prompt, review=review)
                        continue
                    turn_ids.append(turn_id)
                review = {**review, "acted": bool(review.get("acted") or acted)}
                # Only the final agent message can authorize completion; an
                # earlier JSON example must not be accepted as the final result.
                answer = self._answer(texts[-1]) if texts else None
                result_source = "message"
                commit_error = None
                try:
                    committed = handshake.read()
                except (ValueError, OSError, TypeError, KeyError) as exc:
                    committed, commit_error = None, str(exc)
                    self.memory.append("meta_completion_rejected", {"request_id": request_key,
                        "turn_id": completed["id"], "error": commit_error})
                if committed is not None:
                    same_result = answer is None or (answer["status"] == committed["status"] and (
                        answer["proposal_path"] is None and committed["proposal_path"] is None or
                        answer["proposal_path"] is not None and committed["proposal_path"] is not None and
                        (workspace/answer["proposal_path"]).resolve() == Path(committed["proposal_path"]).resolve()))
                    if not same_result:
                        commit_error = "Completion file and final JSON disagree; explicitly commit the intended result again"
                    else:
                        answer, result_source = committed, "file"
                if commit_error:
                    answer = None
                if answer is None:
                    review["continuations"] = review.get("continuations", 0) + 1
                    review["no_action_completions"] = 0 if acted else review.get("no_action_completions", 0) + 1
                    reason = "invalid_final_result"
                    next_prompt = ("刚才只结束了一次模型发言，研究尚未交接。已有文件和工具结果都保留；"
                        "请从实际进度继续，不重做已完成研究、不重复提交任务。"
                        "若研究已完成，请调用下面的完成命令；尚未完成则继续使用工具。" +
                        ("上次完成记录被拒绝：" + commit_error if commit_error else "") + completion_prompt)
                    # Productive research has no turn/time cap. Repeated empty
                    # completions instead receive a fresh, bounded context recovery.
                    if review["no_action_completions"] >= 2:
                        recoveries = review.get("protocol_recoveries", 0)
                        if recoveries >= 2:
                            raise MetaFailure("Meta repeatedly ended without an explicit research completion; files retained",
                                              code="completion_protocol")
                        prior_thread = thread_id
                        fresh = self._rpc("thread/start", params)
                        thread_id = fresh["thread"]["id"]
                        turn_ids = []
                        review.update(protocol_recoveries=recoveries+1, no_action_completions=0)
                        next_prompt = (prompt + "\n原会话连续结束发言却未明确交接，已换新会话。"
                            "先读context.json、Memory和workspace现有笔记与proposal，核对已有任务及产物后继续；"
                            "不要复述旧会话全文，不要重复提交。" + completion_prompt)
                        self.memory.append("meta_protocol_recovery", {"previous_thread_id": prior_thread,
                            "previous_turn_id": completed["id"], "thread_id": thread_id,
                            "attempt": recoveries+1, "workspace": str(workspace)})
                        reason = "fresh_session_after_empty_completions"
                elif answer["status"] == "needs_input" and not review["acted"] and not review.get("action_retry"):
                    review["action_retry"] = True
                    reason = "needs_input_without_actions"
                    next_prompt = ("上一轮没有任何工具执行记录就返回了needs_input。请继续原研究任务，立即用工具读取context.json和Memory。"
                        "‘我将开始读取/研究’不是缺少输入。无需用户批准已授权的读取、CPU分析、skills修改或登记计算任务。"
                        "如果确实存在只有用户能解决的障碍，请明确写出具体障碍和需要用户回答的问题，再返回needs_input。"
                        "否则继续研究，写好proposal.json后明确交接。" + completion_prompt)
                else:
                    self._save(answer=answer, status="completed", turn_id=completed["id"], review=review,
                               result_source=result_source, result_path=str(handshake.result_path), last_error=None)
                    if result_source == "file":
                        self.memory.append("meta_completion_committed", {"request_id": request_key,
                            "result_path": str(handshake.result_path), "answer": answer})
                    return answer
                self._save(thread_id=thread_id, turn_id=None, status="continuing", review=review,
                           previous_turn_ids=turn_ids, pending_prompt=next_prompt, updated=utc())
                self.memory.append("meta_continuation", {"reason": reason, "turn_id": completed["id"],
                    "previous_answer": answer, "review": review})
        except BaseException as exc:
            if isinstance(exc, MetaFailure) and exc.code == "completion_protocol":
                # Only a confirmed terminal turn reaches this branch. A later
                # scheduled retry must not replay the same exhausted turn.
                self.memory.append("meta_protocol_retry_pending", {"thread_id":self.session.get("thread_id"),
                    "turn_id":self.session.get("turn_id"), "workspace":str(workspace)})
                self._save(recovery_parent_thread_id=self.session.get("thread_id"),
                    recovery_parent_turn_id=self.session.get("turn_id"), thread_id=None, turn_id=None,
                    review={}, previous_turn_ids=[], pending_prompt=prompt+completion_prompt)
            self._save(status="failed", updated=utc(), last_error={"code":getattr(exc,"code","unexpected"),
                "message":scrub_known(str(exc),self._secrets), "retryable":getattr(exc,"retryable",False)})
            self.memory.append("meta_failure", {"error_type": type(exc).__name__, "error": scrub_known(str(exc), self._secrets),
                "thread_id": self.session.get("thread_id"), "turn_id": self.session.get("turn_id")})
            raise
        finally:
            if self.proc.stdin:
                self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()  # only this backend's own child
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            for thread in readers:
                thread.join(timeout=2)
            # turn/completed can precede final usage and stderr flushed at EOF.
            # Archive those bytes without dispatching RPCs against a closed stdin.
            while True:
                try:
                    channel, value = self.events.get_nowait()
                except queue.Empty:
                    break
                if channel != "eof":
                    self._record(channel, value)
            self._save(process=None)

    @staticmethod
    def _is_action(item):
        return item.get("type") in {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall", "webSearch"}

    def _wait_turn(self, turn_id):
        # Full events are durable; retain a bounded message window in RAM.
        items = deque(maxlen=16)
        acted = False
        guard = StreamRepetitionGuard()
        repetition = None
        interrupt_deadline = None
        while True:
            if interrupt_deadline is not None and time.monotonic() >= interrupt_deadline:
                raise MetaFailure("Meta repetition detected but turn interruption did not finish; inspect before restarting",
                                  code="interrupt_unconfirmed", retryable=False)
            while self.notifications:
                event = self.notifications.popleft()
                p = event.get("params", {})
                if p.get("turnId") not in (None, turn_id):
                    continue
                if event.get("method") == "item/agentMessage/delta" and repetition is None:
                    repetition = guard.feed(p.get("itemId"), p.get("delta"))
                    if repetition is not None:
                        repetition = scrub_known(repetition, self._secrets)
                        self.memory.append("meta_generation_loop", {"thread_id": self.session.get("thread_id"),
                            "turn_id": turn_id, "item_id": p.get("itemId"), **repetition})
                        self._save(status="interrupting_repetition", updated=utc())
                        self._rpc("turn/interrupt", {"threadId": self.session["thread_id"], "turnId": turn_id})
                        interrupt_deadline = time.monotonic() + 30
                if event.get("method") == "item/completed":
                    item = p["item"]
                    acted = acted or self._is_action(item)
                    if item.get("type") == "agentMessage":
                        items.append(item.get("text", ""))
                if event.get("method") == "turn/completed" and p["turn"]["id"] == turn_id:
                    turn = p["turn"]
                    self._save(status=turn["status"], updated=utc())
                    if repetition is not None and turn["status"] in {"completed", "interrupted"}:
                        raise RepetitiveGeneration(repetition)
                    if turn["status"] != "completed":
                        message = f"Codex turn {turn['status']}: {scrub_known(turn.get('error'), self._secrets)}"
                        if turn["status"] == "interrupted":
                            raise MetaFailure(message, code="interrupted", retryable=False)
                        raise service_failure(message, code="turn_failed")
                    texts = [i.get("text", "") for i in turn.get("items", []) if i.get("type") == "agentMessage"]
                    return turn, texts or list(items), acted or any(self._is_action(i) for i in turn.get("items", []))
            remaining = max(.1, interrupt_deadline-time.monotonic()) if interrupt_deadline is not None else 30
            self._pump(timeout=min(30, remaining))  # no total teacher turn/research timeout

    @staticmethod
    def _answer(text):
        if not isinstance(text, str):
            return None

        def parse(raw):
            def unique(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("Duplicate result key")
                    value[key] = item
                return value
            try:
                value = json.loads(raw, object_pairs_hook=unique)
            except (ValueError, TypeError):
                return None
            if (not isinstance(value, dict) or set(value) != {"status", "summary", "proposal_path"}
                    or not isinstance(value["summary"], str) or not value["summary"].strip()):
                return None
            if value["status"] == "ready":
                return value if isinstance(value["proposal_path"], str) and value["proposal_path"].strip() else None
            if value["status"] == "needs_input" and value["proposal_path"] is None:
                return value
            return None

        answer = parse(text.strip())
        if answer is not None:
            return answer
        # Accept one complete JSON fence, including a final result after prose.
        # Do not guess from arbitrary braces, tool examples or partial JSON.
        blocks = re.findall(r"^[ \t]*```(?:json)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*$",
                            text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        results = [value for block in blocks if (value := parse(block.strip())) is not None]
        return results[0] if len(results) == 1 else None

    def recover_completed(self, workspace):
        """Recover the recorded terminal reply without another model request."""
        workspace = inside(self.site["user_root"], workspace)
        if (self.session.get("process") is not None or self.session.get("status") not in {"failed", "completed"}
                or self.session.get("cwd") != str(workspace)):
            raise ValueError("Recovery requires a stopped session in the registered workspace")
        thread_id, turn_id = self.session.get("thread_id"), self.session.get("turn_id")
        if not thread_id or not turn_id:
            raise ValueError("No recorded Meta thread/turn to recover")
        rows = list(reversed(self.memory.recent(500, "meta_event")))
        completed, messages = None, []
        for row in rows:
            body = row["body"]
            message = body.get("message", {})
            params = message.get("params", {})
            if params.get("threadId", body.get("thread_id")) != thread_id:
                continue
            if message.get("method") == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                completed = params["turn"]
            if message.get("method") == "item/completed" and params.get("turnId") == turn_id:
                item = params.get("item", {})
                if item.get("type") == "agentMessage":
                    messages.append((item.get("text", ""), row["id"]))
        if completed is None or completed.get("status") != "completed":
            raise ValueError("No completed-turn evidence in the recent protocol journal; inspect original trace")
        for item in completed.get("items", []):
            if item.get("type") == "agentMessage":
                messages.append((item.get("text", ""), None))
        # Only the final agent message may authorize a proposal, not an earlier example.
        answer = self._answer(messages[-1][0]) if messages else None
        if answer is None or answer["status"] != "ready":
            raise ValueError("Recorded final reply has no unambiguous ready result")
        self._save(answer=answer, status="completed", updated=utc())
        self.memory.append("meta_result_recovered", {"thread_id": thread_id, "turn_id": turn_id,
            "event_id": messages[-1][1], "answer": answer, "inference_performed": False})
        return answer
