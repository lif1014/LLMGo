"""Codex app-server backend over JSONL stdio, with persistent visible event logs."""
from __future__ import annotations

import json
import os
import queue
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
                raise RuntimeError(f"Codex exited with status {self.proc.returncode}")
            self.memory.append("meta_heartbeat", {"thread_id": self.session.get("thread_id"),
                                                  "turn_id": self.session.get("turn_id")})
            return
        if channel == "eof":
            raise RuntimeError("Codex app-server stream ended")
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
                raise TimeoutError(f"Codex protocol acknowledgement timed out: {method}")
            self._pump(timeout=min(30, max(.1, deadline-time.monotonic())))
        response = self.responses.pop(rid)
        if "error" in response:
            raise RuntimeError(f"Codex {method}: {scrub_known(response['error'], self._secrets)}")
        return response["result"]

    def turn(self, workspace, prompt):
        isolation = sandbox_mode(self.site)
        root = self.site["user_root"]
        workspace = inside(root, workspace)
        context = workspace / "context.json"
        request_key = digest({"workspace": str(workspace), "prompt": prompt,
                              "context": file_hash(context) if context.exists() else None})
        previous = self.session.get("process")
        if previous and _process_identity(previous["pid"]) == previous:
            raise RuntimeError("Previous Codex process is still alive; inspect it before starting another turn")
        if self.session.get("request_key") == request_key and self.session.get("answer"):
            self.memory.append("meta_recovered", {"thread_id": self.session.get("thread_id"),
                                                  "turn_id": self.session.get("turn_id")})
            return self.session["answer"]
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
            "研究过程中正常使用工具；完成后才输出一个JSON对象，字段为status、summary、proposal_path。"
            "status=ready时proposal_path须指向已写好的proposal.json；需要用户输入时status=needs_input且proposal_path=null。"
            "参阅本目录docs/rsi-operations.md、.agents/skills中的Memory、提交和Executor说明。"
        )

        command = [self.site["codex_bin"], "app-server"]
        self.proc = self.popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
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
                    completed, texts, acted = self._wait_turn(turn_id)
                    turn_ids.append(turn_id)
                review = {**review, "acted": bool(review.get("acted") or acted)}
                answer = next((value for text in reversed(texts) if (value := self._answer(text))), None)
                if answer is None:
                    if review.get("format_retry"):
                        raise ValueError("Codex completed without the required structured result after clarification")
                    review["format_retry"] = True
                    reason = "invalid_final_result"
                    next_prompt = ("上一轮结束时缺少有效的最终JSON。如果尚未完成研究，请继续使用工具。"
                        "完成后返回且只返回JSON对象，包含status（ready或needs_input）、summary和proposal_path。"
                        "ready必须给出实际proposal路径；needs_input必须明确用户需要解决的问题，proposal_path=null。")
                elif answer["status"] == "needs_input" and not review["acted"] and not review.get("action_retry"):
                    review["action_retry"] = True
                    reason = "needs_input_without_actions"
                    next_prompt = ("上一轮没有任何工具执行记录就返回了needs_input。请继续原研究任务，立即用工具读取context.json和Memory。"
                        "‘我将开始读取/研究’不是缺少输入。无需用户批准已授权的读取、CPU分析、skills修改或登记计算任务。"
                        "如果确实存在只有用户能解决的障碍，请明确写出具体障碍和需要用户回答的问题，再返回needs_input。"
                        "否则继续研究，写好proposal.json后返回ready及绝对路径。最终返回JSON对象，字段status、summary、proposal_path。")
                else:
                    self._save(answer=answer, status="completed", turn_id=completed["id"], review=review)
                    return answer
                self.memory.append("meta_continuation", {"reason": reason, "turn_id": completed["id"],
                    "previous_answer": answer, "review": review})
        except BaseException as exc:
            self._save(status="failed", updated=utc())
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
        while True:
            while self.notifications:
                event = self.notifications.popleft()
                p = event.get("params", {})
                if p.get("turnId") not in (None, turn_id):
                    continue
                if event.get("method") == "item/completed":
                    item = p["item"]
                    acted = acted or self._is_action(item)
                    if item.get("type") == "agentMessage":
                        items.append(item.get("text", ""))
                if event.get("method") == "turn/completed" and p["turn"]["id"] == turn_id:
                    turn = p["turn"]
                    self._save(status=turn["status"], updated=utc())
                    if turn["status"] != "completed":
                        raise RuntimeError(f"Codex turn {turn['status']}: {scrub_known(turn.get('error'), self._secrets)}")
                    texts = [i.get("text", "") for i in turn.get("items", []) if i.get("type") == "agentMessage"]
                    return turn, texts or list(items), acted or any(self._is_action(i) for i in turn.get("items", []))
            self._pump()  # no total teacher turn/research timeout

    @staticmethod
    def _answer(text):
        try:
            answer = json.loads(text)
        except (ValueError, TypeError):
            return None
        if (isinstance(answer, dict) and answer.get("status") in ("ready", "needs_input")
                and isinstance(answer.get("summary"), str)):
            return answer
        return None
