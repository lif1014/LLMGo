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
                           {"message": value, "thread_id": self.session.get("thread_id")})
        if channel == "protocol" and isinstance(value, dict) and value.get("method") == "thread/tokenUsage/updated":
            usage = value.get("params", {}).get("tokenUsage")
            if isinstance(usage, dict):
                self._save(token_usage=usage, updated=utc())

    def _send(self, value):
        self.memory.append("meta_send", {"message": value, "thread_id": self.session.get("thread_id")})
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
            raise RuntimeError(f"Codex {method}: {response['error']}")
        return response["result"]

    def turn(self, workspace, prompt):
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
        instructions = (
            "你是国际象棋 RSI Meta-Agent，在用户自己的 0 GPU devspace 工作。"
            "每轮先读持久 Memory 和失败记录。学生能力改进只允许 RL 和条件 skills，禁止 SFT/答案模仿；"
            "正式评测没有教师/局面模型。只写 /user/lyh4170，来源 /user/liyaohu4170 只读。"
            "只用 beijing/model-autosft/NORMAL；devspace 0 GPU，GPU 工作通过 job；不得停止他人任务。"
            "本 turn 编辑指定 workspace 的 skills.json、curriculum.jsonl、proposal.json。"
            "Controller 会按 proposal 冻结和提交任务；不要对同一实验重复提交。"
            "参考 workspace/.agents/skills 的 Memory、提交和 RL 审核说明。"
            "禁止更改冻结的框架、验收协议、现有权重、其他候选或覆盖已有 campaign 事实记录；"
            "可通过 Memory 接口追加研究笔记。"
        )
        command = [self.site["codex_bin"], "app-server"]
        self.proc = self.popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self._save(process=_process_identity(self.proc.pid), cwd=str(workspace))
        readers = []
        for stream, channel in ((self.proc.stdout, "stdout"), (self.proc.stderr, "stderr")):
            thread = threading.Thread(target=self._reader, args=(stream, channel, self.events), daemon=True)
            thread.start()
            readers.append(thread)
        try:
            self._rpc("initialize", {"clientInfo": {"name": "llmgo-rsi", "title": "LLMGo RSI", "version": "1.0.0"}})
            self._send({"method": "initialized", "params": {}})
            params = {"cwd": str(workspace), "approvalPolicy": "never", "sandbox": "workspaceWrite",
                      "developerInstructions": instructions}
            if self.site.get("meta_model"):
                params["model"] = self.site["meta_model"]
            if self.session.get("thread_id"):
                result = self._rpc("thread/resume", {**params, "threadId": self.session["thread_id"]})
                prior_turns = result.get("thread", {}).get("turns", [])
                if any(t.get("status") == "inProgress" for t in prior_turns):
                    raise RuntimeError("Resumed thread still has an in-progress turn; reconcile before continuing")
                if self.session.get("request_key") == request_key:
                    for turn in prior_turns:
                        known_turn = turn.get("id") == self.session.get("turn_id")
                        ack_lost = (self.session.get("status") == "starting"
                                    and not self.session.get("turn_id")
                                    and turn.get("id") not in self.session.get("previous_turn_ids", []))
                        if (known_turn or ack_lost) and turn.get("status") == "completed":
                            for item in reversed(turn.get("items", [])):
                                if item.get("type") == "agentMessage":
                                    answer = self._answer(item.get("text", ""))
                                    if answer:
                                        self._save(answer=answer, status="completed", turn_id=turn["id"])
                                        self.memory.append("meta_recovered", {"turn_id": turn["id"]})
                                        return answer
            else:
                result = self._rpc("thread/start", params)
                prior_turns = result.get("thread", {}).get("turns", [])
            thread_id = result["thread"]["id"]
            self._save(thread_id=thread_id, turn_id=None, status="starting", request_key=request_key,
                       previous_turn_ids=[t["id"] for t in prior_turns], answer=None, updated=utc())
            result = self._rpc("turn/start", {
                "threadId": thread_id, "cwd": str(workspace),
                "input": [{"type": "text", "text": prompt}],
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "workspaceWrite", "writableRoots": [root],
                                  "networkAccess": True, "excludeTmpdirEnvVar": True, "excludeSlashTmp": True},
                "outputSchema": FINAL_SCHEMA,
            })
            turn_id = result["turn"]["id"]
            self._save(turn_id=turn_id, status="running")
            # Full events are already durable in Memory. Retain only a small
            # final-message window in RAM during an unbounded research turn.
            items = deque(maxlen=16)
            while True:
                while self.notifications:
                    event = self.notifications.popleft()
                    p = event.get("params", {})
                    if p.get("turnId") not in (None, turn_id):
                        continue
                    if event.get("method") == "item/completed":
                        item = p["item"]
                        if item.get("type") == "agentMessage":
                            items.append(item.get("text", ""))
                    if event.get("method") == "turn/completed" and p["turn"]["id"] == turn_id:
                        status = p["turn"]["status"]
                        self._save(status=status, updated=utc())
                        if status != "completed":
                            raise RuntimeError(f"Codex turn {status}: {p['turn'].get('error')}")
                        texts = [i.get("text", "") for i in p["turn"].get("items", []) if i.get("type") == "agentMessage"]
                        for text in reversed(texts or list(items)):
                            answer = self._answer(text)
                            if answer:
                                self._save(answer=answer)
                                return answer
                        raise ValueError("Codex completed without the required structured result")
                self._pump()  # no total teacher turn/research timeout
        except BaseException as exc:
            self._save(status="failed", updated=utc())
            self.memory.append("meta_failure", {"error_type": type(exc).__name__, "error": str(exc),
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
    def _answer(text):
        try:
            answer = json.loads(text)
        except (ValueError, TypeError):
            return None
        if (isinstance(answer, dict) and answer.get("status") in ("ready", "needs_input")
                and isinstance(answer.get("summary"), str)):
            return answer
        return None
