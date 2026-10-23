"""Bounded Codex app-server smoke test, with secrets removed from saved traces."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import tomllib
import uuid

from .contracts import REPO, inside, load_site, worker_environment
from .codex_auth import resolve_auth, scrub_known
from .meta import FINAL_SCHEMA
from .codex_sandbox import sandbox_mode, thread_sandbox, turn_sandbox


class ProtocolError(RuntimeError):
    pass


def scrub(value, secret=""):
    secrets = (secret,) if isinstance(secret, str) else secret
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.sub(r"[^a-z]", "", k.lower()) in {
            "apikey", "authorization", "accesstoken", "refreshtoken", "password", "secret", "token", "experimentalbearertoken"
        } else scrub(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secret) for v in value]
    if isinstance(value, str):
        value = scrub_known(value, secrets)
        value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", value)
        return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
    return value


def failure_hint(message):
    text = message.lower()
    if "bwrap" in text or "bubblewrap" in text or "sandbox command failed" in text:
        return "Codex shell 执行检查未通过；external 模式由服务器负责隔离，workspace-write 模式还需 bubblewrap/挂载权限。检查 trace.jsonl 中的 command/exec 错误。"
    if "/responses" in text and "not supported" in text:
        return "当前入口或模型通道不支持 Codex 所需的 Responses 协议；需要支持 Responses 的入口/通道，或适配 Chat Completions 后端。"
    if any(term in text for term in ("401", "403", "unauthorized", "invalid api key", "令牌", "鉴权")):
        return "检查该站点的 API key、有效期和访问权限。"
    if any(term in text for term in ("404", "model_not_found", "model not found", "不存在", "无可用渠道")):
        return "检查 base_url 的 /v1 路径及该 key 是否可使用当前模型。"
    if any(term in text for term in ("429", "quota", "balance", "余额")):
        return "检查站点余额、额度和限流。"
    if any(term in text for term in ("html", "<!doctype", "expected value")):
        return "接口可能返回了网页；检查 base_url 是否为 https://yeysai.com/v1。"
    if any(term in text for term in ("timeout", "timed out", "ssl", "tls", "dns", "connect", "network")):
        return "检查当前 devspace 的网络或为本进程设置可用代理。"
    return "查看测试目录的 trace.jsonl，检查 Codex 协议及本地转接服务的错误。"


def check_connection(site, output, *, offline=False, timeout=180, env=None, sandbox_check=False):
    if not math.isfinite(timeout) or not 1 <= timeout <= 1800:
        raise ValueError("timeout must be between 1 and 1800 seconds")
    environment = dict(os.environ if env is None else env)
    home = Path(site["codex_home"])
    config = tomllib.loads((home / "config.toml").read_text())
    provider = config.get("model_provider", "openai")
    settings = config.get("model_providers", {}).get(provider, {})
    environment, authentication, secret = resolve_auth(config, environment)
    key_name = authentication["api_key_env"]
    result = {"ok": False, "offline": offline, "inference_tested": False,
              "shell_execution_tested": False, "model_tool_calls_tested": False,
              "sandbox_mode": sandbox_mode(site),
              "model": site.get("meta_model") or config.get("model"), "provider": provider,
              "base_url": settings.get("base_url"), **authentication, "codex_home": str(home),
              "codex_bin": site["codex_bin"], "output": str(output)}
    output = inside(site["user_root"], output)
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    environment.update({k: v for k, v in worker_environment(site).items()
                        if k in {"TMPDIR", "XDG_CACHE_HOME", "CCTL_NO_UPDATE_CHECK"}})
    environment["CODEX_HOME"] = str(home)
    started = time.monotonic()
    process, readers = None, []
    events = queue.Queue()
    pending = deque()
    stderr_tail = deque(maxlen=30)
    trace_path = output / "trace.jsonl"

    def save():
        result["seconds"] = round(time.monotonic() - started, 3)
        (output / "result.json").write_text(json.dumps(scrub(result, secret), ensure_ascii=False, indent=2) + "\n")

    with trace_path.open("w", encoding="utf-8") as trace:
        os.chmod(trace_path, 0o600)

        def record(channel, message):
            if channel == 'stderr':
                stderr_tail.append(scrub(message.get('text',''),secret))
            trace.write(json.dumps({"channel": channel, "message": scrub(message, secret)}, ensure_ascii=False) + "\n")
            trace.flush()

        def receive(deadline):
            while time.monotonic() < deadline:
                try:
                    channel, message = events.get(timeout=min(1, max(.01, deadline-time.monotonic())))
                except queue.Empty:
                    continue
                record(channel, message)
                if channel == "stdout_eof":
                    raise ProtocolError("Codex app-server closed stdout")
                if channel != "stdout":
                    continue
                if "method" in message and "id" in message:
                    send({"id": message["id"], "error": {"code": -32601, "message": "Interactive request unavailable in connectivity test"}})
                    continue
                return message
            raise TimeoutError("Codex connectivity test timed out")

        def send(message):
            record("send", message)
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()

        sequence = 0

        def rpc(method, params, deadline):
            nonlocal sequence
            sequence += 1
            rid = sequence
            send({"id": rid, "method": method, "params": params})
            while True:
                message = receive(deadline)
                if message.get("id") == rid and "method" not in message:
                    if "error" in message:
                        raise ProtocolError(json.dumps(message["error"], ensure_ascii=False))
                    return message["result"]
                pending.append(message)

        def reader(stream, channel):
            try:
                for line in stream:
                    try:
                        message = json.loads(line) if channel == "stdout" else {"text": line}
                        if not isinstance(message, dict):
                            message = {"text": str(message)}
                    except ValueError:
                        message = {"text": line}
                    events.put((channel, message))
            finally:
                events.put((channel + "_eof", {}))

        try:
            if not authentication["api_key_present"] and not offline:
                raise ValueError(f"Missing environment variable: {key_name}; export it or set experimental_bearer_token in the selected provider")
            deadline = started + timeout
            version = subprocess.run([site["codex_bin"], "--version"], cwd=output, env=environment,
                                     capture_output=True, text=True, timeout=min(20, timeout))
            record("version", {"returncode": version.returncode, "stdout": version.stdout, "stderr": version.stderr})
            if version.returncode:
                raise ProtocolError("Codex --version failed")
            result["version"] = version.stdout.strip()
            process = subprocess.Popen([site["codex_bin"], "app-server"], cwd=output, env=environment,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, bufsize=1)
            for stream, channel in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                thread = threading.Thread(target=reader, args=(stream, channel), daemon=True)
                thread.start()
                readers.append(thread)
            rpc("initialize", {"clientInfo": {"name": "llmgo-rsi-connection-check", "title": "LLMGo connectivity", "version": "1.0.0"}}, deadline)
            send({"method": "initialized", "params": {}})
            if sandbox_check:
                shell = rpc("command/exec", {"command": ["/bin/sh", "-c", "printf READY"],
                    "cwd": str(output), "sandboxPolicy": turn_sandbox(site),
                    "timeoutMs": 10000}, deadline)
                result["sandbox_returncode"] = shell.get("exitCode")
                if shell.get("exitCode") != 0 or shell.get("stdout", "").strip() != "READY":
                    raise ProtocolError("Sandbox command failed: " + shell.get("stderr", ""))
                result["shell_execution_tested"] = True
            params = {"cwd": str(output), "approvalPolicy": "never", "sandbox": thread_sandbox(site), "ephemeral": True,
                      "developerInstructions": "Connectivity test only. Do not use tools, run commands, read files or modify files. Return the requested JSON with summary READY."}
            if site.get("meta_model"):
                params["model"] = site["meta_model"]
            thread = rpc("thread/start", params, deadline)
            result["protocol_passed"] = True
            if offline:
                result.update(ok=True, status="offline_passed", note="CLI and app-server protocol passed; no API request was made")
            else:
                turn = rpc("turn/start", {"threadId": thread["thread"]["id"], "cwd": str(output),
                    "approvalPolicy": "never", "sandboxPolicy": turn_sandbox(site),
                    "outputSchema": FINAL_SCHEMA,
                    "input": [{"type": "text", "text": 'Do not use tools. Return exactly {"status":"needs_input","summary":"READY","proposal_path":null}.'}]}, deadline)
                turn_id = turn["turn"]["id"]
                answers = []
                while True:
                    event = pending.popleft() if pending else receive(deadline)
                    params = event.get("params", {})
                    if event.get("method") == "item/completed":
                        item = params.get("item", {})
                        if item.get("type") == "agentMessage":
                            answers.append(item.get("text", ""))
                    if event.get("method") == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                        completed = params["turn"]
                        if completed.get("status") != "completed":
                            raise ProtocolError(json.dumps(completed.get("error") or completed, ensure_ascii=False))
                        answers.extend(item.get("text", "") for item in completed.get("items", []) if item.get("type") == "agentMessage")
                        valid = False
                        for answer in answers:
                            try:
                                value = json.loads(answer)
                            except (ValueError, TypeError):
                                continue
                            valid = valid or value == {"status": "needs_input", "summary": "READY", "proposal_path": None}
                        if not valid:
                            raise ProtocolError("Completed turn did not return the required READY JSON")
                        result.update(ok=True, status="passed", inference_tested=True, reply="READY")
                        break
        except Exception as error:
            message = scrub(str(error), secret)
            result.update(status="failed", error=type(error).__name__, detail=message, hint=failure_hint(message))
        finally:
            if process is not None:
                if process.stdin:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()  # only this test's child
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                for thread in readers:
                    thread.join(timeout=2)
                while not events.empty():
                    channel, message = events.get_nowait()
                    record(channel, message)
                if not result['ok']:
                    result['process_returncode'] = process.returncode
                    result['stderr_tail'] = ''.join(stderr_tail)[-7000:]
            save()
    return scrub(result, secret)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test Meta-Agent credentials from the configured provider token or environment variable")
    parser.add_argument("--site", default=str(REPO / "configs/rsi/site.beijing.json"))
    parser.add_argument("--offline", action="store_true", help="Check CLI and app-server protocol without an API key or model request")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--sandbox-check", action="store_true", help="Also execute a harmless shell command through app-server using the research isolation policy; does not test model tool selection")
    args = parser.parse_args(argv)
    site = load_site(args.site)
    name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    output = Path(site["user_root"]) / "rsi_readiness/meta-connection" / name
    try:
        result = check_connection(site, output, offline=args.offline, timeout=args.timeout, sandbox_check=args.sandbox_check)
    except Exception as error:
        result = {"ok": False, "status": "configuration_error", "error": type(error).__name__,
                  "detail": scrub(str(error), os.environ.get("OPENAI_API_KEY", ""))}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1
