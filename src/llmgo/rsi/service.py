"""Owned vLLM child lifecycle; no task submission or package installation."""
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

def parse_gpu_list(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        if line.startswith("GPU ") and ":" in line:
            found.append(line.split(":", 1)[0].replace("GPU", "", 1).strip())
    return found

def visible_gpu_ids() -> list[str]:
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if pinned is not None:
        if pinned.strip() in {"", "-1"}:
            return []
        return [item.strip() for item in pinned.split(",") if item.strip()]
    try:
        completed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if completed.returncode != 0:
        return []
    return parse_gpu_list(completed.stdout)

def _pick_ports(preferred: int, count: int) -> list[int]:
    import socket

    chosen = []
    held = []
    port = preferred
    limit = preferred + max(40, count * 5)
    while len(chosen) < count and port < limit:
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            sock.close()
            port += 1
            continue
        chosen.append(port)
        held.append(sock)
        port += 1
    for sock in held:
        sock.close()
    if len(chosen) < count:
        raise RuntimeError(f"從 {preferred} 起沒有 {count} 個空的本機埠")
    return chosen

class VllmServer:
    def __init__(self, model_path: Path, protocol: dict, log_path: Path, port: int, gpu_id: str, python: str | None = None) -> None:
        self.model_path = model_path
        self.protocol = protocol
        self.log_path = log_path
        self.port = port
        self.gpu_id = gpu_id
        self.python = python
        self.proc: subprocess.Popen | None = None
        self.log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        max_tokens = int(self.protocol["max_tokens"])
        max_len = max(max_tokens + 8192, 40960)
        cmd = ([self.python, "-I", "-m", "vllm.entrypoints.cli.main"] if self.python else ["vllm"]) + [
            "serve",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served-model-name",
            str(self.protocol["model_name"]),
            "--language-model-only",
            "--reasoning-parser",
            "qwen3",
            "--enforce-eager",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            str(max_len),
            "--gpu-memory-utilization",
            os.environ.get("CHESS_PTB_GPU_MEMORY_UTILIZATION", "0.90"),
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[3],
            env=env,
        )

    def ready_now(self) -> bool:
        if self.proc is not None and self.proc.poll() is not None:
            return False
        url = f"{self.base_url}/models"
        name = str(self.protocol["model_name"])
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=2) as response:
                body = response.read().decode("utf-8", "replace")
            return name in body
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
        if self.log is not None:
            self.log.close()
            self.log = None

def _wait_servers(servers: list[VllmServer], timeout: float) -> str | None:
    deadline = time.time() + timeout
    pending = list(servers)
    while pending and time.time() < deadline:
        still = []
        for server in pending:
            if server.proc is not None and server.proc.poll() is not None:
                return f"vLLM 沒有就緒。日誌：{server.log_path}"
            if not server.ready_now():
                still.append(server)
        pending = still
        if pending:
            time.sleep(2)
    if pending:
        return f"vLLM 沒有就緒。日誌：{pending[0].log_path}"
    return None
