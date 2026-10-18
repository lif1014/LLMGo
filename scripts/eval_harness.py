#!/usr/bin/env python3
"""用 Harness 和 Fairy-Stockfish 下十分局。一張 GPU 一份模型，有幾張卡就同時下幾局。"""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL_PATH = Path("/user/liyaohu4170/PostTrainBench/protocol.json")
PIP_INDEX = os.environ.get("CHESS_PTB_PIP_INDEX", "https://mirrors.ustc.edu.cn/pypi/simple")
PROXY = os.environ.get(
    "CHESS_PTB_HTTPS_PROXY",
    "http://whitelist-proxy.cybertron.svc.cluster.local:7891",
)
_LOG_FILE = None


class _Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return False


def _output_from_argv(argv: list[str]) -> Path | None:
    if "--output" not in argv:
        return None
    index = argv.index("--output")
    if index + 1 >= len(argv):
        return None
    return Path(argv[index + 1])


def _start_log(argv: list[str]) -> Path:
    global _LOG_FILE
    output = _output_from_argv(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    if output is not None:
        path = output.resolve().parent / f"eval-harness-{stamp}.log"
    else:
        path = ROOT / "runs" / f"eval-harness-{stamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    _LOG_FILE = handle
    print(f"log={path}", flush=True)
    print("==== eval harness ====", flush=True)
    print(f"time={datetime.now(timezone.utc).isoformat()}", flush=True)
    print(f"host={socket.gethostname()}", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)
    print(f"python={sys.executable}", flush=True)
    print(f"version={sys.version.split()[0]}", flush=True)
    print(f"argv={' '.join(argv)}", flush=True)
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    return path


def _append_no_proxy(item: str) -> None:
    current = os.environ.get("no_proxy", "")
    parts = [part.strip() for part in current.split(",") if part.strip()]
    if item not in parts:
        parts.append(item)
    os.environ["no_proxy"] = ",".join(parts)
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


def _ensure_proxy() -> None:
    """叢集節點沒有直連公網。pip 走和 PostTrainBench 同一條白名單代理，本機 vLLM 不走代理。"""
    os.environ["http_proxy"] = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or PROXY
    os.environ["https_proxy"] = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or PROXY
    os.environ["HTTP_PROXY"] = os.environ["http_proxy"]
    os.environ["HTTPS_PROXY"] = os.environ["https_proxy"]
    if not os.environ.get("no_proxy"):
        os.environ["no_proxy"] = os.environ.get("NO_PROXY") or ".paratera.com,.thuwayinfo.com,.modelbest.co"
    for item in ("127.0.0.1", "localhost"):
        _append_no_proxy(item)
    print(f"http_proxy={os.environ['http_proxy']}", flush=True)
    print(f"https_proxy={os.environ['https_proxy']}", flush=True)
    print(f"no_proxy={os.environ['no_proxy']}", flush=True)


def _missing_packages() -> list[str]:
    missing = []
    for package, module in (("python-chess", "chess"), ("httpx", "httpx"), ("tokenizers", "tokenizers")):
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(package)
    return missing


def _ensure_packages() -> None:
    missing = _missing_packages()
    if not missing:
        print("deps=already-installed python-chess httpx tokenizers", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade-strategy",
        "only-if-needed",
        "-i",
        PIP_INDEX,
        *missing,
    ]
    print(f"deps=install {' '.join(missing)}", flush=True)
    print(f"pip={' '.join(command)}", flush=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(f"安裝依賴失敗，結束碼 {completed.returncode}")
    still = _missing_packages()
    if still:
        raise SystemExit(f"安裝後仍無法載入：{', '.join(still)}")
    print("deps=ok", flush=True)


def _import_runtime() -> None:
    global VllmAsker, EngineSeat, HarnessRunner, HarnessSeat, UciPlayer, TrajectoryStore, HarnessConfig
    from llmgo.harness.config import HarnessConfig as _HarnessConfig
    from llmgo.harness.player import VllmAsker as _VllmAsker
    from llmgo.runner.harness_loop import EngineSeat as _EngineSeat
    from llmgo.runner.harness_loop import HarnessRunner as _HarnessRunner
    from llmgo.runner.harness_loop import HarnessSeat as _HarnessSeat
    from llmgo.runtime.uci_player import UciPlayer as _UciPlayer
    from llmgo.store.jsonl import TrajectoryStore as _TrajectoryStore

    VllmAsker = _VllmAsker
    EngineSeat = _EngineSeat
    HarnessRunner = _HarnessRunner
    HarnessSeat = _HarnessSeat
    UciPlayer = _UciPlayer
    TrajectoryStore = _TrajectoryStore
    HarnessConfig = _HarnessConfig


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


def has_gpu() -> bool:
    return bool(visible_gpu_ids())


def choose_gpu_ids(available: list[str], games: int, requested: int | None) -> list[str]:
    """每張卡一份模型。用的張數不超過可見的卡，也不超過對局數。"""
    if games < 1:
        raise ValueError("沒有對局可下")
    if requested is not None and requested < 1:
        raise ValueError("--gpus 必須至少是 1")
    if requested is not None and requested > len(available):
        raise ValueError(f"要求 {requested} 張 GPU，這台機器只看到 {len(available)} 張")
    count = len(available) if requested is None else requested
    count = min(count, games)
    if count < 1:
        raise ValueError("這台機器沒有 GPU")
    return available[:count]


def assign_jobs(seats: list[str], urls: list[str]) -> list[dict]:
    return [
        {"model_seat": seat, "index": index, "url": urls[index % len(urls)]}
        for index, seat in enumerate(seats)
    ]


def load_protocol(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("framework") != "harness" or data.get("enable_thinking") is not True:
        raise ValueError("Protocol must use framework=harness and enable_thinking=true")
    if not 1 <= int(data["max_plies"]) <= 200:
        raise ValueError("Chess harness max_plies must be between 1 and 200")
    if not 1 <= int(data["max_tokens"]) <= 32768:
        raise ValueError("Chess harness max_tokens must be between 1 and 32768")
    return data


def harness_manifest() -> dict:
    root = ROOT / "src" / "llmgo" / "harness"
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def split_seats(protocol: dict, games: int | None) -> tuple[int, int]:
    if games is None or games == int(protocol["games"]):
        return int(protocol["white"]), int(protocol["black"])
    if games < 1:
        raise ValueError("--games 必須至少是 1")
    return (games + 1) // 2, games // 2


def model_points(result: str | None, seat: str) -> float | None:
    if result == "draw":
        return 0.5
    if result == f"{seat}_win":
        return 1.0
    if isinstance(result, str) and result.endswith("_win"):
        return 0.0
    return None


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
    def __init__(self, model_path: Path, protocol: dict, log_path: Path, port: int, gpu_id: str) -> None:
        self.model_path = model_path
        self.protocol = protocol
        self.log_path = log_path
        self.port = port
        self.gpu_id = gpu_id
        self.proc: subprocess.Popen | None = None
        self.log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        max_tokens = int(self.protocol["max_tokens"])
        max_len = max(max_tokens + 8192, 40960)
        cmd = [
            "vllm",
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
        self.proc = subprocess.Popen(
            cmd,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    def ready_now(self) -> bool:
        if self.proc is not None and self.proc.poll() is not None:
            return False
        url = f"{self.base_url}/models"
        name = str(self.protocol["model_name"])
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
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


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


async def _play_one(job: dict, protocol: dict, base_url: str, engine: Path, runs: Path) -> dict:
    game_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    store = TrajectoryStore(runs / game_id)
    model_seat = job["model_seat"]
    engine_seat = "black" if model_seat == "white" else "white"
    harness_config = HarnessConfig.from_dict(protocol.get("harness"))
    store.write_config(
        {
            "id": game_id,
            "eval": "harness",
            "game": "chess",
            "model_seat": model_seat,
            "model": protocol["model_name"],
            "model_path": protocol.get("model_path"),
            "vllm_base_url": base_url,
            "batch_id": protocol["batch_id"],
            "protocol": protocol,
            "seed": int(protocol.get("seed", 0)) + job["index"],
            "harness_artifact": protocol.get("harness_artifact"),
            "harness_config": harness_config.as_dict(),
        }
    )
    asker = VllmAsker(
        base_url,
        str(protocol["model_name"]),
        tokenizer_path=protocol.get('model_path'),
        temperature=float(protocol["temperature"]),
        top_p=float(protocol["top_p"]),
        top_k=int(protocol["top_k"]),
        presence_penalty=float(protocol["presence_penalty"]),
        timeout_sec=harness_config.request_timeout_sec,
    )
    uci = UciPlayer(
        engine_seat,
        str(engine),
        skill=int(protocol["skill"]),
        depth=int(protocol["depth"]),
        movetime=float(protocol["movetime_sec"]),
        name=f"fairy-sf-{engine_seat}",
    )
    seats = {
        model_seat: HarnessSeat(model_seat, asker, max_tokens=int(protocol["max_tokens"]), config=harness_config, seed=int(protocol.get("seed", 0)) + job["index"]),
        engine_seat: EngineSeat(uci),
    }
    runner = HarnessRunner(seats["white"], seats["black"], store, max_plies=int(protocol["max_plies"]))

    async def emit(event: dict) -> None:
        if event.get("type") == "position" and event.get("last_move"):
            move = event["last_move"]
            print(f"{game_id} {move['seat']} {move['uci']}", flush=True)
        elif event.get("type") == "attempt" and event.get("source") == "random":
            print(f"{game_id} random {event.get('seat')} {event.get('uci')}", flush=True)

    started = time.time()
    print(f"start {game_id} model={model_seat}", flush=True)
    try:
        summary = await runner.run(emit)
    finally:
        uci.close()
    elapsed = time.time() - started
    print(
        f"done {game_id} result={summary.get('result')} reason={summary.get('result_reason')} "
        f"random={summary.get('random_moves')} seconds={elapsed:.0f}",
        flush=True,
    )
    return {
        "game_id": game_id,
        "result": summary.get("result"),
        "result_reason": summary.get("result_reason"),
        "model_seat": model_seat,
        "seconds": elapsed,
        "index": job["index"],
        "random_moves": summary.get("random_moves") or 0,
        "tool_uses": summary.get("tool_uses") or {},
        "diagnostics": {key: summary.get(key) for key in ("model_turns", "model_decisions", "model_choices", "model_choice_rate", "forced_moves", "random_rate", "fallback_reasons", "request_count", "completion_tokens", "prompt_tokens", "tokens_exact", "request_errors", "length_count", "latency_ms")},
    }


def collect_metrics(protocol: dict, games: list[dict], model_path: str | None) -> dict:
    wins = draws = losses = forfeits = 0
    score = 0.0
    scored = 0
    random_moves = 0
    tool_uses: Counter[str] = Counter()
    rows = []
    for item in games:
        seat = item.get("model_seat")
        points = model_points(item.get("result"), seat)
        if points is None:
            raise RuntimeError(f"Match {item.get('game_id')} has no valid result ({item.get('result_reason')}); no partial batch score is written")
        random_moves += int(item.get("random_moves") or 0)
        tool_uses.update(item.get("tool_uses") or {})
        if points is not None:
            scored += 1
            score += points
            if points == 1:
                wins += 1
            elif points == 0.5:
                draws += 1
            else:
                losses += 1
            if item.get("result_reason") == "illegal_forfeit":
                forfeits += 1
        rows.append(
            {
                "game_id": item.get("game_id"),
                "result": item.get("result"),
                "result_reason": item.get("result_reason"),
                "model_seat": seat,
                "seconds": item.get("seconds"),
                "random_moves": item.get("random_moves"),
                "tool_uses": item.get("tool_uses"),
                "diagnostics": item.get("diagnostics") or {},
            }
        )
    return {
        "primary_score": (score / scored) if scored else None,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "forfeits": forfeits,
        "random_moves": random_moves,
        "tool_uses": dict(tool_uses),
        "games": rows,
        "protocol": protocol,
        "model_path": model_path,
        "gpus": None,
        "harness": True,
        "harness_artifact": protocol.get("harness_artifact"),
        "model_turns": sum((g.get("diagnostics") or {}).get("model_turns", 0) or 0 for g in games),
        "model_decisions": sum((g.get("diagnostics") or {}).get("model_decisions", 0) or 0 for g in games),
        "forced_moves": sum((g.get("diagnostics") or {}).get("forced_moves", 0) or 0 for g in games),
        "request_count": sum((g.get("diagnostics") or {}).get("request_count", 0) or 0 for g in games),
        "completion_tokens": sum((g.get("diagnostics") or {}).get("completion_tokens", 0) or 0 for g in games),
        "fallback_reasons": dict(sum((Counter((g.get("diagnostics") or {}).get("fallback_reasons") or {}) for g in games), Counter())),
    }


async def _worker(url: str, jobs: list[dict], protocol: dict, engine: Path, runs: Path) -> list[dict]:
    done = []
    for job in jobs:
        done.append(await _play_one(job, protocol, url, engine, runs))
    return done


async def _run_games(protocol: dict, urls: list[str], engine: Path, runs: Path, white: int, black: int) -> list[dict]:
    jobs = assign_jobs(["white"] * white + ["black"] * black, urls)
    grouped: dict[str, list[dict]] = {url: [] for url in urls}
    for job in jobs:
        grouped[job["url"]].append(job)
    parts = await asyncio.gather(*(_worker(url, grouped[url], protocol, engine, runs) for url in urls if grouped[url]))
    rows = [item for part in parts for item in part]
    rows.sort(key=lambda item: item["index"])
    return rows


def main() -> int:
    _import_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--base-url", action="append", default=[])
    parser.add_argument("--gpus", type=int, help="使用的 GPU 張數。不填就用全部可見的卡，一張卡同時一局。")
    parser.add_argument("--games", type=int)
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    parser.add_argument("--batch")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    if not protocol_path.is_file():
        return _fail(f"找不到對局條件：{protocol_path}")
    protocol = load_protocol(protocol_path)
    if args.model_path:
        protocol['model_path'] = str(Path(args.model_path).resolve())
    HarnessConfig.from_dict(protocol.get("harness"))
    protocol["batch_id"] = args.batch or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    protocol["harness_artifact"] = harness_manifest()
    output = Path(args.output).resolve()
    engine = Path(protocol["engine"])
    if not os.access(engine, os.X_OK):
        return _fail(f"引擎不能執行：{engine}")
    try:
        white, black = split_seats(protocol, args.games)
    except ValueError as exc:
        return _fail(str(exc))

    servers: list[VllmServer] = []
    base_urls = [item.rstrip("/") for item in args.base_url if item.strip()]
    runs = output.parent / "harness_runs"
    try:
        if not base_urls:
            available = visible_gpu_ids()
            if not available:
                return _fail("這台機器沒有 GPU，也沒有 --base-url。評測需要在有 GPU 的機器上啟動 vLLM。")
            if not args.model_path:
                return _fail("要啟動 vLLM 時請給 --model-path。")
            model_path = Path(args.model_path).resolve()
            if not model_path.is_dir() or not (model_path / "config.json").is_file():
                return _fail(f"沒有可評的模型目錄：{model_path}")
            try:
                gpu_ids = choose_gpu_ids(available, white + black, args.gpus)
            except ValueError as exc:
                return _fail(str(exc))
            runs.mkdir(parents=True, exist_ok=True)
            ports = _pick_ports(int(os.environ.get("CHESS_PTB_VLLM_PORT", "8000")), len(gpu_ids))
            for index, (gpu_id, port) in enumerate(zip(gpu_ids, ports)):
                server = VllmServer(model_path, protocol, runs / "service_logs" / f"{protocol['batch_id']}-vllm-{index}.log", port, gpu_id)
                servers.append(server)
                server.start()
            timeout = float(os.environ.get("CHESS_PTB_VLLM_READY_TIMEOUT", "600"))
            message = _wait_servers(servers, timeout)
            if message:
                return _fail(message)
            base_urls = [server.base_url for server in servers]
            model_for_metrics = str(model_path)
            print(f"gpus={len(gpu_ids)} games={white + black}", flush=True)
        else:
            runs.mkdir(parents=True, exist_ok=True)
            model_for_metrics = args.model_path
            print(f"servers={len(base_urls)} games={white + black}", flush=True)
        games = asyncio.run(_run_games(protocol, base_urls, engine, runs, white, black))
        metrics = collect_metrics(protocol, games, model_for_metrics)
        metrics['model_choices'] = metrics['model_turns'] - metrics['forced_moves']
        metrics['model_choice_rate'] = metrics['model_decisions'] / metrics['model_choices'] if metrics['model_choices'] else None
        metrics["random_rate"] = metrics["random_moves"] / metrics["model_turns"] if metrics["model_turns"] else None
        metrics["gpus"] = len(base_urls)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        batches = output.parent / "harness_batches"
        batches.mkdir(parents=True, exist_ok=True)
        (batches / f"{protocol['batch_id']}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"primary_score={metrics['primary_score']} output={output}")
        return 0
    finally:
        for server in servers:
            server.stop()


if __name__ == "__main__":
    _start_log(sys.argv)
    try:
        _ensure_proxy()
        _ensure_packages()
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        if _LOG_FILE is not None:
            _LOG_FILE.flush()
