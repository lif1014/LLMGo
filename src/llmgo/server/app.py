"""对局网页。服务跑在这台机器上，浏览器通过端口访问。"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from llmgo.env.games import get_game as load_rules
from llmgo.env.xiangqi import Position
from llmgo.eval.report import aggregate, load_runs
from llmgo.render.frames import draw_attempt
from llmgo.render.video import write_gif
from llmgo.runner.loop import GameRunner
from llmgo.runtime.prompt import llm_input_text
from llmgo.runtime.scripted import ScriptedPlayer
from llmgo.runtime.vllm_player import VllmPlayer
from llmgo.store.jsonl import TrajectoryStore

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
RUNS = Path(os.environ.get("LLMGO_RUNS", ROOT / "runs"))


def load_config() -> dict:
    path = Path(os.environ.get("LLMGO_CONFIG", ROOT / "configs" / "demo.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class NewGame(BaseModel):
    backend: str = "vllm"
    record_video: bool = False
    prompt_policy: str = "blind"
    max_plies: int = Field(default=200, ge=1, le=600)
    max_retries: int = 2
    temperature: float = 0.9
    red_temperature: float | None = None
    black_temperature: float | None = None
    vllm_base_url: str | None = None
    max_tokens: int = Field(default=32768, ge=1, le=32768)
    game: str = "xiangqi"
    opponent: str = "engine"
    model_seat: str = "white"
    harness: dict | None = None


def chess_protocol() -> dict:
    path = Path(os.environ.get("LLMGO_CHESS_PROTOCOL", ROOT.parent / "PostTrainBench" / "protocol.json"))
    return json.loads(path.read_text(encoding="utf-8"))


async def harness_library():
    from llmgo.harness.config import HarnessConfig
    from llmgo.harness.skills import load_cards
    from llmgo.harness.tools import TOOL_LIBRARY
    protocol = chess_protocol()
    return {"tools": TOOL_LIBRARY, "skills": load_cards(), "enable_thinking": True,
            "max_plies": 200, "config": HarnessConfig.from_dict(protocol.get("harness")).as_dict()}


class GameSession:
    def __init__(self, game_id: str, config: dict, request: NewGame):
        self.id = game_id
        self.config = config
        self.request = request
        self.store = TrajectoryStore(RUNS / game_id)
        self.events: list[dict] = []
        self.sockets: set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.task: asyncio.Task | None = None
        self.frames = []
        self.frame_paths: list[str] = []
        self.rules = load_rules(request.game)
        self.position = self.rules.initial() if self.rules.name == "xiangqi" else None
        self.last_attempt: dict | None = None
        self.finished = False
        self.summary: dict | None = None

    async def emit(self, event: dict) -> None:
        if event["type"] == "attempt":
            self.last_attempt = event
            if self.rules.name == "xiangqi" and self.request.record_video and not event.get("ok"):
                self._save_frame(self.position, event)
        elif event["type"] == "position" and self.rules.name == "xiangqi":
            self.position = Position.from_fen(event["fen"])
            if self.request.record_video and event.get("last_move") and self.last_attempt:
                self._save_frame(self.position, self.last_attempt, event["last_move"])
        elif event["type"] == "summary":
            self.summary = event["game"]
        elif event["type"] == "game_finished":
            self.finished = True
            self.summary = event.get("summary")
            self._finish_video()
        async with self.lock:
            self.events.append(event)
            sockets = list(self.sockets)
        dead = []
        for socket in sockets:
            try:
                await socket.send_json(event)
            except Exception:
                dead.append(socket)
        if dead:
            async with self.lock:
                for socket in dead:
                    self.sockets.discard(socket)

    def _save_frame(self, pos: Position, attempt: dict, last: dict | None = None) -> None:
        error = None if attempt.get("ok") else attempt.get("error_detail")
        image = draw_attempt(
            pos,
            attempt["seat"],
            (last or {}).get("uci") or attempt.get("uci"),
            (last or {}).get("zh") or attempt.get("zh"),
            attempt.get("reasoning") or "",
            attempt.get("content") or "",
            error,
        )
        folder = self.store.root / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{len(self.frames):04d}.png"
        image.save(path)
        self.frames.append(image)
        self.frame_paths.append(str(path))

    def _finish_video(self) -> None:
        if not self.frames:
            return
        gif_path = self.store.root / "game.gif"
        with suppress(Exception):
            write_gif(self.frames, gif_path)


class Hub:
    def __init__(self):
        self.games: dict[str, GameSession] = {}
        self.active: str | None = None


hub = Hub()
app = FastAPI(title="LLMGo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.get("/api/harness")(harness_library)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
async def get_config():
    return load_config()


@app.get("/api/reports")
async def reports():
    summaries = load_runs(RUNS)
    grouped: dict[str, list] = {}
    for item in summaries:
        grouped.setdefault(item.get("game") or "xiangqi", []).append(item)
    by_game = {name: {"games": items, "aggregate": aggregate(items)} for name, items in grouped.items()}
    return {"games": summaries, "aggregate": aggregate(summaries), "by_game": by_game}


@app.get("/api/vllm/health")
async def vllm_health():
    config = load_config()
    url = config["vllm_base_url"].rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(url)
        return {"ok": response.status_code < 400, "status": response.status_code, "url": url}
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def _saved_meta(folder: Path | None) -> dict:
    if folder is None:
        return {}
    path = folder / "config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {"model_seat": data.get("model_seat"), "model": data.get("model")}


def _saved_game_dir(game_id: str) -> Path | None:
    if "/" in game_id or game_id in {"", ".", ".."}:
        return None
    for folder in (RUNS / game_id, RUNS / "harness_runs" / game_id):
        if (folder / "events.jsonl").is_file() or (folder / "summary.json").is_file():
            return folder
    return None


def _public_event(event: dict) -> dict | None:
    kind = event.get("type")
    if kind not in {"position", "attempt", "summary", "game_finished", "turn_started", "choice", "model_request", "tool_call", "fallback", "route_recovery"}:
        return None
    shown = dict(event)
    if kind in {"attempt", "turn_started"} and shown.get("prompt") and not shown.get("llm_input"):
        shown["llm_input"] = llm_input_text(shown["seat"], shown["prompt"])
    return shown


@app.get("/api/games/{game_id}/events")
async def game_events(game_id: str, after: int = 0):
    if after < 0:
        raise HTTPException(400, "after 必須非負")
    game = hub.games.get(game_id)
    if game is not None:
        events = [item for event in game.events[after:] if (item := _public_event(event)) is not None]
        return {"id": game_id, "live": bool(game.task and not game.task.done()), "events": events, "cursor": len(game.events)}
    folder = _saved_game_dir(game_id)
    path = folder / "events.jsonl" if folder is not None else None
    if path is None or not path.is_file():
        raise HTTPException(404, "没有这局")
    events = []
    cursor = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                break  # writer may still be appending the last JSON record
            cursor = index
            if index <= after or not line.strip():
                continue
            item = _public_event(json.loads(line))
            if item is not None:
                events.append(item)
    meta = _saved_meta(folder)
    return {"id": game_id, "live": not (folder / "summary.json").is_file(), "external": True,
            "events": events, "cursor": cursor, **meta}


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    game = hub.games.get(game_id)
    if game is None:
        folder = _saved_game_dir(game_id)
        summary_path = folder / "summary.json" if folder is not None else None
        if summary_path is None or not summary_path.is_file():
            if folder is None:
                raise HTTPException(404, "没有这局")
            return {"id": game_id, "summary": None, "live": True, "external": True, **_saved_meta(folder)}
        return {"id": game_id, "summary": json.loads(summary_path.read_text(encoding="utf-8")), "live": False}
    return {
        "id": game.id,
        "live": game.task is not None and not game.task.done(),
        "finished": game.finished,
        "summary": game.summary,
        "model_seat": game.request.model_seat if game.request.game == "chess" and game.request.opponent == "engine" else None,
        "model": game.config.get("model_name"),
        "video": str(game.store.root / "game.gif") if (game.store.root / "game.gif").exists() else None,
    }


@app.post("/api/games/{game_id}/stop")
async def stop_game(game_id: str):
    game = hub.games.get(game_id)
    if game is None or game.task is None:
        raise HTTPException(404, "没有正在下的这一局")
    game.task.cancel()
    return {"ok": True}


@app.post("/api/games")
async def start_game(body: NewGame):
    try:
        load_rules(body.game)
    except KeyError:
        raise HTTPException(400, "没有这种游戏")
    if body.backend == "harness" and body.game != "chess":
        raise HTTPException(400, "Harness 目前只下國際象棋")
    if body.game == "chess":
        if body.max_plies > 200:
            raise HTTPException(400, "國際象棋最多 200 半步")
        if body.backend == "vllm":
            body.backend = "harness"
        if body.opponent not in {"engine", "model"} or body.model_seat not in {"white", "black"}:
            raise HTTPException(400, "無效的對手或模型座位")
        from llmgo.harness.config import HarnessConfig
        try:
            HarnessConfig.from_dict(body.harness or chess_protocol().get("harness"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if body.backend == "harness" and body.opponent == "engine" and not os.access(chess_protocol()["engine"], os.X_OK):
            raise HTTPException(400, "Fairy-Stockfish 不能執行；請核對伺服器引擎路徑及權限")
    if hub.active:
        current = hub.games.get(hub.active)
        if current and current.task and not current.task.done():
            raise HTTPException(409, "已经有一局在下")
    config = load_config()
    game_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:4]
    game = GameSession(game_id, config, body)
    game.store.write_config({"id": game_id, "request": body.model_dump(), "server": config})
    hub.games[game_id] = game
    hub.active = game_id
    game.task = asyncio.create_task(_drive(game))
    return {"id": game_id}


@app.websocket("/ws/games/{game_id}")
async def watch(socket: WebSocket, game_id: str):
    game = hub.games.get(game_id)
    if game is None:
        await socket.close(code=4404)
        return
    await socket.accept()
    async with game.lock:
        await socket.send_json({"type": "replay_begin"})
        # 历史 token 合并成一块再送，避免刷新时把几万条逐字事件堵在锁里。
        pending: dict[tuple[str, str], str] = {}

        async def flush_tokens() -> None:
            for (seat, channel), text in pending.items():
                if text:
                    await socket.send_json({"type": "token", "seat": seat, "channel": channel, "text": text})
            pending.clear()

        for event in game.events:
            if event.get("type") == "token":
                key = (event.get("seat") or "", event.get("channel") or "")
                pending[key] = pending.get(key, "") + (event.get("text") or "")
                continue
            await flush_tokens()
            await socket.send_json(event)
        await flush_tokens()
        game.sockets.add(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        async with game.lock:
            game.sockets.discard(socket)


async def _drive(game: GameSession) -> None:
    engine_seat = None
    body = game.request
    config = game.config
    rules = game.rules
    first, second = rules.seats
    first_temp = body.temperature if body.red_temperature is None else body.red_temperature
    second_temp = body.temperature if body.black_temperature is None else body.black_temperature
    base_url = body.vllm_base_url or config["vllm_base_url"]
    if body.backend == "scripted":
        red = ScriptedPlayer(first, "legal")
        black = ScriptedPlayer(second, "legal")
        runner = GameRunner(
            red,
            black,
            game.store,
            max_plies=body.max_plies,
            max_retries=body.max_retries,
            policy=body.prompt_policy,
            rules=rules,
        )
    elif body.backend == "vllm":
        shared = dict(
            base_url=base_url,
            model=config["model_name"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            max_tokens=body.max_tokens,
            presence_penalty=config["presence_penalty"],
        )
        red = VllmPlayer(first, temperature=first_temp, **shared)
        black = VllmPlayer(second, temperature=second_temp, **shared)
        runner = GameRunner(
            red,
            black,
            game.store,
            max_plies=body.max_plies,
            max_retries=body.max_retries,
            policy=body.prompt_policy,
            rules=rules,
        )
    elif body.backend == "harness":
        from llmgo.harness.player import VllmAsker
        from llmgo.harness.config import HarnessConfig
        from llmgo.runner.harness_loop import EngineSeat, HarnessRunner, HarnessSeat
        from llmgo.runtime.uci_player import UciPlayer

        protocol = chess_protocol()
        harness_config = HarnessConfig.from_dict(body.harness or protocol.get("harness"))

        sampling = dict(
            base_url=base_url,
            model=config["model_name"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            presence_penalty=config["presence_penalty"],
            timeout_sec=harness_config.request_timeout_sec,
        )
        budget = body.max_tokens or 32768
        seats = {first: HarnessSeat(first, VllmAsker(temperature=first_temp, **sampling), max_tokens=budget, config=harness_config),
                 second: HarnessSeat(second, VllmAsker(temperature=second_temp, **sampling), max_tokens=budget, config=harness_config)}
        if body.opponent == "engine":
            opponent = "black" if body.model_seat == "white" else "white"
            engine_seat = EngineSeat(UciPlayer(opponent, protocol["engine"], skill=int(protocol["skill"]), depth=int(protocol["depth"]), movetime=float(protocol["movetime_sec"])))
            seats[opponent] = engine_seat
        game.store.write_config({"id": game.id, "request": body.model_dump(), "server": config,
                                 "game": "chess", "model_seat": body.model_seat if body.opponent == "engine" else None,
                                 "model": config["model_name"], "protocol": protocol, "harness": harness_config.as_dict()})
        runner = HarnessRunner(
            seats[first], seats[second],
            game.store,
            max_plies=body.max_plies,
        )
    else:
        await game.emit({"type": "game_finished", "result": "aborted", "result_reason": "unknown_backend", "summary": None})
        return
    try:
        await runner.run(game.emit)
    except asyncio.CancelledError:
        if not game.finished:
            await game.emit(
                {
                    "type": "game_finished",
                    "result": "aborted",
                    "result_reason": "stopped",
                    "summary": game.summary,
                }
            )
        raise
    finally:
        if engine_seat is not None:
            engine_seat.close()
        game._finish_video()
