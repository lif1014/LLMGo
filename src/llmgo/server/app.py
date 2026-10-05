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
from pydantic import BaseModel

from llmgo.env.xiangqi import BLACK, RED, Position
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
RUNS = ROOT / "runs"


def load_config() -> dict:
    path = Path(os.environ.get("LLMGO_CONFIG", ROOT / "configs" / "demo.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class NewGame(BaseModel):
    backend: str = "vllm"
    record_video: bool = False
    prompt_policy: str = "blind"
    max_plies: int = 300
    max_retries: int = 2
    temperature: float = 0.9
    red_temperature: float | None = None
    black_temperature: float | None = None
    vllm_base_url: str | None = None
    max_tokens: int = 16384


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
        self.position = Position.initial()
        self.last_attempt: dict | None = None
        self.finished = False
        self.summary: dict | None = None

    async def emit(self, event: dict) -> None:
        if event["type"] == "attempt":
            self.last_attempt = event
            if self.request.record_video and not event.get("ok"):
                self._save_frame(self.position, event)
        elif event["type"] == "position":
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


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
async def get_config():
    return load_config()


@app.get("/api/reports")
async def reports():
    summaries = load_runs(RUNS)
    return {"games": summaries, "aggregate": aggregate(summaries)}


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


def _public_event(event: dict) -> dict | None:
    kind = event.get("type")
    if kind not in {"position", "attempt", "summary", "game_finished", "turn_started"}:
        return None
    shown = dict(event)
    if kind in {"attempt", "turn_started"} and shown.get("prompt") and not shown.get("llm_input"):
        shown["llm_input"] = llm_input_text(shown["seat"], shown["prompt"])
    return shown


@app.get("/api/games/{game_id}/events")
async def game_events(game_id: str):
    game = hub.games.get(game_id)
    if game is not None:
        events = [item for event in game.events if (item := _public_event(event)) is not None]
        return {"id": game_id, "live": bool(game.task and not game.task.done()), "events": events}
    path = RUNS / game_id / "events.jsonl"
    if not path.exists():
        raise HTTPException(404, "没有这局")
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = _public_event(json.loads(line))
        if item is not None:
            events.append(item)
    return {"id": game_id, "live": False, "events": events}


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    game = hub.games.get(game_id)
    if game is None:
        summary_path = RUNS / game_id / "summary.json"
        if not summary_path.exists():
            raise HTTPException(404, "没有这局")
        return {"id": game_id, "summary": json.loads(summary_path.read_text(encoding="utf-8")), "live": False}
    return {
        "id": game.id,
        "live": game.task is not None and not game.task.done(),
        "finished": game.finished,
        "summary": game.summary,
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
    body = game.request
    config = game.config
    red_temp = body.temperature if body.red_temperature is None else body.red_temperature
    black_temp = body.temperature if body.black_temperature is None else body.black_temperature
    base_url = body.vllm_base_url or config["vllm_base_url"]
    if body.backend == "scripted":
        red = ScriptedPlayer(RED, "legal")
        black = ScriptedPlayer(BLACK, "legal")
    elif body.backend == "vllm":
        shared = dict(
            base_url=base_url,
            model=config["model_name"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            max_tokens=body.max_tokens,
            presence_penalty=config["presence_penalty"],
        )
        red = VllmPlayer(RED, temperature=red_temp, **shared)
        black = VllmPlayer(BLACK, temperature=black_temp, **shared)
    else:
        await game.emit({"type": "game_finished", "result": "aborted", "result_reason": "unknown_backend", "summary": None})
        return
    runner = GameRunner(
        red,
        black,
        game.store,
        max_plies=body.max_plies,
        max_retries=body.max_retries,
        policy=body.prompt_policy,
    )
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
        game._finish_video()
