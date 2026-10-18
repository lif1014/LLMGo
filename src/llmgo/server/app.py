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
from llmgo.eval.report import aggregate, load_runs, summarize_attempts, observed_summary
from llmgo.render.frames import draw_attempt
from llmgo.render.video import write_gif
from llmgo.runner.loop import GameRunner
from llmgo.runtime.prompt import llm_input_text
from llmgo.runtime.scripted import ScriptedPlayer
from llmgo.runtime.vllm_player import VllmPlayer
from llmgo.store.jsonl import TrajectoryStore
from llmgo.store.compact import backfill, compact_event, from_config

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
    return {"tools": TOOL_LIBRARY, "skills": load_cards(), "enable_thinking": True, "decision_mode": "single_choice",
            "max_plies": 200, "config": HarnessConfig.from_dict(protocol.get("harness")).as_dict()}


class GameSession:
    def __init__(self, game_id: str, config: dict, request: NewGame):
        self.id = game_id
        self.config = config
        self.request = request
        self.store = TrajectoryStore(RUNS / game_id)
        self.events: list[dict] = []
        self.pending_tokens: dict[tuple, dict] = {}
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
            if event['type'] == 'token':
                key = (event.get('request_id'), event.get('channel'))
                previous = self.pending_tokens.get(key, {}).get('text', '')
                text = event.get('text', '') if event.get('replace') else previous + event.get('text', '')
                self.pending_tokens[key] = {**event, 'text': text[-20000:], 'replace': True,
                                           'display_tail_only': len(text) > 20000}
                shown = event
            else:
                shown = self.store.latest_short
                if shown is None or shown.get('sequence') != event.get('sequence'):
                    shown = compact_event(event, self.store.windows)
                self.events.append(shown)
                if event['type'] == 'model_request' and event.get('status') == 'finished':
                    for key in list(self.pending_tokens):
                        if key[0] == event.get('request_id'):
                            del self.pending_tokens[key]
            sockets = list(self.sockets)
        dead = []
        for socket in sockets:
            try:
                await socket.send_json(shown)
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
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
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
        if any((folder / name).is_file() for name in ('events.jsonl', 'short_events.jsonl', 'summary.json')):
            return folder
    return None


def _public_event(event: dict) -> dict | None:
    kind = event.get("type")
    if kind not in {"position", "attempt", "summary", "game_finished", "turn_started", "choice", "model_request", "tool_call", "fallback", "route_recovery", "token"}:
        return None
    shown = dict(event)
    if kind in {"attempt", "turn_started"} and shown.get("prompt") and not shown.get("llm_input"):
        shown["llm_input"] = llm_input_text(shown["seat"], shown["prompt"])
    return shown


_short_locks: dict[Path, asyncio.Lock] = {}


async def _short_path(folder: Path) -> Path:
    path = folder / 'short_events.jsonl'
    if not path.exists():
        if not (folder / 'events.jsonl').is_file():
            raise HTTPException(404, '没有这局的事件记录')
        async with _short_locks.setdefault(folder, asyncio.Lock()):
            if not path.exists():
                await asyncio.to_thread(backfill, folder)
    return path


@app.get("/api/games/{game_id}/events")
async def game_events(game_id: str, after: int = 0, view: str = 'full', limit: int = 512):
    if after < 0:
        raise HTTPException(400, "after 必須非負")
    if view not in {'full', 'compact'} or not 1 <= limit <= 2048:
        raise HTTPException(400, 'view must be full/compact; limit must be 1..2048')
    game = hub.games.get(game_id)
    folder = game.store.root if game is not None else _saved_game_dir(game_id)
    if folder is None:
        raise HTTPException(404, '没有这局')
    if view == 'compact':
        path = await _short_path(folder)
        def read_page():
            events = []
            with path.open('rb') as handle:
                if after > 0:
                    handle.seek(after - 1)
                    if handle.read(1) != b'\n':
                        raise HTTPException(400, 'cursor is not a record boundary')
                handle.seek(after)
                while len(events) < limit:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line.endswith(b'\n'):
                        handle.seek(offset)
                        break
                    if line.strip() and (item := _public_event(json.loads(line))) is not None:
                        events.append(item)
                cursor = handle.tell()
                more = bool(handle.read(1))
            return events, cursor, more
        events, cursor, more = await asyncio.to_thread(read_page)
        return {'id': game_id, 'events': events, 'cursor': cursor, 'more': more, 'view': 'compact',
                'live': bool(game.task and not game.task.done()) if game else (folder / 'events.jsonl').exists() and not (folder / 'summary.json').exists(),
                **_saved_meta(folder)}
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


@app.get('/api/games/{game_id}/event')
async def raw_event(game_id: str, offset: int):
    folder = _saved_game_dir(game_id)
    path = folder / 'events.jsonl' if folder else None
    if path is None or not path.is_file():
        raise HTTPException(404, '完整事件只保留在产生对局的机器上')
    def read():
        with path.open('rb') as handle:
            if offset < 0 or offset >= path.stat().st_size:
                raise HTTPException(400, 'Invalid raw event offset')
            if offset:
                handle.seek(offset - 1)
                if handle.read(1) != b'\n':
                    raise HTTPException(400, 'Offset is not a record boundary')
            handle.seek(offset)
            line = handle.readline()
            if not line.endswith(b'\n'):
                raise HTTPException(409, '事件仍在写入')
            return json.loads(line)
    return await asyncio.to_thread(read)


@app.get('/api/games/{game_id}/raw')
async def raw_download(game_id: str):
    folder = _saved_game_dir(game_id)
    if folder is None or not (folder / 'events.jsonl').is_file():
        raise HTTPException(404, '完整事件只保留在产生对局的机器上')
    return FileResponse(folder / 'events.jsonl', filename=f'{game_id}-events.jsonl')


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    game = hub.games.get(game_id)
    if game is None:
        folder = _saved_game_dir(game_id)
        summary_path = folder / "summary.json" if folder is not None else None
        if summary_path is None or not summary_path.is_file():
            if folder is None:
                raise HTTPException(404, "没有这局")
            await _short_path(folder)
            progress = await asyncio.to_thread(observed_summary, folder)
            return {"id": game_id, "summary": progress, "live": (folder / 'events.jsonl').exists(), "external": True, **_saved_meta(folder)}
        return {"id": game_id, "summary": json.loads(summary_path.read_text(encoding="utf-8")), "live": False, **_saved_meta(folder)}
    return {
        "id": game.id,
        "live": game.task is not None and not game.task.done(),
        "finished": game.finished,
        "summary": game.summary or (await asyncio.to_thread(observed_summary, game.store.root) if game.rules.name == 'chess' else None),
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


def _coalesce_tokens(events):
    """Combine replay chunks without losing request IDs or replacement semantics."""
    pending = {}
    for event in events:
        if event.get('type') != 'token':
            yield from pending.values()
            pending.clear()
            yield event
            continue
        key = (event.get('seat'), event.get('request_id'), event.get('channel'))
        if key not in pending or event.get('replace'):
            pending[key] = dict(event)
        else:
            pending[key]['text'] += event.get('text', '')
            for field in ('sequence', 'timestamp', 'chunk_index', 'elapsed_ms'):
                if field in event: pending[key][field] = event[field]
    yield from pending.values()


async def _watch_saved(socket: WebSocket, folder: Path):
    """Tail CLI-written JSONL as it grows; never consume a partial record."""
    await socket.accept()
    await socket.send_json({'type': 'replay_begin'})

    async def publish():
        compact_path = await _short_path(folder)
        config_path = folder / 'config.json'
        windows = from_config(json.loads(config_path.read_text()) if config_path.is_file() else {})
        raw_cursor = 0
        with compact_path.open(encoding='utf-8') as compact:
            for line in compact:
                if not line.endswith('\n'):
                    break
                item = json.loads(line)
                ref = item.get('raw_ref') or {}
                raw_cursor = max(raw_cursor, ref.get('offset', 0) + ref.get('length', 0))
                if (event := _public_event(item)) is not None:
                    await socket.send_json(event)
                    if event['type'] == 'game_finished':
                        await socket.close(code=1000)
                        return
        with (folder / 'events.jsonl').open('rb') as handle:
            handle.seek(raw_cursor)
            while True:
                batch = []
                at_end = False
                for _ in range(1024):
                    offset = handle.tell()
                    line = handle.readline()
                    if not line.endswith(b'\n'):
                        handle.seek(offset)
                        at_end = True
                        break
                    if line.strip():
                        raw = json.loads(line)
                        shown = raw if raw.get('type') == 'token' else compact_event(raw, windows, {'offset': offset, 'length': len(line)})
                        event = _public_event(shown)
                        if event is not None: batch.append(event)
                for event in _coalesce_tokens(batch):
                    await socket.send_json(event)
                    if event['type'] == 'game_finished':
                        await socket.close(code=1000)
                        return
                if at_end:
                    summary = folder / 'summary.json'
                    if summary.is_file():
                        try:
                            result = json.loads(summary.read_text(encoding='utf-8'))
                        except json.JSONDecodeError:
                            pass  # summary writer may still be flushing
                        else:
                            await socket.send_json({'type': 'game_finished', 'result': result.get('result'),
                                                    'result_reason': result.get('result_reason'), 'summary': result})
                            await socket.close(code=1000)
                            return
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0)

    async def receive():
        while True:
            message = await socket.receive()
            if message['type'] == 'websocket.disconnect': return

    tasks = {asyncio.create_task(publish()), asyncio.create_task(receive())}
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done: task.result()
    except (WebSocketDisconnect, OSError, RuntimeError):
        pass
    finally:
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@app.websocket("/ws/games/{game_id}")
async def watch(socket: WebSocket, game_id: str):
    game = hub.games.get(game_id)
    if game is None:
        folder = _saved_game_dir(game_id)
        if folder is not None and (folder / 'events.jsonl').is_file():
            await _watch_saved(socket, folder)
        else:
            await socket.close(code=4404)
        return
    await socket.accept()
    async with game.lock:
        await socket.send_json({"type": "replay_begin"})
        for event in _coalesce_tokens(game.events):
            await socket.send_json(event)
        for event in game.pending_tokens.values():
            await socket.send_json(event)
        game.sockets.add(socket)
    try:
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
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
            tokenizer_path=config.get('model_path'),
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
            attempts = [event for event in game.events if event['type'] == 'attempt']
            summary = (observed_summary(game.store.root) if body.backend == 'harness' else None) or summarize_attempts(game.id, attempts, 'aborted', 'stopped', rules.seats, rules.name)
            summary.update(result='aborted', result_reason='stopped', provisional=False)
            game.store.write_summary(summary)
            done = {"type": "game_finished", "result": "aborted", "result_reason": "stopped", "summary": summary}
            game.store.append(done)
            await game.emit(done)
        raise
    finally:
        if engine_seat is not None:
            engine_seat.close()
        game._finish_video()
