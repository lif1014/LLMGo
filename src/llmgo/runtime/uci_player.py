"""UCI 引擎座位。支援完整棋局歷史，不看模型提示，座位各有獨立行程。"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import AsyncIterator

import chess
import chess.engine

from llmgo.runtime.player import StreamEvent
from llmgo.runtime.prompt import Observation


class UciPlayer:
    def __init__(
        self,
        seat: str,
        engine_path: str,
        *,
        skill: int,
        depth: int | None,
        movetime: float | None,
        nodes: int | None = None,
        threads: int = 1,
        hash_mb: int = 16,
        name: str | None = None,
    ):
        self.seat = seat
        self.name = name or f"uci-{seat}"
        self.engine_path = engine_path
        self.skill = skill
        self.depth = depth
        self.movetime = movetime
        self.nodes = nodes
        if nodes is not None and (nodes < 1 or depth is not None or movetime is not None):
            raise ValueError("Node limits require positive nodes and no depth/time cap")
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: chess.engine.SimpleEngine | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            return self._engine
        engine = chess.engine.SimpleEngine.popen_uci(
            self.engine_path,
            cwd=str(Path(self.engine_path).resolve().parent),
            timeout=120.0,
        )
        option = engine.options.get("Skill Level")
        if option is None or option.min is None or option.max is None:
            engine.close()
            raise RuntimeError(f"{self.engine_path} 沒有 Skill Level 選項")
        if not option.min <= self.skill <= option.max:
            engine.close()
            raise RuntimeError(
                f"Skill Level {self.skill} 超出引擎範圍 {option.min}..{option.max}"
            )
        config = {"Skill Level": self.skill, "Threads": self.threads, "Hash": self.hash_mb}
        if "UCI_LimitStrength" in engine.options:
            config["UCI_LimitStrength"] = False
        variant = engine.options.get("UCI_Variant")
        if variant is not None:
            if "chess" not in (variant.var or []):
                engine.close()
                raise RuntimeError("Engine does not support standard chess")
            # python-chess manages UCI_Variant from the Board type on play().
            # Configuring this managed option explicitly raises EngineError.
        eval_file = engine.options.get("EvalFile")
        if eval_file is not None and eval_file.default:
            candidate = Path(self.engine_path).resolve().parent / str(eval_file.default)
            if candidate.exists():
                config["EvalFile"] = str(candidate)
        try:
            engine.configure(config)
        except BaseException:
            engine.close()
            raise
        self._engine = engine
        return engine

    def _play_board(self, board: chess.Board):
        with self._lock:
            engine = self._ensure()
            played = engine.play(board, chess.engine.Limit(time=self.movetime, depth=self.depth, nodes=self.nodes),
                                 info=chess.engine.INFO_BASIC)
            if played.move is None:
                raise RuntimeError(f"{self.name} 沒有走出著法")
            uci = played.move.uci()
            if played.move not in board.legal_moves:
                raise RuntimeError(f"{self.name} 走出非法著法 {uci}")
            return uci, {"search": {key: played.info.get(key) for key in ("depth", "seldepth", "nodes", "time", "nps")},
                         "limits": {"skill": self.skill, "depth": self.depth, "movetime_sec": self.movetime, "nodes": self.nodes},
                         "engine": engine.id}

    def _best_uci(self, fen: str) -> str:
        return self._play_board(chess.Board(fen))[0]

    async def play_board(self, board: chess.Board):
        return await asyncio.to_thread(self._play_board, board.copy(stack=True))

    async def play(self, obs: Observation) -> AsyncIterator[StreamEvent]:
        if obs.seat != self.seat:
            raise RuntimeError(f"{self.name} 收到了 {obs.seat} 的局面")
        uci = await asyncio.to_thread(self._best_uci, obs.fen)
        yield StreamEvent("content", f"<move>{uci}</move>")
        yield StreamEvent("done", finish_reason="stop")

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            engine.close()
