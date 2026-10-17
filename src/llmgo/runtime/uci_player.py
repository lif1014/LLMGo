"""UCI 引擎座位。只讀當前 FEN，不看模型提示，也不和其他座位共享行程。"""

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
        depth: int,
        movetime: float,
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
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: chess.engine.SimpleEngine | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> chess.engine.SimpleEngine:
        if self._engine is not None:
            return self._engine
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
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
        eval_file = engine.options.get("EvalFile")
        if eval_file is not None and eval_file.default:
            candidate = Path(self.engine_path).resolve().parent / str(eval_file.default)
            if candidate.exists():
                config["EvalFile"] = str(candidate)
        engine.configure(config)
        self._engine = engine
        return engine

    def _best_uci(self, fen: str) -> str:
        with self._lock:
            engine = self._ensure()
            board = chess.Board(fen)
            played = engine.play(board, chess.engine.Limit(time=self.movetime, depth=self.depth))
            if played.move is None:
                raise RuntimeError(f"{self.name} 沒有走出著法")
            uci = played.move.uci()
            if played.move not in board.legal_moves:
                raise RuntimeError(f"{self.name} 走出非法著法 {uci}")
            return uci

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
