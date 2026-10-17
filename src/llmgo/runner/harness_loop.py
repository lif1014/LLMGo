"""國際象棋 Harness 對局。選擇題事件從這裡送出，不改原來的對局循環。"""

from __future__ import annotations

import re
import random
from collections import Counter
from typing import Awaitable, Callable

import chess

from llmgo.env.chess_game import CHESS, Move, Position
from llmgo.eval.report import summarize_attempts
from llmgo.harness.agent import Ask, Decision, choose_move
from llmgo.harness.config import HarnessConfig
from llmgo.runtime.player import StreamEvent
from llmgo.runtime.prompt import Observation
from llmgo.store.jsonl import TrajectoryStore

Emit = Callable[[dict], Awaitable[None]]
_MOVE_RE = re.compile(r"<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>", re.IGNORECASE)


class HarnessSeat:
    def __init__(self, seat: str, ask: Ask, max_tokens: int = 32768, config: HarnessConfig | None = None, seed: int | None = None):
        self.seat = seat
        self.ask = ask
        self.max_tokens = max_tokens
        self.name = f"harness-{seat}"
        self.config = config or HarnessConfig.from_dict()
        self.rng = random.Random(seed)

    async def choose(self, board: chess.Board, history: list[dict], emit: Emit) -> Decision:
        return await choose_move(
            board,
            history,
            self.ask,
            emit,
            seat=self.seat,
            budget=self.max_tokens,
            config=self.config,
            rng=self.rng,
        )


class EngineSeat:
    """包一層 UCI 引擎，讓它和 Harness 座位用同一種 choose。"""

    def __init__(self, player):
        self.player = player
        self.seat = player.seat
        self.name = player.name

    async def choose(self, board: chess.Board, history: list[dict], emit: Emit) -> Decision:
        obs = Observation(
            seat=self.seat,
            fen=board.fen(),
            board_text="",
            history=tuple(history),
            feedback=None,
            legal_moves=(),
            ply=len(history),
            attempt_index=0,
        )
        chunks: list[str] = []
        async for event in self.player.play(obs):
            if isinstance(event, StreamEvent) and event.kind == "content" and event.text:
                chunks.append(event.text)
        text = "".join(chunks)
        found = _MOVE_RE.search(text)
        if found is None:
            raise RuntimeError(f"{self.name} 沒有交出著法")
        return Decision(uci=found.group(1).lower(), source="engine")

    def close(self) -> None:
        closer = getattr(self.player, "close", None)
        if closer is not None:
            closer()


class HarnessRunner:
    def __init__(
        self,
        white: HarnessSeat | EngineSeat,
        black: HarnessSeat | EngineSeat,
        store: TrajectoryStore,
        max_plies: int = 200,
    ):
        self.players = {"white": white, "black": black}
        self.store = store
        if not 1 <= max_plies <= 200:
            raise ValueError("Chess harness max_plies must be between 1 and 200")
        self.max_plies = max_plies
        self.rules = CHESS

    async def run(self, emit: Emit) -> dict:
        pos = Position(chess.Board())
        history: list[dict] = []
        attempts: list[dict] = []
        result = None
        reason = None
        tool_uses = {"position": 0, "candidates": 0}
        diagnostics = []
        random_moves = 0
        await self._emit(emit, {"type": "position", **self._position_payload(pos, None)})
        while result is None:
            if len(history) >= self.max_plies:
                result, reason = "draw", "truncated"
                break
            seat = pos.side
            player = self.players[seat]
            try:
                decision = await player.choose(pos.board, history, self._choice_emit(emit))
            except Exception as exc:
                source = "engine" if isinstance(player, EngineSeat) else "model"
                record = self._attempt(seat, len(history), False, None, source, "request_error", str(exc), "")
                attempts.append(record)
                await self._emit(emit, record)
                if isinstance(player, EngineSeat):
                    result, reason = "aborted", "engine_error"
                else:
                    winner = self.rules.opponent(seat)
                    result, reason = f"{winner}_win", "harness_error"
                break
            move = Move(decision.uci)
            verdict = pos.classify(move)
            if not verdict.ok:
                record = self._attempt(
                    seat, len(history), False, decision.uci, decision.source, verdict.error_type, verdict.error_detail, decision.prompt
                )
                attempts.append(record)
                await self._emit(emit, record)
                winner = self.rules.opponent(seat)
                result, reason = f"{winner}_win", "illegal_forfeit"
                break
            for name in decision.tools:
                tool_uses[name] = tool_uses.get(name, 0) + 1
            if decision.source == "random":
                random_moves += 1
            if decision.source != "engine":
                diagnostics.append(decision.diagnostics)
            item = {"seat": seat, "uci": move.uci, "zh": None}
            record = self._attempt(
                seat, len(history), True, move.uci, decision.source, None, None, decision.prompt, decision.tools, decision.reasoning, decision.diagnostics
            )
            attempts.append(record)
            await self._emit(emit, record)
            pos, result, reason = pos.apply(move)
            history.append(item)
            await self._emit(emit, {"type": "position", **self._position_payload(pos, item)})
        final = summarize_attempts(self.store.root.name, attempts, result, reason, self.rules.seats, self.rules.name)
        final["random_moves"] = random_moves
        final["tool_uses"] = tool_uses
        final["harness"] = True
        final["model_turns"] = len(diagnostics)
        final["forced_moves"] = sum(item.get("source") == "forced" for item in attempts)
        final["model_decisions"] = sum(item.get("source") == "model" and item.get("ok") for item in attempts)
        final["random_rate"] = random_moves / len(diagnostics) if diagnostics else None
        final["fallback_reasons"] = dict(Counter(d["fallback_reason"] for d in diagnostics if d.get("fallback_reason")))
        for key in ("request_count", "completion_tokens", "prompt_tokens", "request_errors", "length_count", "latency_ms"):
            final[key] = sum(d.get(key, 0) for d in diagnostics)
        final["tokens_exact"] = all(d.get("tokens_exact", True) for d in diagnostics)
        final["finish_length_count"] = final["length_count"]
        self.store.write_summary(final)
        done = {"type": "game_finished", "result": result, "result_reason": reason, "summary": final}
        await self._emit(emit, done)
        return final

    def _choice_emit(self, emit: Emit) -> Emit:
        async def send(event: dict) -> None:
            await self._emit(emit, event)

        return send

    async def _emit(self, emit: Emit, event: dict) -> None:
        self.store.append(event)
        await emit(event)

    def _position_payload(self, pos: Position, last: dict | None) -> dict:
        return {
            "game": self.rules.name,
            "files": self.rules.files,
            "ranks": self.rules.ranks,
            "fen": pos.fen(),
            "side": pos.side,
            "pieces": pos.pieces(),
            "last_move": last,
            "in_check": pos.in_check(pos.side),
        }

    def _attempt(self, seat, ply, ok, uci, source, error_type, error_detail, prompt, tools=None, reasoning="", diagnostics=None) -> dict:
        diagnostic = diagnostics or {}
        return {
            "type": "attempt",
            "seat": seat,
            "seat_name": self.rules.seat_name(seat),
            "ply": ply,
            "attempt_index": 0,
            "reasoning": reasoning,
            "content": f"<move>{uci}</move>" if uci else "",
            "finish_reason": (diagnostic.get("finish_reasons") or [None])[-1],
            "ok": ok,
            "uci": uci,
            "zh": None,
            "error_type": error_type,
            "error_detail": error_detail,
            "parsed_by": "harness",
            "prompt": prompt,
            "llm_input": prompt,
            "source": source,
            "tools": tools or [],
            "diagnostics": diagnostic,
            "fallback_reason": diagnostic.get("fallback_reason"),
        }
