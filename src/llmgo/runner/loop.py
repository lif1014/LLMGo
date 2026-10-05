"""对局循环。红方和黑方只通过棋盘和已落子交换信息。"""

from __future__ import annotations

from typing import Awaitable, Callable

from llmgo.env.notation import parse_model_move
from llmgo.env.xiangqi import BLACK, RED, Position
from llmgo.eval.report import summarize_attempts
from llmgo.runtime.player import Player, public_history_item
from llmgo.runtime.prompt import SEAT_NAME, Observation, llm_input_text, render_user_prompt
from llmgo.store.jsonl import TrajectoryStore

Emit = Callable[[dict], Awaitable[None]]


class GameRunner:
    def __init__(
        self,
        red: Player,
        black: Player,
        store: TrajectoryStore,
        max_plies: int = 300,
        max_retries: int = 2,
        policy: str = "blind",
    ):
        self.players = {RED: red, BLACK: black}
        self.store = store
        self.max_plies = max_plies
        self.max_retries = max_retries
        self.policy = policy

    async def run(self, emit: Emit) -> dict:
        pos = Position.initial()
        history: list[dict] = []
        attempts: list[dict] = []
        result = None
        reason = None
        await self._emit(emit, {"type": "position", **self._position_payload(pos, None)})
        while result is None:
            if len(history) >= self.max_plies:
                result, reason = "draw", "truncated"
                break
            seat = pos.side
            player = self.players[seat]
            feedback = None
            accepted = False
            for attempt_index in range(self.max_retries + 1):
                obs = Observation.from_position(pos, seat, history, feedback, attempt_index, self.policy)
                prompt = render_user_prompt(obs)
                await self._emit(
                    emit,
                    {
                        "type": "turn_started",
                        "seat": seat,
                        "ply": len(history),
                        "attempt_index": attempt_index,
                        "prompt": prompt,
                        "llm_input": llm_input_text(seat, prompt),
                    },
                )
                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                finish = None
                try:
                    async for event in player.play(obs):
                        if event.kind == "reasoning" and event.text:
                            reasoning_parts.append(event.text)
                            await self._emit(
                                emit,
                                {"type": "token", "seat": seat, "channel": "reasoning", "text": event.text},
                            )
                        elif event.kind == "content" and event.text:
                            content_parts.append(event.text)
                            await self._emit(
                                emit,
                                {"type": "token", "seat": seat, "channel": "content", "text": event.text},
                            )
                        elif event.kind == "done":
                            finish = event.finish_reason
                except Exception as exc:
                    record = self._attempt(
                        seat,
                        len(history),
                        attempt_index,
                        "".join(reasoning_parts),
                        "".join(content_parts),
                        finish,
                        False,
                        None,
                        None,
                        "request_error",
                        str(exc),
                        None,
                        prompt,
                    )
                    attempts.append(record)
                    await self._emit(emit, record)
                    feedback = f"请求失败：{exc}"
                    continue
                reasoning = "".join(reasoning_parts)
                content = "".join(content_parts)
                move, parsed_by = parse_model_move(content, pos)
                source = "content"
                if move is None:
                    move, parsed_by = parse_model_move(reasoning, pos)
                    source = "reasoning"
                if move is None:
                    record = self._attempt(
                        seat, len(history), attempt_index, reasoning, content, finish,
                        False, None, None, "parse_fail", "没有找到 <move> 坐标或中文记谱", None, prompt,
                    )
                    attempts.append(record)
                    await self._emit(emit, record)
                    feedback = "没有找到可执行的走法。请只输出一个 <move>h2e2</move> 这样的标签。"
                    continue
                verdict = pos.classify(move)
                if not verdict.ok:
                    record = self._attempt(
                        seat, len(history), attempt_index, reasoning, content, finish,
                        False, move.uci, None, verdict.error_type, verdict.error_detail, parsed_by, prompt,
                    )
                    attempts.append(record)
                    await self._emit(emit, record)
                    feedback = f"走法 {move.uci} 不能执行：{verdict.error_detail}"
                    continue
                item = public_history_item(pos, move)
                record = self._attempt(
                    seat, len(history), attempt_index, reasoning, content, finish,
                    True, move.uci, item["zh"], None, None, parsed_by if source == "content" else f"{parsed_by}_in_reasoning",
                    prompt,
                )
                attempts.append(record)
                await self._emit(emit, record)
                pos, result, reason = pos.apply(move)
                history.append(item)
                await self._emit(emit, {"type": "position", **self._position_payload(pos, item)})
                accepted = True
                break
            if not accepted and result is None:
                winner = BLACK if seat == RED else RED
                result, reason = f"{winner}_win", "illegal_forfeit"
            summary = summarize_attempts(self.store.root.name, attempts, result, reason)
            self.store.write_summary(summary)
            await self._emit(emit, {"type": "summary", "game": summary})
            if result is not None:
                break
        final = summarize_attempts(self.store.root.name, attempts, result, reason)
        self.store.write_summary(final)
        done = {"type": "game_finished", "result": result, "result_reason": reason, "summary": final}
        await self._emit(emit, done)
        return final

    async def _emit(self, emit: Emit, event: dict) -> None:
        self.store.append(event)
        await emit(event)

    @staticmethod
    def _position_payload(pos: Position, last: dict | None) -> dict:
        return {
            "fen": pos.fen(),
            "side": pos.side,
            "pieces": pos.pieces(),
            "last_move": last,
            "in_check": pos.in_check(pos.side),
        }

    @staticmethod
    def _attempt(
        seat, ply, attempt_index, reasoning, content, finish, ok, uci, zh,
        error_type, error_detail, parsed_by, prompt,
    ) -> dict:
        return {
            "type": "attempt",
            "seat": seat,
            "seat_name": SEAT_NAME[seat],
            "ply": ply,
            "attempt_index": attempt_index,
            "reasoning": reasoning,
            "content": content,
            "finish_reason": finish,
            "ok": ok,
            "uci": uci,
            "zh": zh,
            "error_type": error_type,
            "error_detail": error_detail,
            "parsed_by": parsed_by,
            "prompt": prompt,
            "llm_input": llm_input_text(seat, prompt),
        }
