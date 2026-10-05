"""不调用模型的座位，用来验收棋盘、重试和网页。"""

from __future__ import annotations

from typing import AsyncIterator

from llmgo.runtime.player import StreamEvent
from llmgo.runtime.prompt import Observation


class ScriptedPlayer:
    def __init__(self, seat: str, mode: str = "legal"):
        self.seat = seat
        self.name = f"scripted-{seat}"
        self.mode = mode
        self.calls = 0

    async def play(self, obs: Observation) -> AsyncIterator[StreamEvent]:
        if obs.seat != self.seat:
            raise RuntimeError(f"{self.name} 收到了对方的局面")
        self.calls += 1
        name = "红方" if self.seat == "red" else "黑方"
        if self.mode == "illegal_once" and obs.attempt_index == 0:
            move = "a0a0"
            reasoning = f"{name}单独思考：先试一个不合法的坐标 {move}。这段文字只属于{name}。"
        else:
            move = obs.legal_moves[0]
            reasoning = f"{name}单独思考：选择 {move}。这段文字只属于{name}。"
        yield StreamEvent("reasoning", reasoning)
        yield StreamEvent("content", f"<move>{move}</move>")
        yield StreamEvent("done", finish_reason="stop")
