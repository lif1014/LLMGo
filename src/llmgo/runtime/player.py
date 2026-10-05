"""座位各自生成，不共享对话。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from llmgo.env.xiangqi import Move, Position
from llmgo.runtime.prompt import Observation, messages_for


@dataclass(frozen=True)
class StreamEvent:
    kind: str
    text: str = ""
    finish_reason: str | None = None


class Player(Protocol):
    seat: str
    name: str

    def play(self, obs: Observation) -> AsyncIterator[StreamEvent]:
        ...


def public_history_item(pos_before: Position, move: Move) -> dict:
    from llmgo.env.notation import chinese_notation

    return {
        "seat": pos_before.side,
        "uci": move.uci,
        "zh": chinese_notation(pos_before, move),
    }


def seat_messages(obs: Observation) -> list[dict[str, str]]:
    """每次只发送这一方的系统提示和当前局面，不附带任何历史对话。"""
    return messages_for(obs)
