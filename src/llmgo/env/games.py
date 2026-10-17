"""按名字取出一套游戏。规则本身不写在这里。"""

from __future__ import annotations

from llmgo.env.chess_game import CHESS
from llmgo.env.gomoku import GOMOKU
from llmgo.env.xiangqi_game import XIANGQI

GAMES = {item.name: item for item in (XIANGQI, CHESS, GOMOKU)}


def get_game(name: str):
    try:
        return GAMES[name]
    except KeyError as exc:
        raise KeyError(name) from exc
