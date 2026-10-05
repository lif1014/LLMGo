"""把公开局面写成给单一座位的提示。"""

from __future__ import annotations

from dataclasses import dataclass

from llmgo.env.notation import render_board
from llmgo.env.xiangqi import BLACK, RED, Position

SEAT_NAME = {RED: "红方", BLACK: "黑方"}

SYSTEM_PROMPT = """你正在下中国象棋，只执一方。你只能看到棋盘和双方已经落下的走法，看不到对方的思考、草稿或原始输出。不要猜测对方为什么这样走，只根据当前棋盘判断。

规则：
- 帅、将和仕、士只能在九宫内。帅、将走一格直线，仕、士走一格斜线。
- 相、象走田字，不能过河，塞住象眼就不能走。
- 马走日字，马腿被挡住就不能走。
- 车走直线，不能越过棋子。
- 炮走直线；吃子时必须隔恰好一个棋子。
- 兵、卒没过河只能向前；过河后可以向前或横走，不能后退。
- 将帅不能在同一条直线上照面。
- 走完以后，自己的帅或将不能被将军。

棋盘上（）是红方棋子，【】是黑方棋子。红方在下，黑方在上。从左到右是 a 到 i，从下到上是 0 到 9。中间写着楚河汉界。
走法写成四个字符，例如红方右炮平移到中路是 h2e2。
思考之后，只给出一个标签：
<move>h2e2</move>
"""


@dataclass(frozen=True)
class Observation:
    seat: str
    fen: str
    board_text: str
    history: tuple[dict, ...]
    feedback: str | None
    legal_moves: tuple[str, ...]
    ply: int
    attempt_index: int
    policy: str = "blind"

    @classmethod
    def from_position(
        cls,
        pos: Position,
        seat: str,
        history: list[dict],
        feedback: str | None,
        attempt_index: int,
        policy: str,
    ) -> Observation:
        legal = tuple(move.uci for move in pos.legal_moves())
        return cls(
            seat=seat,
            fen=pos.fen(),
            board_text=render_board(pos),
            history=tuple(history),
            feedback=feedback,
            legal_moves=legal,
            ply=len(history),
            attempt_index=attempt_index,
            policy=policy,
        )


def render_user_prompt(obs: Observation) -> str:
    lines = [obs.board_text, "", f"轮到：{SEAT_NAME[obs.seat]}", f"你是{SEAT_NAME[obs.seat]}。请只为这一方走一步。"]
    if obs.history:
        lines.append("已经落下的走法：")
        for index, item in enumerate(obs.history, start=1):
            lines.append(f"{index}. {SEAT_NAME[item['seat']]} {item['uci']}（{item['zh']}）")
    else:
        lines.append("还没有人走子。")
    if obs.policy == "legal_list":
        lines.append("可走的坐标：" + " ".join(obs.legal_moves))
    if obs.feedback:
        lines.append("")
        lines.append("你自己的上一手没有被接受，对方看不到这段话：")
        lines.append(obs.feedback)
        lines.append("请重新输出一个 <move> 标签。")
    else:
        lines.append("请输出一个 <move> 标签。")
    return "\n".join(lines)


def system_prompt_for(seat: str) -> str:
    return SYSTEM_PROMPT + f"\n你的座位是{SEAT_NAME[seat]}。对方的思考不会出现在这条消息里。"


def messages_for(obs: Observation) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt_for(obs.seat)},
        {"role": "user", "content": render_user_prompt(obs)},
    ]


def llm_input_text(seat: str, user_prompt: str) -> str:
    """页面上展示的完整输入，和实际发给模型的两条消息一致。"""
    return f"【system】\n{system_prompt_for(seat)}\n\n【user】\n{user_prompt}"
