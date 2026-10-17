"""中国象棋对局入口。规则和中文提示仍在原来的文件里，这里只把它们接到共用循环。"""

from __future__ import annotations

from llmgo.env.notation import chinese_notation, parse_model_move
from llmgo.env.xiangqi import BLACK, RED, Move, Position
from llmgo.runtime.prompt import Observation, llm_input_text, render_user_prompt


class XiangqiGame:
    name = "xiangqi"
    files = 9
    ranks = 10
    seats = (RED, BLACK)

    def initial(self) -> Position:
        return Position.initial()

    def opponent(self, seat: str) -> str:
        return BLACK if seat == RED else RED

    def seat_name(self, seat: str) -> str:
        return "红方" if seat == RED else "黑方"

    def observe(
        self,
        pos: Position,
        seat: str,
        history: list[dict],
        feedback: str | None,
        attempt_index: int,
        policy: str,
    ) -> Observation:
        return Observation.from_position(pos, seat, history, feedback, attempt_index, policy)

    def user_prompt(self, obs: Observation) -> str:
        return render_user_prompt(obs)

    def llm_input(self, obs: Observation, user_prompt: str) -> str:
        return llm_input_text(obs.seat, user_prompt)

    def parse_move(self, text: str, pos: Position) -> tuple[Move | None, str | None]:
        return parse_model_move(text, pos)

    def label(self, pos: Position, move: Move) -> str | None:
        return chinese_notation(pos, move)

    def parse_fail_detail(self) -> str:
        return "没有找到 <move> 坐标或中文记谱"

    def parse_fail_feedback(self) -> str:
        return "没有找到可执行的走法。请只输出一个 <move>h2e2</move> 这样的标签。"

    def illegal_feedback(self, uci: str, detail: str | None) -> str:
        return f"走法 {uci} 不能执行：{detail}"

    def request_feedback(self, exc: BaseException) -> str:
        return f"请求失败：{exc}"


XIANGQI = XiangqiGame()
