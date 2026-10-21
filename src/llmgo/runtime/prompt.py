"""Public chess observation contract; no player sees opponent private reasoning."""
from dataclasses import dataclass

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
    system_text: str | None = None
    user_text: str | None = None
