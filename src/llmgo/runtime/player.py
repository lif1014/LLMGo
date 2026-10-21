"""Small streaming contract used by the UCI seat."""
from dataclasses import dataclass
from typing import AsyncIterator, Protocol
from .prompt import Observation

@dataclass(frozen=True)
class StreamEvent:
    kind: str
    text: str = ""
    finish_reason: str | None = None

class Player(Protocol):
    seat: str
    name: str
    def play(self, obs: Observation) -> AsyncIterator[StreamEvent]: ...
