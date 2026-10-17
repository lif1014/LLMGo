"""Budgets apply to generation on the local vLLM server, including thinking."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

@dataclass(frozen=True)
class HarnessConfig:
    tool_tokens: int = 4096
    piece_tokens: int = 8192
    preview_tokens: int = 4096
    move_tokens: int = 16384
    max_lookups: int = 2
    attempts_per_choice: int = 2
    request_timeout_sec: float = 540.0
    turn_timeout_sec: float = 1800.0
    skills_limit: int = 3
    history_plies: int = 12

    def __post_init__(self):
        for name in ('tool_tokens', 'piece_tokens', 'preview_tokens', 'move_tokens', 'max_lookups', 'attempts_per_choice', 'skills_limit', 'history_plies'):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be an integer')
        for name in ('tool_tokens', 'piece_tokens', 'preview_tokens', 'move_tokens', 'attempts_per_choice'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if not 0 <= self.max_lookups <= 2:
            raise ValueError('max_lookups must be between 0 and 2')
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for value in (self.request_timeout_sec, self.turn_timeout_sec)):
            raise ValueError('timeouts must be positive')
        if not 0 <= self.skills_limit <= 3 or self.history_plies < 0:
            raise ValueError('skills_limit must be 0..3; history_plies must be nonnegative')

    @classmethod
    def from_dict(cls, data=None):
        names = {f.name for f in fields(cls)}
        path = Path(__file__).with_name('settings.json')
        data = {**(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}), **(data or {})}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f'Unknown harness settings: {sorted(unknown)}')
        return cls(**data)

    def as_dict(self):
        return asdict(self)
