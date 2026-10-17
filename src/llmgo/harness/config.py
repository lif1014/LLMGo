"""One complete-move question, with a shared generation budget including thinking."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

@dataclass(frozen=True)
class HarnessConfig:
    move_tokens: int = 32768
    attempts_per_choice: int = 2
    request_timeout_sec: float = 540.0
    turn_timeout_sec: float = 1800.0
    skills_limit: int = 2
    history_plies: int = 12

    def __post_init__(self):
        for name in ('move_tokens', 'attempts_per_choice', 'skills_limit', 'history_plies'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f'{name} must be an integer')
        if self.move_tokens < 1 or self.attempts_per_choice < 1:
            raise ValueError('move_tokens and attempts_per_choice must be positive')
        for name in ('request_timeout_sec', 'turn_timeout_sec'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('timeouts must be positive and finite')
        if not 0 <= self.skills_limit <= 2 or self.history_plies < 0:
            raise ValueError('skills_limit must be 0..2; history_plies must be nonnegative')

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
