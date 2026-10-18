"""一局一个目录，事件追加写入。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from llmgo.store.compact import TokenWindows, compact_event, from_config


class TrajectoryStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = root / "events.jsonl"
        self.short_events_path = root / "short_events.jsonl"
        if not self.events_path.exists():
            self.short_events_path.touch(exist_ok=True)
        self.summary_path = root / "summary.json"
        self.config_path = root / "config.json"
        self.sequence = 0
        self.windows = TokenWindows()
        self.latest_short = None

    def write_config(self, config: dict) -> None:
        self.windows = from_config(config)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, event: dict) -> None:
        self.sequence += 1
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        event.setdefault("sequence", self.sequence)
        raw = (json.dumps(event, ensure_ascii=False) + "\n").encode('utf-8')
        with self.events_path.open("ab") as handle:
            offset = handle.tell()
            handle.write(raw)
        self.latest_short = compact_event(event, self.windows, {'offset': offset, 'length': len(raw)})
        if self.latest_short is not None:
            with self.short_events_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(self.latest_short, ensure_ascii=False) + '\n')

    def write_summary(self, summary: dict) -> None:
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        rows = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
