"""一局一个目录，事件追加写入。"""

from __future__ import annotations

import json
from pathlib import Path


class TrajectoryStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = root / "events.jsonl"
        self.summary_path = root / "summary.json"
        self.config_path = root / "config.json"

    def write_config(self, config: dict) -> None:
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, event: dict) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

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
