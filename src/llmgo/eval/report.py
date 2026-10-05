"""首步合法率和错误类型。按座位分开，再汇总。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _seat_bucket() -> dict:
    return {"first_try_total": 0, "first_try_legal": 0, "error_types": {}}


def summarize_attempts(game_id: str, attempts: list[dict], result: str | None, reason: str | None) -> dict:
    by_seat = {"red": _seat_bucket(), "black": _seat_bucket()}
    errors: Counter[str] = Counter()
    cut = 0
    for item in attempts:
        seat = item["seat"]
        if item.get("attempt_index") == 0:
            by_seat[seat]["first_try_total"] += 1
            if item.get("ok"):
                by_seat[seat]["first_try_legal"] += 1
        if not item.get("ok") and item.get("error_type"):
            errors[item["error_type"]] += 1
            bucket = by_seat[seat]["error_types"]
            bucket[item["error_type"]] = bucket.get(item["error_type"], 0) + 1
        if item.get("finish_reason") == "length":
            cut += 1
    total = sum(bucket["first_try_total"] for bucket in by_seat.values())
    legal = sum(bucket["first_try_legal"] for bucket in by_seat.values())
    return {
        "game_id": game_id,
        "result": result,
        "result_reason": reason,
        "attempts": len(attempts),
        "first_try_total": total,
        "first_try_legal": legal,
        "first_try_legal_rate": (legal / total) if total else None,
        "error_types": dict(errors),
        "by_seat": by_seat,
        "finish_length_count": cut,
    }


def aggregate(summaries: list[dict]) -> dict:
    errors: Counter[str] = Counter()
    by_seat = {"red": _seat_bucket(), "black": _seat_bucket()}
    total = legal = games = 0
    for item in summaries:
        games += 1
        total += item.get("first_try_total") or 0
        legal += item.get("first_try_legal") or 0
        errors.update(item.get("error_types") or {})
        for seat in ("red", "black"):
            src = (item.get("by_seat") or {}).get(seat) or {}
            by_seat[seat]["first_try_total"] += src.get("first_try_total") or 0
            by_seat[seat]["first_try_legal"] += src.get("first_try_legal") or 0
            for name, count in (src.get("error_types") or {}).items():
                bucket = by_seat[seat]["error_types"]
                bucket[name] = bucket.get(name, 0) + count
    return {
        "games": games,
        "first_try_total": total,
        "first_try_legal": legal,
        "first_try_legal_rate": (legal / total) if total else None,
        "error_types": dict(errors),
        "by_seat": by_seat,
    }


def load_runs(runs_dir: Path) -> list[dict]:
    found = []
    if not runs_dir.exists():
        return found
    for path in sorted(runs_dir.glob("*/summary.json")):
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return found
