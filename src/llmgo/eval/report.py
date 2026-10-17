"""首步合法率和错误类型。按座位分开，再汇总。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _seat_bucket() -> dict:
    return {"first_try_total": 0, "first_try_legal": 0, "error_types": {}}


def summarize_attempts(
    game_id: str,
    attempts: list[dict],
    result: str | None,
    reason: str | None,
    seats: tuple[str, ...] = ("red", "black"),
    game: str = "xiangqi",
) -> dict:
    by_seat = {seat: _seat_bucket() for seat in seats}
    errors: Counter[str] = Counter()
    cut = 0
    for item in attempts:
        seat = item["seat"]
        by_seat.setdefault(seat, _seat_bucket())
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
        "game": game,
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
    by_seat: dict[str, dict] = {}
    total = legal = games = 0
    for item in summaries:
        games += 1
        total += item.get("first_try_total") or 0
        legal += item.get("first_try_legal") or 0
        errors.update(item.get("error_types") or {})
        for seat, src in (item.get("by_seat") or {}).items():
            bucket = by_seat.setdefault(seat, _seat_bucket())
            bucket["first_try_total"] += src.get("first_try_total") or 0
            bucket["first_try_legal"] += src.get("first_try_legal") or 0
            for name, count in (src.get("error_types") or {}).items():
                kinds = bucket["error_types"]
                kinds[name] = kinds.get(name, 0) + count
    report = {
        "games": games,
        "first_try_total": total,
        "first_try_legal": legal,
        "first_try_legal_rate": (legal / total) if total else None,
        "error_types": dict(errors),
        "by_seat": by_seat,
    }
    harness_games = [item for item in summaries if item.get('harness')]
    if harness_games:
        report['harness'] = True
        for key in ('model_turns', 'model_decisions', 'forced_moves', 'random_moves', 'request_count', 'completion_tokens', 'request_errors', 'finish_length_count'):
            report[key] = sum(item.get(key, 0) or 0 for item in harness_games)
        report['random_rate'] = report['random_moves'] / report['model_turns'] if report['model_turns'] else None
        report['tokens_exact'] = all(item.get('tokens_exact', True) for item in harness_games)
        counts = Counter()
        for item in harness_games:
            counts.update(item.get('fallback_reasons') or {})
        report['fallback_reasons'] = dict(counts)
    return report


def _rules_name(folder: Path) -> str:
    config_path = folder / "config.json"
    if not config_path.exists():
        return "xiangqi"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "xiangqi"
    return ((data.get("request") or {}).get("game")) or "xiangqi"


def _config_fields(folder: Path) -> dict:
    path = folder / "config.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    fields = {}
    if data.get("model_seat"):
        fields["model_seat"] = data["model_seat"]
    if data.get("model"):
        fields["model"] = data["model"]
    return fields


def _summary_from(path: Path) -> dict | None:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not item.get("game"):
        item["game"] = _rules_name(path.parent)
    item.update({key: value for key, value in _config_fields(path.parent).items() if not item.get(key)})
    return item


def load_runs(runs_dir: Path) -> list[dict]:
    found = []
    if not runs_dir.exists():
        return found
    seen = set()
    paths = sorted(runs_dir.glob("*/summary.json")) + sorted(runs_dir.glob("harness_runs/*/summary.json"))
    for path in paths:
        item = _summary_from(path)
        if item is None:
            continue
        game_id = str(item.get("game_id") or path.parent.name)
        if game_id in seen:
            continue
        seen.add(game_id)
        found.append(item)
    for events in sorted(runs_dir.glob("harness_runs/*/events.jsonl")):
        game_id = events.parent.name
        if game_id in seen or (events.parent / "summary.json").exists():
            continue
        seen.add(game_id)
        item = {"game_id": game_id, "game": "chess", "result": None, "result_reason": "playing"}
        item.update(_config_fields(events.parent))
        found.append(item)
    return found
