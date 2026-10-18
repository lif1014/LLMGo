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


def summarize_harness_events(game_id: str, events, game: str = 'chess') -> dict:
    """Observed progress only: do not invent a result or overwrite stored metrics."""
    attempts, finished, started = [], {}, set()
    for event in events:
        kind = event.get('type')
        if kind == 'attempt':
            attempts.append(event)
        elif kind == 'model_request':
            key = event.get('request_id')
            if event.get('status') == 'started':
                started.add(key)
            elif event.get('status') == 'finished':
                finished[key] = event
    report = summarize_attempts(game_id, attempts, None, 'unfinished_snapshot', ('white', 'black'), game)
    turns = [event for event in attempts if event.get('source') in {'model', 'random', 'forced'}]
    report.update(harness=True, provisional=True, model_turns=len(turns),
                  model_decisions=sum(bool(e.get('ok')) and e.get('source') == 'model' for e in turns),
                  forced_moves=sum(e.get('source') == 'forced' for e in turns),
                  random_moves=sum(e.get('source') == 'random' for e in turns),
                  request_count=len(started | finished.keys()), completed_requests=len(finished),
                  pending_requests=len(started - finished.keys()),
                  completion_tokens=sum(e.get('completion_tokens', 0) or 0 for e in finished.values()),
                  tokens_exact=all(e.get('tokens_exact', False) for e in finished.values()),
                  token_scope='completed_requests',
                  fallback_reasons=dict(Counter(e['fallback_reason'] for e in turns if e.get('fallback_reason'))),
                  request_errors=sum(bool(e.get('error')) for e in finished.values()),
                  finish_length_count=sum(e.get('finish_reason') == 'length' for e in finished.values()))
    report['model_choices'] = report['model_turns'] - report['forced_moves']
    report['model_choice_rate'] = report['model_decisions'] / report['model_choices'] if report['model_choices'] else None
    report['random_rate'] = report['random_moves'] / report['model_turns'] if report['model_turns'] else None
    return report


def observed_summary(folder: Path) -> dict | None:
    path = folder / 'short_events.jsonl'
    if not path.is_file():
        return None
    if _rules_name(folder) != 'chess':
        return None
    def records():
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.endswith('\n'):
                    break
                if line.strip():
                    yield json.loads(line)
    return summarize_harness_events(folder.name, records())


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
        report['model_choices'] = report['model_turns'] - report['forced_moves']
        report['model_choice_rate'] = report['model_decisions'] / report['model_choices'] if report['model_choices'] else None
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
    return data.get('game') or ((data.get("request") or {}).get("game")) or "xiangqi"


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
    event_paths = sorted(set(runs_dir.glob('harness_runs/*/events.jsonl')) | set(runs_dir.glob('harness_runs/*/short_events.jsonl')))
    for events in event_paths:
        game_id = events.parent.name
        if game_id in seen or (events.parent / "summary.json").exists():
            continue
        seen.add(game_id)
        item = observed_summary(events.parent) or {"game_id": game_id, "game": "chess", "result": None, "result_reason": "playing"}
        item.update(_config_fields(events.parent))
        found.append(item)
    return found
