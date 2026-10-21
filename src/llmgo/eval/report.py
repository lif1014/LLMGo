"""首步合法率和错误类型。按座位分开，再汇总。"""

from __future__ import annotations

from collections import Counter


def _seat_bucket() -> dict:
    return {"first_try_total": 0, "first_try_legal": 0, "error_types": {}}


def summarize_attempts(
    game_id: str,
    attempts: list[dict],
    result: str | None,
    reason: str | None,
    seats: tuple[str, ...] = ("white", "black"),
    game: str = "chess",
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
