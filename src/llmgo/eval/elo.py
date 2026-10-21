"""用已知對手錨點，把勝和負反推成絕對 Elo。不參與對局循環。"""

from __future__ import annotations

import json
import math
from pathlib import Path


def performance_rating(anchor_elo: float, score: float, games: int) -> dict:
    """Elo 期望公式的反推。得分率為 0 或 1 時不給點估計。"""
    if games <= 0:
        raise ValueError("games 必須大於 0")
    p = score / games
    out = {"anchor_elo": anchor_elo, "score": score, "games": games, "p": p, "rating": None, "se": None, "saturated": None}
    if p <= 0 or p >= 1:
        out["saturated"] = "low" if p <= 0 else "high"
        return out
    out["rating"] = anchor_elo + 400.0 * math.log10(p / (1.0 - p))
    out["se"] = (400.0 / (math.log(10) * p * (1.0 - p))) * math.sqrt(p * (1.0 - p) / games)
    return out


def game_points(result: str | None, model_seat: str) -> float | None:
    """勝 1、和 0.5、負 0。沒有下完的對局不計分。"""
    if result == "draw":
        return 0.5
    if result == f"{model_seat}_win":
        return 1.0
    if isinstance(result, str) and result.endswith("_win"):
        return 0.0
    return None


def _kind(reason: str | None, points: float | None) -> str:
    if points is None:
        return "unfinished"
    if reason == "illegal_forfeit":
        return "forfeit"
    return "played"


def summarize_games(games: list[dict]) -> dict:
    """games 每一項要有 result、result_reason、model_seat、anchor_elo。"""
    buckets: dict[float, list[dict]] = {}
    unrated = []
    unfinished = 0
    for item in games:
        points = game_points(item.get("result"), item["model_seat"])
        kind = _kind(item.get("result_reason"), points)
        if points is None:
            unfinished += 1
            continue
        row = {"points": points, "kind": kind}
        if item.get("anchor_elo") is None:
            unrated.append(row)
            continue
        buckets.setdefault(float(item["anchor_elo"]), []).append(row)

    by_anchor = []
    for anchor, rows in sorted(buckets.items()):
        all_score = sum(row["points"] for row in rows)
        played = [row for row in rows if row["kind"] == "played"]
        played_score = sum(row["points"] for row in played)
        by_anchor.append(
            {
                "anchor_elo": anchor,
                "all": performance_rating(anchor, all_score, len(rows)),
                "played": performance_rating(anchor, played_score, len(played)) if played else None,
                "forfeits": sum(1 for row in rows if row["kind"] == "forfeit"),
                "wins": sum(1 for row in rows if row["points"] == 1),
                "draws": sum(1 for row in rows if row["points"] == 0.5),
                "losses": sum(1 for row in rows if row["points"] == 0),
            }
        )
    return {"finished": sum(len(rows) for rows in buckets.values()) + len(unrated),
            "unfinished": unfinished, "by_anchor": by_anchor,
            "unrated": {"games": len(unrated), "wins": sum(r["points"] == 1 for r in unrated),
                        "draws": sum(r["points"] == .5 for r in unrated),
                        "losses": sum(r["points"] == 0 for r in unrated),
                        "score": sum(r["points"] for r in unrated)}}


def load_batch(runs_dir: Path, batch_id: str) -> list[dict]:
    found = []
    if not runs_dir.exists():
        return found
    for folder in sorted(runs_dir.glob("*/")):
        config_path = folder / "config.json"
        summary_path = folder / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if config.get("batch_id") != batch_id or config.get("eval") != "arena_ladder":
            continue
        found.append(
            {
                "game_id": folder.name,
                "result": summary.get("result"),
                "result_reason": summary.get("result_reason"),
                "model_seat": config.get("model_seat"),
                "anchor_elo": config.get("anchor_elo"),
                "first_try_legal_rate": summary.get("first_try_legal_rate"),
                "first_try_legal": summary.get("first_try_legal"),
                "first_try_total": summary.get("first_try_total"),
            }
        )
    return found
