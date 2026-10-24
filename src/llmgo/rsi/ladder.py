"""Fixed opponent panel. There is no promotion ladder or mastery threshold."""
from __future__ import annotations
from statistics import mean
import math

LADDER_VERSION = "chess-panel-v2"
LEVELS = tuple(range(1, 9))
PRESETS = [(-9, .05), (-5, .10), (-1, .15), (3, .15),
           (7, .15), (11, .15), (15, .15), (20, .15)]


def opponent(level=1, **unused):
    if type(level) is not int or level not in LEVELS:
        raise ValueError("Fixed evaluation supports L1-L8 only")
    skill, seconds = PRESETS[level-1]
    return {"level": level, "level_id": f"L{level}", "version": LADDER_VERSION,
            "strength_mode": "skill", "threads": 1, "hash_mb": 16, "variant": "chess",
            "skill": skill, "depth": 5, "movetime_sec": seconds, "nodes": None,
            "search_limit_mode": "depth_time", "anchor_elo": None}


def evaluation_plan(data=None):
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError("evaluation must be an object or false")
    levels = data.get("levels", list(LEVELS))
    tracks = data.get("tracks", ["player"])
    if (not isinstance(levels, list) or not levels or len(set(levels)) != len(levels)
            or any(type(x) is not int or x not in LEVELS for x in levels)):
        raise ValueError("levels must be unique integers from L1-L8")
    if (not isinstance(tracks, list) or not tracks or len(set(tracks)) != len(tracks)
            or any(x not in ("player", "model") for x in tracks)):
        raise ValueError("tracks must contain player and/or model")
    return {"levels": sorted(levels), "tracks": tracks}


def batch_stats(metrics, *, expected_games=20):
    if type(expected_games) is not int or expected_games not in (10, 20):
        raise ValueError("Supported batches contain 10 baseline or 20 evaluation games")
    pair_count = expected_games//2
    games = metrics.get("games", [])
    if len(games) != expected_games:
        raise ValueError(f"A batch must contain exactly {expected_games} games")
    ids, pairs = set(), {}
    wins = draws = losses = 0
    by_side = {"white": [], "black": []}
    for game in games:
        gid, side, result = game["game_id"], game["model_seat"], game["result"]
        if gid in ids or side not in by_side:
            raise ValueError("Duplicate game or invalid side")
        ids.add(gid)
        if result not in ("white_win", "black_win", "draw"):
            raise ValueError("Incomplete/aborted game cannot be scored")
        if game.get("result_reason") in ("engine_error", "harness_error", "cancelled", "aborted"):
            raise ValueError("Infrastructure failures cannot be scored")
        point = .5 if result == "draw" else float(result == f"{side}_win")
        wins += point == 1
        draws += point == .5
        losses += point == 0
        by_side[side].append(point)
        key = game.get("pair_id")
        if type(key) is not int or not 0 <= key < pair_count or (key, side) in pairs:
            raise ValueError(f"Batches require {pair_count} unique white/black seed pairs")
        pairs[key, side] = point
    if any(len(points) != pair_count for points in by_side.values()):
        raise ValueError(f"Expected {pair_count} white and {pair_count} black games")
    scores = [(pairs[i, "white"]+pairs[i, "black"])/2 for i in range(pair_count)]
    m = mean(scores)
    # Descriptive uncertainty only; repeated development tests are adaptive.
    se = math.sqrt(sum((x-m)**2 for x in scores)/(pair_count-1)/pair_count)
    return {"games": expected_games, "wins": wins, "draws": draws, "losses": losses, "score": m,
            "white_score": mean(by_side["white"]), "black_score": mean(by_side["black"]),
            "pair_scores": scores, "pair_standard_error": se,
            "truncated": sum(g.get("result_reason") == "truncated" for g in games)}


def panel_summary(rows):
    result = {}
    for track in ("player", "model"):
        selected = [x for x in rows if x.get("track", "player") == track]
        latest = {x["level"]: x for x in selected}
        complete = set(latest) == set(LEVELS)
        result[track] = {"complete": complete, "levels": {str(k): v["statistics"] for k,v in sorted(latest.items())},
                         "score": mean(x["statistics"]["score"] for x in latest.values()) if complete else None,
                         "games": sum(x["statistics"].get("games", 20) for x in latest.values()),
                         "note": "Development results are adaptive, not calibrated Elo; initial baseline has fewer games."}
    return result
