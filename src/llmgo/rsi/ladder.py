"""Versioned opponent presets and independent candidate/level promotion rules."""
from __future__ import annotations

import random
from statistics import mean
from .contracts import digest

LADDER_VERSION = "chess-skill-v1"
PRESETS = [
    (-9, .05, 800), (-5, .10, 1100), (-1, .15, 1400),
    (3, .15, None), (7, .15, None), (11, .15, None), (15, .15, None), (20, .15, None),
]


def opponent(level=1, *, nodes_base=None):
    if type(level) is not int or level < 1:
        raise ValueError("The lowest project level is L1")
    spec = {"level": level, "level_id": f"L{level}", "version": LADDER_VERSION,
            "strength_mode": "skill", "threads": 1, "hash_mb": 16, "variant": "chess"}
    if level <= 8:
        skill, seconds, anchor = PRESETS[level - 1]
        spec.update(skill=skill, depth=5, movetime_sec=seconds, nodes=None,
                    search_limit_mode="depth_time", anchor_elo=anchor)
    else:
        if not isinstance(nodes_base, int) or nodes_base < 1:
            raise ValueError("L9+ needs a node baseline measured from completed L8 games")
        spec.update(skill=20, depth=None, movetime_sec=None, nodes=nodes_base * 2 ** (level - 9),
                    search_limit_mode="nodes", anchor_elo=None)
    return spec


def batch_stats(metrics: dict) -> dict:
    games = metrics.get("games", [])
    if len(games) != 20:
        raise ValueError("A formal batch must contain exactly 20 games")
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
        if game.get("result_reason") in ("engine_error", "cancelled", "aborted"):
            raise ValueError("Infrastructure failures cannot be scored")
        point = .5 if result == "draw" else float(result == f"{side}_win")
        wins += point == 1
        draws += point == .5
        losses += point == 0
        by_side[side].append(point)
        key = game.get("pair_id")
        if type(key) is not int or not 0 <= key < 10 or (key, side) in pairs:
            raise ValueError("Formal batches require ten unique white/black seed pairs")
        pairs[key, side] = point
    if any(len(points) != 10 for points in by_side.values()):
        raise ValueError("Expected 10 white and 10 black games")
    pair_scores = [(pairs[i, "white"] + pairs[i, "black"]) / 2 for i in range(10)]
    return {"wins": wins, "draws": draws, "losses": losses, "score": mean(pair_scores),
            "white_score": mean(by_side["white"]), "black_score": mean(by_side["black"]),
            "pair_scores": pair_scores,
            "truncated": sum(g.get("result_reason") == "truncated" for g in games)}


def mastered(metrics):
    s = batch_stats(metrics)
    return s["wins"] >= 16 and s["score"] >= .85 and s["white_score"] >= .60 and s["black_score"] >= .60


def compare(candidate_batches, incumbent_batches, *, min_improvement=.05, confidence=.95):
    if len(candidate_batches) != len(incumbent_batches) or not candidate_batches:
        raise ValueError("Matched baseline and candidate batches are required")
    differences = []
    for candidate, baseline in zip(candidate_batches, incumbent_batches):
        for key in ("comparison_id", "opponent_identity", "protocol_identity", "seed_schedule"):
            if not candidate.get(key) or candidate[key] != baseline.get(key):
                raise ValueError(f"Cannot compare incompatible {key}")
        c, b = batch_stats(candidate), batch_stats(baseline)
        differences.extend(x-y for x, y in zip(c["pair_scores"], b["pair_scores"]))
    gain = mean(differences)
    rng = random.Random(4170)
    bootstrap = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(4000))
    alpha = (1-confidence)/2
    lower, upper = bootstrap[int(alpha*4000)], bootstrap[min(3999, int((1-alpha)*4000))]
    verdict = "accept" if gain >= min_improvement - 1e-12 and lower > 0 else "reject" if gain <= 0 else "inconclusive"
    return {"verdict": verdict, "gain": gain, "interval": [lower, upper], "confidence": confidence,
            "method": "paired_game_cluster_percentile_bootstrap", "pairs": len(differences),
            "min_improvement": min_improvement,
            "note": "Small adaptive development samples are not a calibrated Elo estimate."}
