"""Move-axis GAE for complete games; no torch/verl dependency."""
from __future__ import annotations

import math

MODE = "full_game_ppo_lc0"


def game_gae(transitions, *, gamma=1., lam=.95):
    """One transition is a student decision plus the opponent's reply.

    LC0 values and rewards use the SAME student's POV throughout. The final
    transition alone terminates. Infrastructure-aborted games cannot enter here.
    No centering, protocol bonus, terminal value double counting, or token GAE.
    """
    if not transitions or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("GAE requires a complete nonempty game and gamma/lambda in [0,1]")
    identity = {(row["episode_id"], row["policy_version"], row["student_color"]) for row in transitions}
    if len(identity) != 1:
        raise ValueError("Mixed episodes, policies or value perspectives")
    for i, row in enumerate(transitions):
        if row["turn_index"] != i or bool(row["done"]) != (i == len(transitions)-1):
            raise ValueError("Broken move order or incomplete/early terminal boundary")
        v, r = row["value"], row["reward"]
        if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (v,r)) or not -1 <= v <= 1:
            raise ValueError("Nonfinite or wrong-scale LC0 value/reward")
        if row.get("aborted") or r not in (-1.,0.,1.) or (not row["done"] and r != 0):
            raise ValueError("Only terminal win/loss/draw rewards are permitted")
        if i+1 < len(transitions) and "next_history" in row:
            following = transitions[i+1]
            if (row["next_history"] != following["record"]["moves"]
                    or row["after_fen"] != following["before_fen"]):
                raise ValueError("Next state does not match the next student decision")
    advantage, result = 0., []
    for i in reversed(range(len(transitions))):
        row = transitions[i]
        following = 0. if row["done"] else transitions[i+1]["value"]
        delta = row["reward"] + gamma*following - row["value"]
        advantage = delta + gamma*lam*(not row["done"])*advantage
        result.append({**row, "next_value":following, "bootstrap_mask":int(not row["done"]),
            "delta":delta, "advantage":advantage, "return":advantage+row["value"]})
    return list(reversed(result))


def request_weights(transitions):
    """One equal-weight chess decision even when it took several model requests.

    verl averages each request over its generated tokens, then over request rows.
    Compensate for that second average to recover an average over STUDENT MOVES.
    Padding is handled separately with zero response masks.
    """
    total_requests = sum(len(row["requests"]) for row in transitions)
    if not total_requests or any(not row["requests"] for row in transitions):
        raise ValueError("Every optimized decision needs generated requests")
    weights = []
    for row in transitions:
        lengths = [int(request["response_tokens"]) for request in row["requests"]]
        if any(n < 1 for n in lengths):
            raise ValueError("Empty model output cannot enter actor loss")
        weights.append([total_requests/len(transitions)*n/sum(lengths) for n in lengths])
    return weights
