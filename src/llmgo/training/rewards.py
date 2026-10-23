"""Frozen Lc0 TD(0) targets, available only to offline training Workers."""
from __future__ import annotations

import os
import time
from pathlib import Path
import chess
import chess.engine

from llmgo.harness.choices import parse_move
from llmgo.rsi.contracts import BT4_SHA256, file_hash, finite
from .rollout import board_from_record


def validate_reward(config):
    if not isinstance(config, dict):
        raise ValueError("reward configuration must be an object")
    if set(config) - {"mode", "nodes", "backend"}:
        raise ValueError("Unknown reward fields; migrate old quality/sign/epsilon configurations explicitly")
    if config.get("mode", "lc0_bootstrap") != "lc0_bootstrap":
        raise ValueError("PPO requires lc0_bootstrap, not a second value-difference reward")
    nodes = config.get("nodes", 10000)
    if type(nodes) is not int or not 1 <= nodes <= 10000000:
        raise ValueError("Reward search nodes must be a positive integer")
    backend = config.get("backend", "cuda-auto")
    if not isinstance(backend, str) or not backend or len(backend) > 100 or any(c.isspace() for c in backend):
        raise ValueError("Invalid Lc0 backend")
    return {"mode": "lc0_bootstrap", "nodes": nodes, "backend": backend}


def terminal_value(board, actor):
    """Rules expectation E in [0,1]; the PPO value is Q=2E-1."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return .5 if outcome.winner is None else float(outcome.winner == actor)


class Lc0Reward:
    """Lazy offline engine; the caller owns its lifecycle and GPU placement."""
    def __init__(self, site, config):
        if os.environ.get("LLMGO_RSI_WORKER") != "1":
            raise RuntimeError("Lc0 reward is only enabled inside offline compute workers")
        self.site, self.config, self.engine = site, validate_reward(config), None
        if file_hash(site["reward_weights"]) != BT4_SHA256:
            raise ValueError("Reward network differs from frozen BT4-it332")
        self.identity = {"network": BT4_SHA256, "engine": file_hash(site["lc0_bin"]),
                         "config": self.config, "score": "native_wdl_Q_win_minus_loss_actor_pov",
                         "uci_options": {"UCI_ShowWDL": True, "Contempt": 0, "WDLCalibrationElo": 0}}

    def _ensure(self):
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                [self.site["lc0_bin"], f"--weights={self.site['reward_weights']}",
                 f"--backend={self.config['backend']}"], cwd=self.site["user_root"], timeout=600)
            options = {k: v for k, v in self.identity["uci_options"].items() if k in self.engine.options}
            self.engine.configure(options)
        return self.engine

    def value(self, board, actor):
        terminal = terminal_value(board, actor)
        if terminal is not None:
            return terminal, {"terminal": True}
        started = time.monotonic()
        engine = self._ensure()
        info = engine.analyse(board, chess.engine.Limit(nodes=self.config["nodes"]), info=chess.engine.INFO_ALL)
        if "wdl" not in info:
            raise RuntimeError("Lc0 did not return native WDL; centipawn conversion is not a substitute")
        # The UCI WDL is side-to-move relative. Always keep the original student POV.
        wdl = info["wdl"].pov(actor)
        return wdl.expectation(), {"wdl": list(wdl), "seconds": time.monotonic()-started,
            "engine_id": engine.id, "uci_options": {k: v for k,v in self.identity["uci_options"].items() if k in engine.options},
            **{k: info.get(k) for k in ("nodes", "depth", "time")}}

    def close(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            engine.quit()


def evaluate_action(record, text, provider, *, truncated=False, reward_config=None):
    """One student half-move; next state is immediately after that same move.

    gamma=1, lambda=0; A = r_env + bootstrap - V_before. No additional
    delta-shaped reward is added. Invalid output is a training protocol failure,
    not an actual lost chess game, and never invokes a fallback move.
    """
    validate_reward(reward_config or {})
    board = board_from_record(record)
    if len(record["moves"]) >= 200:
        raise ValueError("Position is beyond the 200-ply protocol limit")
    actor = board.turn
    before_e, before_info = provider.value(board, actor)
    finite(before_e, "Lc0 expectation", 0, 1)
    before = 2*before_e-1
    uci = None if truncated or "</think>" not in text else parse_move(text, {m.uci() for m in board.legal_moves})
    failure = "length" if truncated else "incomplete_thinking" if "</think>" not in text else "invalid_final_answer" if uci is None else None
    before_fen = board.fen()
    cap = False
    after_info = None
    if failure:
        env_reward, bootstrap, after_value, terminal = -1., 0., None, True
    else:
        board.push_uci(uci)
        outcome = terminal_value(board, actor)
        cap = outcome is None and len(record["moves"])+1 >= 200
        terminal = outcome is not None or cap
        if terminal:
            env_reward = 0. if cap else 2*outcome-1
            bootstrap, after_value = 0., env_reward
            after_info = {"terminal": True, "protocol_cap_draw": cap,
                          "result": board.result(claim_draw=True) if not cap else "1/2-1/2"}
        else:
            after_e, after_info = provider.value(board, actor)
            finite(after_e, "Lc0 expectation", 0, 1)
            env_reward, bootstrap = 0., 2*after_e-1
            after_value = bootstrap
    target = env_reward + bootstrap
    return {"uci": uci, "actor": "white" if actor else "black", "before_fen": before_fen,
            "after_fen": board.fen() if uci else None, "value_before": before,
            "value_after": after_value, "environment_reward": env_reward, "bootstrap": bootstrap,
            "bootstrap_mask": int(not terminal), "target": target, "advantage": target-before,
            "gamma": 1., "lambda": 0., "terminated": terminal, "generation_truncated": truncated,
            "protocol_cap_draw": cap, "segment_truncated": not terminal, "failure_reason": failure,
            "before_info": before_info, "after_info": after_info}
