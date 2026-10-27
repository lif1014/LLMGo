"""Frozen offline state values, independent of environment rewards and PPO."""
from __future__ import annotations

import os
import time
import chess
import chess.engine

from llmgo.rsi.contracts import BT4_SHA256, file_hash


def teacher_config(config=None):
    config = {} if config is None else config
    if not isinstance(config, dict) or set(config) - {"mode", "nodes", "backend"}:
        raise ValueError("Unknown LC0 value configuration")
    if config.get("mode", "lc0_value") != "lc0_value":
        raise ValueError("Use mode=lc0_value; single-step reward recipes must migrate explicitly")
    nodes, backend = config.get("nodes", 10000), config.get("backend", "cuda-auto")
    if type(nodes) is not int or not 1 <= nodes <= 10000000:
        raise ValueError("LC0 nodes must be a positive integer")
    if not isinstance(backend, str) or not backend or len(backend) > 100 or any(c.isspace() for c in backend):
        raise ValueError("Invalid LC0 backend")
    return {"mode": "lc0_value", "nodes": nodes, "backend": backend}


def terminal_value(board, actor):
    """Terminal expectation in [0,1]; nonterminal is None."""
    outcome = board.outcome(claim_draw=True)
    return None if outcome is None else .5 if outcome.winner is None else float(outcome.winner == actor)


class Lc0Value:
    """Native WDL value for the student's fixed color. Never returns teacher moves."""
    def __init__(self, site, config=None):
        if os.environ.get("LLMGO_RSI_WORKER") != "1":
            raise RuntimeError("LC0 values are available only in offline compute workers")
        self.site, self.config, self.engine = site, teacher_config(config), None
        if file_hash(site["reward_weights"]) != BT4_SHA256:
            raise ValueError("Teacher network differs from frozen BT4-it332")
        self.identity = {"network": BT4_SHA256, "engine": file_hash(site["lc0_bin"]),
            "config": self.config, "score": "native_wdl_Q_win_minus_loss_student_pov",
            "uci_options": {"UCI_ShowWDL": True, "Contempt": 0, "WDLCalibrationElo": 0}}

    def _ensure(self):
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                [self.site["lc0_bin"], f"--weights={self.site['reward_weights']}",
                 f"--backend={self.config['backend']}"], cwd=self.site["user_root"], timeout=600)
            self.engine.configure({k:v for k,v in self.identity["uci_options"].items() if k in self.engine.options})
        return self.engine

    def value(self, board, actor):
        terminal = terminal_value(board, actor)
        if terminal is not None:
            return 2*terminal-1, {"terminal": True}
        started = time.monotonic()
        engine = self._ensure()
        info = engine.analyse(board, chess.engine.Limit(nodes=self.config["nodes"]), info=chess.engine.INFO_ALL)
        if "wdl" not in info:
            raise RuntimeError("LC0 returned no native WDL; centipawn conversion is forbidden")
        wdl = info["wdl"].pov(actor)
        return 2*wdl.expectation()-1, {"wdl": list(wdl), "seconds": time.monotonic()-started,
            **{k:info.get(k) for k in ("nodes", "depth", "time")}}

    def close(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            engine.quit()
