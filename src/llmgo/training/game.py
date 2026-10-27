"""Complete Qwen-vs-Fairy environment, with append-only diagnostic events."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
import chess

from llmgo.rsi.contracts import atomic_json, utc
from llmgo.rsi.ladder import opponent
from llmgo.runtime.uci_player import UciPlayer


class ChessEpisode:
    def __init__(self, episode_id, policy_version, student_color, level, engine_path, output,
                 *, seed=4170, initial_fen=chess.STARTING_FEN, engine_factory=UciPlayer):
        if student_color not in ("white","black"):
            raise ValueError("Student color must be white or black")
        self.id, self.policy, self.color = episode_id, policy_version, student_color
        self.actor, self.board = student_color == "white", chess.Board(initial_fen)
        if not self.board.is_valid():
            raise ValueError("Invalid starting board")
        self.initial_fen, self.seed = initial_fen, seed
        self.rng = random.Random(seed)
        self.output, self.transitions = Path(output), []
        self.output.mkdir(parents=True,exist_ok=False)
        self.done, self.aborted, self.result, self.reason = False, False, None, None
        self.opponent = opponent(level)
        p = self.opponent
        self.engine = engine_factory("black" if self.actor else "white",engine_path,
            skill=p["skill"],depth=p["depth"],movetime=p["movetime_sec"],nodes=p["nodes"],
            threads=p["threads"],hash_mb=p["hash_mb"])
        self.event("episode_started",student_color=student_color,opponent=p,seed=seed,
                   policy_version=policy_version,initial_fen=initial_fen,
                   invalid_move_policy="random_legal",action_attribution="student_player")

    def event(self, kind, **data):
        with (self.output/"events.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({"type":kind,"episode_id":self.id,"utc":utc(),
                "ply":len(self.board.move_stack),**data},ensure_ascii=False)+"\n")

    def record(self):
        return {"initial_fen":self.initial_fen,"moves":[m.uci() for m in self.board.move_stack],
                "split":"train","source":self.id,"history_complete":self.initial_fen==chess.STARTING_FEN}

    def finish(self, score, reason):
        self.done, self.result, self.reason = True, float(score), reason
        if self.transitions:
            self.transitions[-1].update(done=True,reward=self.result,after_fen=self.board.fen(),
                next_history=self.record()["moves"],termination_reason=reason)
        self.event("episode_finished",result=self.result,reason=reason,fen=self.board.fen())
        self.write_summary()

    def boundary(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            self.finish(0 if outcome.winner is None else 1 if outcome.winner == self.actor else -1,
                        outcome.termination.name.lower())
        elif len(self.board.move_stack) >= 200:
            self.finish(0,"protocol_cap_draw")
        return self.done

    def advance(self):
        """Run opponent and forced student moves until the next genuine decision."""
        while not self.done and not self.boundary():
            if self.board.turn != self.actor:
                uci, detail = self.engine._play_board(self.board.copy(stack=True))
                self.board.push_uci(uci)
                self.event("opponent_move",uci=uci,fen=self.board.fen(),diagnostics=detail)
            elif self.board.legal_moves.count() == 1:
                move = next(iter(self.board.legal_moves))
                self.board.push(move)
                self.event("forced_move",uci=move.uci(),fen=self.board.fen())
            else:
                break
        if self.transitions and not self.done:
            self.transitions[-1].update(after_fen=self.board.fen(),next_history=self.record()["moves"])

    def act(self, uci, requests, *, fallback_reason=None):
        if self.done or self.board.turn != self.actor or not requests:
            raise ValueError("Student action outside a live decision")
        model_uci = uci
        reason = None
        if uci is None:
            legal = sorted(self.board.legal_moves,key=lambda move:move.uci())
            if not legal:
                raise ValueError("No random move from a terminal board")
            move = self.rng.choice(legal)
            uci = move.uci()
            reason = fallback_reason or requests[-1].get("failure_reason") or "invalid_final_answer"
        else:
            move = chess.Move.from_uci(uci)
            if move not in self.board.legal_moves:
                raise ValueError("Parser admitted an illegal student move")
        source = "random" if model_uci is None else "model"
        row = {"episode_id":self.id,"policy_version":self.policy,"student_color":self.color,
            "turn_index":len(self.transitions),"record":self.record(),"before_fen":self.board.fen(),
            "requests":requests,"uci":uci,"model_uci":model_uci,"action_source":source,
            "fallback_reason":reason,"action_attribution":"student_player",
            "reward":0.,"done":False,"aborted":False}
        self.transitions.append(row)
        if source == "random":
            self.event("fallback",uci=uci,source=source,reason=reason,scope="all_legal_moves",
                legal_count=len(legal),attempt_count=len(requests),
                last_failure_reason=requests[-1].get("failure_reason"),
                response_tokens=sum(r["response_tokens"] for r in requests),
                action_attribution="student_player",training_samples="original_model_responses")
        self.board.push(move)
        self.event("student_move",uci=uci,model_uci=model_uci,source=source,
                   fallback_reason=reason,fen=self.board.fen())
        self.advance()

    def abort(self, error):
        self.aborted, self.reason = True, str(error)
        self.event("episode_aborted",error=str(error))
        self.write_summary()

    def write_summary(self):
        random_moves = sum(t.get("action_source")=="random" for t in self.transitions)
        atomic_json(self.output/"summary.json",{"episode_id":self.id,"policy_version":self.policy,
            "student_color":self.color,"opponent":self.opponent,"seed":self.seed,
            "completed":self.done,"aborted":self.aborted,"result":self.result,"reason":self.reason,
            "plies":len(self.board.move_stack),"decisions":len(self.transitions),
            "random_moves":random_moves,"random_rate":random_moves/len(self.transitions) if self.transitions else 0.,
            "fallback_reasons":dict(Counter(t["fallback_reason"] for t in self.transitions if t.get("fallback_reason"))),
            "invalid_move_policy":"random_legal",
            "requests":sum(len(t["requests"]) for t in self.transitions),
            "response_tokens":sum(r["response_tokens"] for t in self.transitions for r in t["requests"]),
            "updated":utc()})

    def close(self):
        self.engine.close()
