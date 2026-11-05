"""Complete Qwen-vs-Fairy environment, with append-only diagnostic events."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
import chess

from llmgo.env.rules import chess_outcome
from llmgo.rsi.contracts import MAX_GAME_PLIES, atomic_json, utc
from llmgo.rsi.ladder import opponent
from llmgo.runtime.uci_player import UciPlayer


class ChessEpisode:
    def __init__(self, episode_id, policy_version, student_color, level, engine_path, output,
                 *, seed=4170, initial_fen=chess.STARTING_FEN, engine_factory=UciPlayer,
                 opponent_spec=None, resume_state=None):
        if student_color not in ("white","black"):
            raise ValueError("Student color must be white or black")
        self.id, self.policy, self.color = episode_id, policy_version, student_color
        self.actor, self.board = student_color == "white", chess.Board(initial_fen)
        if not self.board.is_valid():
            raise ValueError("Invalid starting board")
        self.initial_fen, self.seed = initial_fen, seed
        self.rng = random.Random(seed)
        self.output, self.transitions = Path(output), []
        self.output.mkdir(parents=True,exist_ok=resume_state is not None)
        self.done, self.aborted, self.result, self.reason = False, False, None, None
        self.decision_offset=0
        self.prior_counts={'random_moves':0,'requests':0,'response_tokens':0}
        self.prior_fallback_reasons={}
        if resume_state is not None:
            if (resume_state['episode_id']!=episode_id or resume_state['student_color']!=student_color
                    or resume_state['seed']!=seed or resume_state['initial_fen']!=initial_fen):
                raise ValueError('Episode resume identity mismatch')
            for move in resume_state['moves']:self.board.push_uci(move)
            if self.board.fen()!=resume_state['fen']:raise ValueError('Episode resume board mismatch')
            def tuples(value):return tuple(tuples(x) for x in value) if isinstance(value,list) else value
            self.rng.setstate(tuples(resume_state['rng']))
            self.decision_offset=resume_state['decisions']
            self.prior_counts=resume_state['counts'];self.prior_fallback_reasons=resume_state['fallback_reasons']
        self.opponent = opponent(level) if opponent_spec is None else dict(opponent_spec)
        p = self.opponent
        self.engine = engine_factory("black" if self.actor else "white",engine_path,
            skill=p["skill"],depth=p["depth"],movetime=p["movetime_sec"],nodes=p["nodes"],
            threads=p["threads"],hash_mb=p["hash_mb"])
        self.event("episode_resumed" if resume_state is not None else "episode_started",student_color=student_color,opponent=p,seed=seed,
                   policy_version=policy_version,initial_fen=initial_fen,
                   fen=self.board.fen(),moves=self.record()["moves"],decision_offset=self.decision_offset,
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
        outcome = chess_outcome(self.board)
        if outcome is not None:
            self.finish(0 if outcome.winner is None else 1 if outcome.winner == self.actor else -1,
                        outcome.termination.name.lower())
        elif len(self.board.move_stack) >= MAX_GAME_PLIES:
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
            "turn_index":self.decision_offset+len(self.transitions),"record":self.record(),"before_fen":self.board.fen(),
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
        random_moves = self.prior_counts['random_moves']+sum(t.get("action_source")=="random" for t in self.transitions)
        decisions=self.decision_offset+len(self.transitions)
        atomic_json(self.output/"summary.json",{"episode_id":self.id,"policy_version":self.policy,
            "student_color":self.color,"opponent":self.opponent,"seed":self.seed,
            "completed":self.done,"aborted":self.aborted,"result":self.result,"reason":self.reason,
            "plies":len(self.board.move_stack),"decisions":decisions,
            "random_moves":random_moves,"random_rate":random_moves/decisions if decisions else 0.,
            "fallback_reasons":dict(Counter(self.prior_fallback_reasons)+Counter(t["fallback_reason"] for t in self.transitions if t.get("fallback_reason"))),
            "invalid_move_policy":"random_legal",
            "requests":self.prior_counts['requests']+sum(len(t["requests"]) for t in self.transitions),
            "response_tokens":self.prior_counts['response_tokens']+sum(r["response_tokens"] for t in self.transitions for r in t["requests"]),
            "updated":utc()})

    def snapshot(self):
        if self.done or self.aborted:raise ValueError('Only live episodes can resume')
        return {'episode_id':self.id,'policy_version':self.policy,'student_color':self.color,
            'seed':self.seed,'initial_fen':self.initial_fen,'moves':self.record()['moves'],
            'fen':self.board.fen(),'rng':self.rng.getstate(),
            'decisions':self.decision_offset+len(self.transitions),
            'counts':{'random_moves':self.prior_counts['random_moves']+sum(t.get('action_source')=='random' for t in self.transitions),
                'requests':self.prior_counts['requests']+sum(len(t['requests']) for t in self.transitions),
                'response_tokens':self.prior_counts['response_tokens']+sum(r['response_tokens'] for t in self.transitions for r in t['requests'])},
            'fallback_reasons':dict(Counter(self.prior_fallback_reasons)+Counter(t['fallback_reason'] for t in self.transitions if t.get('fallback_reason')))}

    def discard_consumed(self):
        """Keep the environment/RNG; full evidence remains in events and segment files."""
        state=self.snapshot()
        self.decision_offset=state['decisions'];self.prior_counts=state['counts']
        self.prior_fallback_reasons=state['fallback_reasons'];self.transitions=[]

    def close(self):
        self.engine.close()
