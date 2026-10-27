"""History-preserving curriculum records and the exact student move prompt."""
from __future__ import annotations

import json
import random
from pathlib import Path
import chess

from llmgo.harness.config import HarnessConfig
from llmgo.harness.prompt import move_prompt
from llmgo.harness.skills import select_skills
from llmgo.harness.tools import prepare_candidates
from llmgo.rsi.contracts import digest


def board_from_record(record):
    board = chess.Board(record["initial_fen"])
    if not board.is_valid():
        raise ValueError("Invalid curriculum starting position")
    for uci in record["moves"]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal curriculum history move: {uci}")
        board.push(move)
    if board.outcome(claim_draw=True):
        raise ValueError("Training prompt must be nonterminal")
    return board


def record_identity(record):
    return digest({"initial_fen": record["initial_fen"], "moves": record["moves"]})


def load_curriculum(path):
    result, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split", "train") != "train":
            raise ValueError("Only training-split records can enter the RL buffer")
        if set(record) - {"id", "initial_fen", "moves", "source", "split", "history_complete"}:
            raise ValueError("Curriculum must contain positions/provenance, not teacher answers")
        board = board_from_record(record)
        if len(record["moves"]) >= 200:
            raise ValueError("Curriculum position is beyond the 200-ply protocol limit")
        key = record_identity(record)
        if key in seen or board.legal_moves.count() < 2:
            continue
        seen.add(key)
        result.append({**record, "id": key})
    if not result:
        raise ValueError("Curriculum has no nonforced, nonterminal positions")
    return result


def prompt_for_record(record, cards, config=None, *, correction=None):
    cfg = config or HarnessConfig.from_dict()
    board = board_from_record(record)
    history_board = chess.Board(record["initial_fen"])
    history = []
    for uci in record["moves"]:
        history.append({"seat": "white" if history_board.turn else "black", "uci": uci})
        history_board.push_uci(uci)
    candidates, observations = prepare_candidates(board)
    selected = select_skills(board, "move", cfg.skills_limit, candidates=candidates, cards=cards)
    prompt = move_prompt(board, history, candidates, observations, skills=selected,
                         history_plies=cfg.history_plies, correction=correction)
    return {"prompt": prompt, "messages": [{"role": "user", "content": prompt}],
            "legal_moves": [move.uci() for move in board.legal_moves],
            "position_id": record_identity(record), "skills": [c["id"] for c in selected],
            "candidates":candidates,"observations":observations}


def seed_curriculum(count=16, seed=4170):
    """A reproducible fallback distribution; explicitly not human games or teacher lines."""
    rng = random.Random(seed)
    records = []
    for i in range(count):
        board = chess.Board()
        for _ in range(rng.randrange(0, 40)):
            if board.outcome(claim_draw=True):
                break
            board.push(rng.choice(sorted(board.legal_moves, key=lambda move: move.uci())))
        if board.outcome(claim_draw=True) or board.legal_moves.count() < 2:
            continue
        records.append({"initial_fen": chess.STARTING_FEN, "moves": [m.uci() for m in board.move_stack],
                        "source": f"seeded-random-playout:{seed}:{i}", "split": "train", "history_complete": True})
    return records


def from_game_events(path):
    """Extract complete replayable positions without copying any model/teacher answer."""
    board = None
    initial = None
    for number, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        event = json.loads(line)
        if event.get("type") != "position":
            continue
        if board is None:
            initial = event["fen"]
            board = chess.Board(initial)
        elif event.get("last_move"):
            board.push_uci(event["last_move"]["uci"])
        if board.fen() != chess.Board(event["fen"]).fen():
            raise ValueError(f"History/FEN mismatch at {path}:{number}")
        if not board.outcome(claim_draw=True) and board.legal_moves.count() > 1:
            yield {"initial_fen": initial, "moves": [m.uci() for m in board.move_stack],
                   "source": f"{Path(path).resolve()}:{number}", "split": "train",
                   "history_complete": initial == chess.STARTING_FEN}


def write_curriculum(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return len(load_curriculum(path))
