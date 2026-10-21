"""Candidate-side JSONL adapter, launched as a separate process by the evaluator.

This is a process/API boundary, not an OS security sandbox for hostile code.
"""
from __future__ import annotations
import asyncio
import dataclasses
import inspect
import json
import random
import sys
from pathlib import Path


def main():
    candidate = Path(sys.argv[1]).resolve()
    # Only the child imports editable student code. Keep stdout for the protocol.
    wire = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, str(candidate / "src"))
    import chess
    from llmgo.harness.agent import choose_move, AskResult
    from llmgo.harness.config import HarnessConfig
    def send(value):
        wire.write(json.dumps(value, ensure_ascii=False)+"\n")
        wire.flush()
    def receive():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("Evaluator closed the candidate channel")
        return json.loads(line)
    class Ask:
        async def __call__(self, prompt, max_tokens):
            return await self.stream(prompt, max_tokens, None)
        async def stream(self, prompt, max_tokens, emit, **kwargs):
            send({"op": "ask", "prompt": prompt, "max_tokens": max_tokens})
            while True:
                message = receive()
                if message["op"] == "token":
                    if emit:
                        await emit(message["event"])
                elif message["op"] == "response":
                    return AskResult(**message["result"])
                else:
                    raise RuntimeError(message.get("error", "Invalid evaluator response"))
    async def emit(event):
        send({"op": "event", "event": event})
    while True:
        try:
            message = receive()
        except EOFError:
            return
        if message.get("op") == "close":
            return
        board = chess.Board()
        for item in message["history"]:
            board.push_uci(item["uci"])
        if board.fen() != message["fen"]:
            raise ValueError("Candidate board/history mismatch")
        kwargs = {"seat": message["seat"], "budget": message["budget"],
                  "rng": random.Random(message["seed"]+len(message["history"])),
                  "config": HarnessConfig.from_dict()}
        if "skill_cards" in inspect.signature(choose_move).parameters:
            # The editable harness owns skill format and trigger semantics.
            cards = candidate / "skills.json"
            if cards.is_file():
                kwargs["skill_cards"] = json.loads(cards.read_text())
        result = asyncio.run(choose_move(board, message["history"], Ask(), emit, **kwargs))
        decision = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
        send({"op": "decision", "decision": decision})


if __name__ == "__main__":
    main()
