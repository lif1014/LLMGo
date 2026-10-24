"""Candidate adapter with length-framed JSON, launched by the evaluator.

This is a process/API boundary, not an OS security sandbox for hostile code.
"""
from __future__ import annotations
import asyncio
import dataclasses
import inspect
import json
import random
import struct
import sys
from pathlib import Path


MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def encode_message(value):
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(data) <= MAX_MESSAGE_BYTES:
        raise ValueError("Candidate protocol message exceeds 64 MiB")
    return struct.pack(">I", len(data)) + data


async def read_message(reader):
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise ValueError("Incomplete candidate protocol header") from exc
    size = struct.unpack(">I", header)[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid candidate protocol message length")
    value = json.loads(await reader.readexactly(size))
    if not isinstance(value, dict):
        raise ValueError("Candidate protocol message must be an object")
    return value


def receive_message(stream):
    header = stream.read(4)
    if not header:
        raise EOFError("Evaluator closed the candidate channel")
    if len(header) != 4:
        raise ValueError("Incomplete evaluator protocol header")
    size = struct.unpack(">I", header)[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid evaluator protocol message length")
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("Incomplete evaluator protocol message")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Evaluator protocol message must be an object")
    return value


def main():
    candidate = Path(sys.argv[1]).resolve()
    # Only the child imports editable student code. Keep stdout for the protocol.
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    sys.path.insert(0, str(candidate / "src"))
    import chess
    from llmgo.harness.agent import choose_move, AskResult
    from llmgo.harness.config import HarnessConfig
    def send(value):
        wire.write(encode_message(value))
        wire.flush()
    def receive():
        return receive_message(sys.stdin.buffer)
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
