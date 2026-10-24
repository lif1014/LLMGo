"""Trusted move adapter: candidate imports never enter the referee process."""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time

from llmgo.harness.agent import Decision
from .candidate_bridge import encode_message, read_message


class CandidateSeat:
    def __init__(self, seat, asker, candidate_root, output, *, seed=0, budget=32768,
                 turn_timeout=1800, python=None):
        self.seat, self.asker = seat, asker
        self.root, self.output = Path(candidate_root), Path(output)
        self.seed, self.budget, self.timeout = seed, budget, turn_timeout
        self.python = python or sys.executable
        self.name = "candidate-"+seat
        self.proc, self.log = None, None

    async def start(self):
        if self.proc:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.output.open("a")
        env = os.environ.copy()
        # Candidate cannot accidentally use a training-only teacher or see GPU memory.
        env.pop("LLMGO_RSI_WORKER", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
        self.proc = await asyncio.create_subprocess_exec(
            self.python, "-I", str(Path(__file__).with_name("candidate_bridge.py")), str(self.root),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log,
            cwd=self.root, env=env, limit=65536)

    async def send(self, message):
        self.proc.stdin.write(encode_message(message))
        await asyncio.wait_for(self.proc.stdin.drain(), 30)

    async def choose(self, board, history, emit):
        try:
            return await asyncio.wait_for(self._choose(board, history, emit), self.timeout)
        except BaseException:
            await self.close(force=True)
            raise

    async def _choose(self, board, history, emit):
        await self.start()
        started = time.monotonic()
        await self.send({"op": "choose", "fen": board.fen(), "history": history,
                         "seat": self.seat, "seed": self.seed, "budget": self.budget})
        async def traced(event):
            await emit({**event, "seat": self.seat, "ply": len(history)})
        messages = asyncio.Queue(maxsize=64)

        async def receive():
            try:
                while True:
                    message = await read_message(self.proc.stdout)
                    if message is None:
                        raise RuntimeError("Candidate process exited; see candidate.log")
                    if message.get("op") == "event":
                        event = message.get("event")
                        if not isinstance(event, dict):
                            raise ValueError("Invalid candidate event")
                        if event.get("type") in {"position", "attempt", "game_finished", "engine_search", "summary", "token"}:
                            await traced({"type": "candidate_event", "payload": event, "origin": "candidate"})
                        else:
                            await traced({**event, "origin": "candidate", "authoritative": False})
                    else:
                        await messages.put(message)
            except Exception as exc:
                await messages.put(exc)

        # Drain child output during inference too: the child forwards stream events
        # while consuming stdin. Sequential reads can deadlock both OS pipes.
        receiver = asyncio.create_task(receive())
        try:
            return await self._dialog(messages, traced, history, started)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def _dialog(self, messages, traced, history, started):
        spent, requests, prompt_tokens = 0, 0, 0
        actual_tokens, exact, errors, lengths = 0, True, 0, 0
        finishes = []
        while True:
            message = await messages.get()
            if isinstance(message, Exception):
                raise message
            op = message.get("op")
            if op == "ask":
                n, prompt = message.get("max_tokens"), message.get("prompt")
                if type(n) is not int or n < 1 or not isinstance(prompt, str):
                    raise ValueError("Invalid candidate inference request")
                if spent >= self.budget or requests >= 64:
                    raise ValueError("External per-turn inference budget exhausted")
                requested = min(n, self.budget-spent)
                requests += 1
                rid = f"referee:{len(history)}:{requests}"
                await traced({"type": "request", "status": "started", "request_id": rid,
                              "prompt": prompt, "max_tokens": requested, "authoritative": True})
                async def token(event):
                    await traced({**event, "type": "token", "request_id": rid, "authoritative": True})
                    await self.send({"op": "token", "event": event})
                try:
                    response = await self.asker.stream(prompt, requested, token)
                except asyncio.CancelledError as exc:
                    partial = getattr(exc, "partial", None)
                    await asyncio.shield(traced({"type":"request", "status":"finished", "request_id":rid,
                        **(asdict(partial) if partial is not None else {}),
                        "error":"external_turn_cancelled", "charged_tokens":requested,
                        "authoritative":True, "cancelled":True}))
                    raise
                # Missing usage cannot grant extra inference allowance.
                charge = response.tokens if response.tokens_exact else requested
                if type(charge) is not int or charge < 0 or charge > requested:
                    raise ValueError("Invalid serving token usage")
                spent += charge
                actual_tokens += response.tokens
                exact = exact and response.tokens_exact
                errors += bool(response.error)
                lengths += response.finish_reason == "length"
                prompt_tokens += int(response.usage.get("prompt_tokens") or 0)
                finishes.append(response.finish_reason)
                await traced({"type": "request", "status": "finished", "request_id": rid,
                              **asdict(response), "charged_tokens": charge, "authoritative": True})
                await self.send({"op": "response", "result": asdict(response)})
            elif op == "decision":
                d = message.get("decision")
                if not isinstance(d, dict) or not isinstance(d.get("uci"), str):
                    raise ValueError("Candidate did not return a UCI decision")
                diagnostics = d.get("diagnostics") or {}
                diagnostics.update(request_count=requests, completion_tokens=actual_tokens, prompt_tokens=prompt_tokens,
                                   tokens_exact=exact, budget_tokens_charged=spent, request_errors=errors, length_count=lengths,
                                   finish_reasons=finishes, latency_ms=(time.monotonic()-started)*1000,
                                   external_budget=True)
                return Decision(uci=d["uci"], source=d.get("source", "model"), tools=d.get("tools", []),
                                tokens=actual_tokens, prompt=d.get("prompt", ""), reasoning=d.get("reasoning", ""),
                                diagnostics=diagnostics)
            else:
                raise ValueError("Unknown candidate protocol operation")

    async def close(self, *, force=False):
        proc, self.proc = self.proc, None
        log, self.log = self.log, None
        if proc is None:
            if log:
                log.close()
            return

        async def discard():
            while await proc.stdout.read(65536):
                pass

        # Waiting for a subprocess with a full stdout pipe can itself hang.
        drainer = asyncio.create_task(discard())
        try:
            if proc.returncode is None and not force:
                try:
                    async with asyncio.timeout(2):
                        proc.stdin.write(encode_message({"op": "close"}))
                        await proc.stdin.drain()
                        await proc.wait()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    force = True
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            async with asyncio.timeout(2):
                await proc.wait()
        finally:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.transport.abort()
            drainer.cancel()
            await asyncio.gather(drainer, return_exceptions=True)
            if log:
                log.close()
