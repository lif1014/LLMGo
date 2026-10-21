"""Trusted move adapter: candidate imports never enter the referee process."""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

from llmgo.harness.agent import Decision


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
            cwd=self.root, env=env, limit=4*1024*1024)

    async def send(self, message):
        self.proc.stdin.write((json.dumps(message, ensure_ascii=False)+"\n").encode())
        await self.proc.stdin.drain()

    async def choose(self, board, history, emit):
        try:
            return await asyncio.wait_for(self._choose(board, history, emit), self.timeout)
        except BaseException:
            await self.close()
            raise

    async def _choose(self, board, history, emit):
        await self.start()
        spent, requests, prompt_tokens = 0, 0, 0
        actual_tokens, exact, errors, lengths = 0, True, 0, 0
        started, finishes = time.monotonic(), []
        await self.send({"op": "choose", "fen": board.fen(), "history": history,
                         "seat": self.seat, "seed": self.seed, "budget": self.budget})
        async def traced(event):
            await emit({**event, "seat": self.seat, "ply": len(history)})
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Candidate process exited; see candidate.log")
            message = json.loads(line)
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
            elif op == "event":
                event = message.get("event")
                if not isinstance(event, dict):
                    raise ValueError("Invalid candidate event")
                # Candidate diagnostics are retained, but cannot forge referee positions/results.
                if event.get("type") in {"position", "attempt", "game_finished", "engine_search", "summary", "token"}:
                    await traced({"type": "candidate_event", "payload": event, "origin": "candidate"})
                else:
                    await traced({**event, "origin": "candidate", "authoritative": False})
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

    async def close(self):
        if self.proc and self.proc.returncode is None:
            try:
                await self.send({"op": "close"})
                await asyncio.wait_for(self.proc.wait(), 2)
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError):
                if self.proc.returncode is None:
                    self.proc.kill()
                    await self.proc.wait()
        self.proc = None
        if self.log:
            self.log.close()
            self.log = None
