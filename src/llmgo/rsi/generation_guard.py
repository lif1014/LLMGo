"""Bounded detection of literal cyclic text in one assistant stream."""
from __future__ import annotations

import hashlib
import math


class StreamRepetitionGuard:
    def __init__(self, *, window=8192, min_chars=2048, min_cycles=12, max_period=256, check_chars=256):
        self.window, self.min_chars, self.min_cycles = window, min_chars, min_cycles
        self.max_period, self.check_chars = max_period, check_chars
        self.reset()

    def reset(self):
        self.key = None
        self.tail = ""
        self.observed = self.unchecked = 0

    def feed(self, key, delta):
        if not isinstance(delta, str) or not delta:
            return None
        if key != self.key:
            self.reset()
            self.key = key
        self.observed += len(delta)
        self.unchecked += len(delta)
        self.tail = (self.tail + delta[-self.window:])[-self.window:]
        if len(self.tail) < self.min_chars or self.unchecked < self.check_chars:
            return None
        self.unchecked = 0
        for period in range(8, self.max_period + 1):
            cycles = max(self.min_cycles, math.ceil(self.min_chars / period))
            count = period * cycles
            if count > len(self.tail):
                continue
            unit = self.tail[-period:]
            # Numeric tables, padding and punctuation alone are not evidence.
            if len({c.casefold() for c in unit if c.isalpha()}) < 3:
                continue
            if self.tail[-count:] == unit * cycles:
                return {"period_chars": period, "cycles": cycles, "repeated_chars": count,
                        "observed_chars": self.observed, "sample": unit[:160],
                        "pattern_sha256": hashlib.sha256(unit.encode()).hexdigest()}
        return None


class RepetitiveGeneration(RuntimeError):
    def __init__(self, details):
        super().__init__("Assistant output contains a long literal repetition loop")
        self.details = details
