#!/usr/bin/env python3
"""Run a secret-safe connection test using the same Codex protocol as RSI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmgo.rsi.connection_check import main

if __name__ == "__main__":
    raise SystemExit(main())
