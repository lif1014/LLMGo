#!/usr/bin/env python3
"""Run the RSI controller in the user's 0 GPU devspace."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmgo.rsi.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
