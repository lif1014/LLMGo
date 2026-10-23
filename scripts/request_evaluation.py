#!/usr/bin/env python3
"""Request one difficulty through the Controller/Executor proposal interface."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.evaluation_request import main

if __name__ == "__main__":
    raise SystemExit(main())
