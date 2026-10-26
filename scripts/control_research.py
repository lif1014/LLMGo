#!/usr/bin/env python3
"""Operator research hold; writes no campaign state and sends no process signals."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.research_control import main

if __name__ == "__main__":
    main()
