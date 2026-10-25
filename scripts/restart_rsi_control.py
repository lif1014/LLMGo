#!/usr/bin/env python3
"""Checked Controller stop helper used by start_rsi.sh --restart-control."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.restart_control import main

if __name__ == "__main__":
    main()
