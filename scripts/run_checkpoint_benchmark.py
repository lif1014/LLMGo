#!/usr/bin/env python3
"""Operate an independent checkpoint benchmark; every artifact stays on the remote project disk."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.checkpoint_benchmark import main

if __name__=='__main__':
    main()
