#!/usr/bin/env python3
"""Run the held-out Qwen versus DeepSeek Flash non-thinking final test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.deepseek_final import main

if __name__ == '__main__':
    raise SystemExit(main())
