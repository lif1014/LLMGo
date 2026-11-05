#!/usr/bin/env python3
"""Write derived per-game fallback counts without modifying original game records."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.random_moves import main
if __name__=='__main__':
    raise SystemExit(main())
