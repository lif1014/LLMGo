#!/usr/bin/env python3
"""Private loopback bridge; normally started and stopped by the Codex launcher."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.codex_bridge import serve_main

if __name__=='__main__':
    serve_main()
