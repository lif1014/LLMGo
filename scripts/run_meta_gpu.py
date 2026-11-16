#!/usr/bin/env python3
"""Frozen GPU Meta worker entry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llmgo.rsi.meta_gpu_worker import main
if __name__=='__main__':main()
