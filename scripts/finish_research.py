#!/usr/bin/env python3
"""Explicitly hand a finished research proposal back to its waiting Controller."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.meta_protocol import ResearchHandshake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Current request.json path supplied by Controller")
    parser.add_argument("--status", choices=["ready", "needs_input"], default="ready")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--proposal", default="proposal.json")
    args = parser.parse_args()
    handshake = ResearchHandshake.from_request(args.request)
    answer = handshake.commit({"status": args.status, "summary": args.summary,
        "proposal_path": args.proposal if args.status == "ready" else None})
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == "__main__":
    main()
