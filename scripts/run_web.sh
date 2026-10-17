#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LLMGO_CONFIG="${LLMGO_CONFIG:-${ROOT}/configs/demo.yaml}"
HOST="${LLMGO_WEB_HOST:-0.0.0.0}"
PORT="${LLMGO_WEB_PORT:-7860}"
exec /home/test1267/test-6/miniconda3/bin/python3 -m uvicorn llmgo.server.app:app --host "${HOST}" --port "${PORT}"
