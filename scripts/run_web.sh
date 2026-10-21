#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m uvicorn llmgo.server.app:app --host "${LLMGO_WEB_HOST:-0.0.0.0}" --port "${LLMGO_WEB_PORT:-7860}"
