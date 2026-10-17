#!/usr/bin/env bash
# 在多台機器上啟動 Arena 推理，等服務就緒後跑一檔階梯。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 不要在 g37、g82、g83、g84 啟動。
HOSTS=(${HOSTS:-g32 g33})
for host in "${HOSTS[@]}"; do
  case "${host}" in
    g37|g82|g83|g84)
      echo "不要在 ${host} 啟動 Arena" >&2
      exit 1
      ;;
  esac
done
PY="${PY:-/home/test1267/test-6/miniconda3/bin/python3}"
WHITE="${WHITE:-20}"
BLACK="${BLACK:-20}"
LEVEL="${LEVEL:-1}"
BATCH="${BATCH:-}"

tmpdir="$(mktemp -d)"
for host in "${HOSTS[@]}"; do
  ssh -o BatchMode=yes "${host}" "bash '${ROOT}/scripts/serve_arena_fleet.sh'" > "${tmpdir}/${host}.txt" &
done
wait

urls=()
for host in "${HOSTS[@]}"; do
  while read -r line; do
    case "${line}" in
      http://*) urls+=("${line}") ;;
    esac
  done < "${tmpdir}/${host}.txt"
done

if [[ ${#urls[@]} -eq 0 ]]; then
  echo "沒有可用的 vLLM" >&2
  cat "${tmpdir}"/*.txt >&2
  exit 1
fi

echo "等待 ${#urls[@]} 個服務"
deadline=$((SECONDS + 900))
pending=("${urls[@]}")
while [[ ${#pending[@]} -gt 0 && SECONDS -lt deadline ]]; do
  next=()
  for url in "${pending[@]}"; do
    if curl -sf --max-time 5 "${url}/models" >/dev/null; then
      echo "ready ${url}"
    else
      next+=("${url}")
    fi
  done
  if [[ ${#next[@]} -eq 0 ]]; then
    pending=()
  else
    pending=("${next[@]}")
    sleep 10
  fi
done
if [[ ${#pending[@]} -gt 0 ]]; then
  echo "這些服務沒有起來: ${pending[*]}" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
args=(--urls "${urls[@]}" --white "${WHITE}" --black "${BLACK}" --level "${LEVEL}")
if [[ -n "${BATCH}" ]]; then
  args+=(--batch "${BATCH}")
fi
exec "${PY}" -u "${ROOT}/scripts/arena_ladder.py" "${args[@]}"
