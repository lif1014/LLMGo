#!/usr/bin/env bash
# 在本機每張夠空的 GPU 上啟動一個 Arena 推理服務。埠號是 8100+GPU 編號。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_FREE_MIB="${MIN_FREE_MIB:-28000}"

if [[ "${STOP:-0}" == "1" ]]; then
  docker ps -aq --filter name=llmgo-arena- | xargs -r docker rm -f
  exit 0
fi

mapfile -t ROWS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
for row in "${ROWS[@]}"; do
  gpu="${row%%,*}"
  gpu="${gpu// /}"
  free="${row##*,}"
  free="${free// /}"
  if [[ "${free}" -lt "${MIN_FREE_MIB}" ]]; then
    echo "跳過 GPU ${gpu}，空閒 ${free} MiB"
    continue
  fi
  GPU="${gpu}" PORT="$((8100 + gpu))" MIN_FREE_MIB="${MIN_FREE_MIB}" bash "${ROOT}/serve_arena_one.sh" || true
done

echo "URLS"
hostname="$(hostname)"
for row in "${ROWS[@]}"; do
  gpu="${row%%,*}"
  gpu="${gpu// /}"
  if docker ps --format '{{.Names}}' | grep -qx "llmgo-arena-${gpu}"; then
    echo "http://${hostname}:$((8100 + gpu))/v1"
  fi
done
