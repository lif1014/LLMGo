#!/usr/bin/env bash
# 在指定 GPU 上啟動一個 Qwen3.5-4B，供 Arena 階梯呼叫。
# 需要鏡像 vllm/vllm-openai:v0.20.1（裡面才有 qwen3_5）。
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:v0.20.1}"
MODEL="${MODEL:-/user/liyaohu4170/models/Qwen3.5-4B}"
GPU="${GPU:?需要 GPU}"
PORT="${PORT:?需要 PORT}"
NAME="${NAME:-llmgo-arena-${GPU}}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "找不到鏡像 ${IMAGE}" >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "已在跑 ${NAME}"
  exit 0
fi
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  docker rm -f "${NAME}" >/dev/null
fi

read -r FREE TOTAL < <(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "${GPU}" | awk -F', ' '{print $1, $2}')
if [[ "${FREE}" -lt "${MIN_FREE_MIB:-28000}" && "${FORCE:-0}" != "1" ]]; then
  echo "GPU ${GPU} 只剩 ${FREE} MiB，跳過" >&2
  exit 1
fi

if [[ -z "${UTIL:-}" ]]; then
  UTIL="$(awk -v free="${FREE}" -v total="${TOTAL}" 'BEGIN {
    u = (free - 8192) / total
    if (u > 0.45) u = 0.45
    if (u < 0.20) u = 0.20
    printf "%.2f", u
  }')"
fi

echo "啟動 ${NAME} GPU ${GPU} port ${PORT} free=${FREE} util=${UTIL}"
docker run -d --name "${NAME}" \
  --gpus "device=${GPU}" \
  --network host \
  -v "${MODEL}:/model:ro" \
  "${IMAGE}" \
  /model \
  --served-model-name Qwen3.5-4B \
  --host "${HOST_BIND}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --max-num-seqs 2 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gpu-memory-utilization "${UTIL}"
