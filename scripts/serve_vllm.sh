#!/usr/bin/env bash
# 在本机已有的 vLLM 0.20.1 镜像里启动 Qwen3.5-4B。
# g84 上的 vllm 0.11 / 0.12 镜像没有 qwen3_5，不能用。
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:v0.20.1}"
MODEL="${MODEL:-/home/test/testdata/models/Qwen3.5-4B}"
PORT="${PORT:-8000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "找不到镜像 ${IMAGE}" >&2
  exit 1
fi

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v min="${MIN_FREE_MIB}" '$2+0 >= min { print $1; exit }'
}

if [[ -z "${GPU:-}" ]]; then
  GPU="$(pick_gpu || true)"
fi
if [[ -z "${GPU}" ]]; then
  echo "现在没有空闲超过 ${MIN_FREE_MIB} MiB 的 GPU，没有启动 vLLM。" >&2
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  exit 1
fi

FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
if [[ "${FREE}" -lt "${MIN_FREE_MIB}" && "${FORCE:-0}" != "1" ]]; then
  echo "GPU ${GPU} 只剩 ${FREE} MiB。确认要挤上去再执行 FORCE=1 GPU=${GPU} $0" >&2
  exit 1
fi

echo "使用镜像 ${IMAGE} ，GPU ${GPU} ，max-model-len 32768"
exec docker run --rm --name llmgo-vllm \
  --gpus "device=${GPU}" \
  --network host \
  -v "${MODEL}:/model:ro" \
  "${IMAGE}" \
  /model \
  --served-model-name Qwen3.5-4B \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.34}"
