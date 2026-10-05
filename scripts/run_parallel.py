"""在多個 tp=1 的 vLLM 上同時下多局。每個地址同一時間只有一局。"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from llmgo.runner.loop import GameRunner
from llmgo.runtime.vllm_player import VllmPlayer
from llmgo.store.jsonl import TrajectoryStore

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


async def play_one(base_url: str) -> None:
    game_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:4]
    store = TrajectoryStore(RUNS / game_id)
    store.write_config(
        {
            "id": game_id,
            "backend": "vllm",
            "vllm_base_url": base_url,
            "tensor_parallel_size": 1,
            "max_plies": 300,
        }
    )
    shared = dict(
        base_url=base_url,
        model="Qwen3.5-4B",
        temperature=0.9,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
        presence_penalty=0.0,
    )
    runner = GameRunner(
        VllmPlayer("red", **shared),
        VllmPlayer("black", **shared),
        store,
        max_plies=300,
        max_retries=2,
        policy="blind",
    )

    async def emit(_event):
        return None

    print(f"start {game_id} {base_url}", flush=True)
    summary = await runner.run(emit)
    print(
        f"done {game_id} result={summary.get('result')} reason={summary.get('result_reason')} "
        f"first_try={summary.get('first_try_legal')}/{summary.get('first_try_total')}",
        flush=True,
    )


async def worker(base_url: str, games: int) -> None:
    for _ in range(games):
        await play_one(base_url)


async def main() -> None:
    urls = sys.argv[1:]
    if not urls:
        raise SystemExit("需要至少一個 vLLM 地址")
    # 連同已經在跑的那一局，總共湊滿五局：這裡再補四局，分到這些地址上。
    counts = [0] * len(urls)
    for index in range(4):
        counts[index % len(urls)] += 1
    await asyncio.gather(*(worker(url, count) for url, count in zip(urls, counts) if count))


if __name__ == "__main__":
    asyncio.run(main())
