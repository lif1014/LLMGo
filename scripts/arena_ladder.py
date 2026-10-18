"""模型和引擎下國際象棋，並在賽後用錨定 Elo 反推實力。

每個 vLLM 位址同一時間只下一局。引擎每局一個行程。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmgo.env.chess_game import CHESS  # noqa: E402
from llmgo.eval.elo import load_batch, summarize_games  # noqa: E402
from llmgo.runner.harness_loop import HarnessRunner, HarnessSeat, EngineSeat  # noqa: E402
from llmgo.harness.player import VllmAsker  # noqa: E402
from llmgo.harness.config import HarnessConfig  # noqa: E402
from llmgo.runtime.uci_player import UciPlayer  # noqa: E402
from llmgo.store.jsonl import TrajectoryStore  # noqa: E402

# Project difficulty presets; anchor ratings are comparison labels, not calibrated human ratings.
LEVELS = {
    1: {"anchor_elo": 800, "skill": -9, "movetime": 0.05, "depth": 5},
    2: {"anchor_elo": 1100, "skill": -5, "movetime": 0.10, "depth": 5},
    3: {"anchor_elo": 1400, "skill": -1, "movetime": 0.15, "depth": 5},
}


def _jobs(urls: list[str], white: int, black: int) -> list[dict]:
    seats = ["white"] * white + ["black"] * black
    return [{"url": urls[i % len(urls)], "model_seat": seat, "index": i} for i, seat in enumerate(seats)]


async def _play_one(job: dict, args, batch_id: str, level: dict) -> dict:
    game_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:4]
    store = TrajectoryStore(Path(args.runs) / game_id)
    model_seat = job["model_seat"]
    engine_seat = "black" if model_seat == "white" else "white"
    store.write_config(
        {
            "id": game_id,
            "eval": "arena_ladder",
            "framework": "harness",
            "enable_thinking": True,
            "batch_id": batch_id,
            "game": "chess",
            "opponent": "fairy-stockfish",
            "lichess_level": args.level,
            "anchor_elo": level["anchor_elo"],
            "skill": level["skill"],
            "movetime_ms": int(level["movetime"] * 1000),
            "depth": level["depth"],
            "model_seat": model_seat,
            "model": args.model,
            "model_path": getattr(args, 'model_path', None),
            "vllm_base_url": job["url"],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_plies": args.max_plies,
            "harness_config": HarnessConfig.from_dict().as_dict(),
        }
    )
    sampling = dict(
        tokenizer_path=getattr(args, 'model_path', None),
        base_url=job["url"],
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
    )
    engine = UciPlayer(
        engine_seat,
        args.engine,
        skill=level["skill"],
        depth=level["depth"],
        movetime=level["movetime"],
        name=f"fairy-sf-l{args.level}-{engine_seat}",
    )
    players = {
        model_seat: HarnessSeat(model_seat, VllmAsker(**sampling), max_tokens=args.max_tokens, config=HarnessConfig.from_dict(), seed=job['index']),
        engine_seat: EngineSeat(engine),
    }
    runner = HarnessRunner(
        players["white"],
        players["black"],
        store,
        max_plies=args.max_plies,
    )

    async def emit(event: dict) -> None:
        kind = event.get("type")
        if kind == "position" and event.get("last_move"):
            move = event["last_move"]
            print(f"{game_id} {move['seat']} {move['uci']}", flush=True)
        elif kind == "attempt" and not event.get("ok"):
            print(
                f"{game_id} reject {event.get('seat')} {event.get('error_type')} {event.get('uci') or ''}",
                flush=True,
            )

    started = time.time()
    print(f"start {game_id} model={model_seat} url={job['url']}", flush=True)
    try:
        summary = await runner.run(emit)
    finally:
        engine.close()
    elapsed = time.time() - started
    print(
        f"done {game_id} result={summary.get('result')} reason={summary.get('result_reason')} "
        f"first_try={summary.get('first_try_legal')}/{summary.get('first_try_total')} "
        f"seconds={elapsed:.0f}",
        flush=True,
    )
    return {
        "game_id": game_id,
        "result": summary.get("result"),
        "result_reason": summary.get("result_reason"),
        "model_seat": model_seat,
        "anchor_elo": level["anchor_elo"],
        "seconds": elapsed,
    }


async def _worker(url: str, jobs: list[dict], args, batch_id: str, level: dict) -> list[dict]:
    done = []
    for job in jobs:
        done.append(await _play_one(job, args, batch_id, level))
    return done


def _print_report(batch_id: str, report: dict) -> None:
    print(f"batch {batch_id} finished={report['finished']} unfinished={report['unfinished']}", flush=True)
    for item in report["by_anchor"]:
        overall = item["all"]
        rating = overall["rating"]
        rating_text = f"{rating:.0f}" if rating is not None else f"saturated-{overall['saturated']}"
        se = overall["se"]
        se_text = f"{se:.0f}" if se is not None else "-"
        print(
            f"anchor {item['anchor_elo']:.0f} W-D-L {item['wins']}-{item['draws']}-{item['losses']} "
            f"forfeits={item['forfeits']} elo={rating_text} se={se_text} p={overall['p']:.3f}",
            flush=True,
        )
        played = item["played"]
        if played and played["games"] != overall["games"]:
            pr = played["rating"]
            pr_text = f"{pr:.0f}" if pr is not None else f"saturated-{played['saturated']}"
            print(
                f"played_only games={played['games']} elo={pr_text} p={played['p']:.3f}",
                flush=True,
            )


async def _run(args) -> None:
    level = LEVELS[args.level]
    urls = args.urls
    jobs = _jobs(urls, args.white, args.black)
    if not jobs:
        raise SystemExit("沒有對局可下")
    batch_id = args.batch or (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:4])
    grouped: dict[str, list[dict]] = {url: [] for url in urls}
    for job in jobs:
        grouped[job["url"]].append(job)
    print(
        f"batch {batch_id} level={args.level} anchor={level['anchor_elo']} "
        f"games={len(jobs)} servers={len(urls)}",
        flush=True,
    )
    parts = await asyncio.gather(*(_worker(url, grouped[url], args, batch_id, level) for url in urls if grouped[url]))
    rows = [item for part in parts for item in part]
    report = summarize_games(rows)
    out = Path(args.runs) / "arena-batches"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{batch_id}.json").write_text(
        json.dumps({"batch_id": batch_id, "level": level, "games": rows, "report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_report(batch_id, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", default=[])
    parser.add_argument("--white", type=int, default=20)
    parser.add_argument("--black", type=int, default=20)
    parser.add_argument("--level", type=int, default=1, choices=sorted(LEVELS))
    parser.add_argument("--engine", default=str(ROOT / "engines" / "fairy-stockfish"))
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--batch")
    parser.add_argument("--report-batch")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument('--model-path', help='Existing model directory used to decode reasoning boundary token IDs')
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--policy", default="blind", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.report_batch:
        rows = load_batch(Path(args.runs), args.report_batch)
        _print_report(args.report_batch, summarize_games(rows))
        return
    if not args.urls:
        raise SystemExit("需要至少一個 vLLM 位址")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
