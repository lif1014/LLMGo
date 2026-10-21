"""Frozen 20-game evaluation against Fairy-Stockfish. No teacher/reward access."""
from __future__ import annotations

import asyncio
import importlib.util
import platform
import time
from pathlib import Path

from llmgo.harness.config import HarnessConfig
from llmgo.harness.player import VllmAsker
from llmgo.harness.skills import validate_cards
from llmgo.runner.harness_loop import HarnessRunner, HarnessSeat, EngineSeat
from llmgo.runtime.uci_player import UciPlayer
from llmgo.store.jsonl import TrajectoryStore
from .contracts import atomic_json, digest, file_hash, model_identity, read_json, validate_protocol
from .ladder import batch_stats


def engine_identity(engine_path, spec):
    player = engine_player("black", engine_path, spec)
    try:
        engine = player._ensure()
        eval_option = engine.options.get("EvalFile")
        weights = Path(engine_path).resolve().parent / str(eval_option.default) if eval_option and eval_option.default else None
        if eval_option and eval_option.default and not weights.is_file():
            raise ValueError(f"Expected NNUE file is missing: {weights}")
        return {"binary_sha256": file_hash(engine_path), "uci_id": engine.id,
                "nnue_sha256": file_hash(weights) if weights else None, "spec": spec}
    finally:
        player.close()


def engine_player(side, path, spec):
    return UciPlayer(side, path, skill=spec["skill"], depth=spec.get("depth"),
                     movetime=spec.get("movetime_sec"), nodes=spec.get("nodes"),
                     threads=spec["threads"], hash_mb=spec["hash_mb"])


def runtime_helpers(source_root):
    path = Path(source_root) / "scripts/eval_harness.py"
    spec = importlib.util.spec_from_file_location("_llmgo_eval_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def play_batch(payload, url, cards, output):
    p = payload["protocol"]
    cfg = HarnessConfig.from_dict(p["harness"])
    games = []
    for pair_id, seed in enumerate(payload["seed_schedule"]):
        for side in ("white", "black"):
            index = pair_id * 2 + (side == "black")
            game_id = f"{payload['job_id']}-{index:02d}"
            result_path = output / "game_results" / f"{index:02d}.json"
            if result_path.exists():
                games.append(read_json(result_path))
                continue
            game_root = output / "harness_runs" / game_id
            if game_root.exists():
                raise RuntimeError(f"Unfinished game retained at {game_root}; do not silently replay it")
            store = TrajectoryStore(game_root)
            store.write_config({"id": game_id, "game": "chess", "model_seat": side,
                                "protocol": p, "candidate_id": payload["candidate_id"],
                                "skills_sha256": payload["skills_sha256"], "seed": seed,
                                "comparison_id": payload["comparison_id"]})
            asker = VllmAsker(url, p["model_name"], tokenizer_path=payload["model_path"],
                             temperature=p["temperature"], top_p=p["top_p"], top_k=p["top_k"],
                             presence_penalty=p["presence_penalty"], seed=seed, timeout_sec=cfg.request_timeout_sec)
            student = HarnessSeat(side, asker, config=cfg, seed=seed, skill_cards=cards)
            other = "black" if side == "white" else "white"
            engine = engine_player(other, p["engine"], p["opponent"])
            seats = {side: student, other: EngineSeat(engine)}
            runner = HarnessRunner(seats["white"], seats["black"], store, max_plies=200)
            nodes = []

            async def emit(event):
                if event["type"] == "engine_search" and event["search"].get("nodes") is not None:
                    nodes.append(event["search"]["nodes"])

            start = time.monotonic()
            try:
                summary = await runner.run(emit)
            finally:
                engine.close()
            row = {"game_id": game_id, "pair_id": pair_id, "seed": seed, "model_seat": side,
                   "result": summary["result"], "result_reason": summary.get("result_reason"),
                   "seconds": time.monotonic()-start, "diagnostics": summary, "engine_nodes": nodes,
                   "raw_events": str(game_root / "events.jsonl"),
                   "short_events": str(game_root / "short_events.jsonl")}
            atomic_json(result_path, row)
            games.append(row)
    return games


def evaluate(payload):
    p, output = payload["protocol"], Path(payload["output"])
    validate_protocol(p)
    cards = validate_cards(read_json(payload["skills_path"]))
    identity = model_identity(payload["model_path"])
    if payload.get("expected_model_sha256") and identity["sha256"] != payload["expected_model_sha256"]:
        raise ValueError("Candidate model changed since its registered checkpoint")
    actual_engine = engine_identity(p["engine"], p["opponent"])
    if actual_engine != payload["engine_identity"]:
        raise ValueError("Opponent binary, NNUE or options changed")
    helpers = runtime_helpers(payload["source"]["root"])
    gpus = helpers.visible_gpu_ids()
    if not gpus:
        raise RuntimeError("Official evaluation must run in a GPU job")
    port = helpers._pick_ports(8000, 1)[0]
    service = helpers.VllmServer(Path(payload["model_path"]), p, output / "vllm.log", port, gpus[0])
    try:
        service.start()
        error = helpers._wait_servers([service], 1800)
        if error:
            raise RuntimeError(error)
        games = asyncio.run(play_batch(payload, service.base_url, cards, output))
    finally:
        service.stop()  # only the child service created by this evaluator
    metrics = {
        "schema_version": 1, "candidate_id": payload["candidate_id"],
        "comparison_id": payload["comparison_id"], "seed_schedule": payload["seed_schedule"],
        "opponent_identity": digest(actual_engine), "protocol_identity": digest(p),
        "source_identity": payload["source"]["sha256"], "skills_sha256": payload["skills_sha256"],
        "model_identity": identity, "engine_identity": actual_engine, "protocol": p,
        "host": platform.node(), "games": games,
    }
    metrics["statistics"] = batch_stats(metrics)
    metrics["primary_score"] = metrics["statistics"]["score"]
    atomic_json(output / "metrics.json", metrics)
    return output / "metrics.json"
