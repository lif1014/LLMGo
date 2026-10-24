"""Frozen Fairy-Stockfish referee: 10-game baseline, 20-game candidate evaluation."""
from __future__ import annotations

import asyncio
import importlib.util
import platform
import shutil
import traceback
import time
import sys
from pathlib import Path

from llmgo.harness.config import HarnessConfig
from llmgo.harness.player import VllmAsker
from llmgo.harness.skills import validate_cards
from llmgo.runner.harness_loop import HarnessRunner, HarnessSeat, EngineSeat
from llmgo.runtime.uci_player import UciPlayer
from llmgo.store.jsonl import TrajectoryStore
from .contracts import atomic_json, digest, file_hash, model_identity, read_json, validate_protocol, utc, inside
from .ladder import batch_stats
from .candidate import CandidateSeat
from .artifacts import verify_source


INFRASTRUCTURE_REASONS = {"engine_error", "harness_error", "cancelled", "aborted"}


def infrastructure_failure(row):
    return row.get("result") == "aborted" or row.get("result_reason") in INFRASTRUCTURE_REASONS


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


def runtime_helpers(source_root=None):
    from . import service
    return service


def evaluation_key(payload):
    """Exclude scheduling and output paths; include every chess-affecting identity."""
    return digest({k:payload.get(k) for k in (
        "protocol", "seed_schedule", "comparison_id", "track", "skills_sha256",
        "expected_model_sha256", "engine_identity") } | {
        "candidate_source":payload.get("candidate_source", {}).get("sha256"),
        "reference_source":payload.get("reference_source", {}).get("sha256"),
        "source":payload.get("source", {}).get("sha256"),
        "evaluator_version":"parallel-v2"})


def checked_game(path, payload, index):
    row = read_json(path)
    if infrastructure_failure(row):
        raise ValueError(f"Infrastructure failure is not completed evaluation evidence: {path}")
    unsigned = {k:v for k,v in row.items() if k != "integrity"}
    pair = index//2
    if (row.get("integrity") != digest(unsigned) or row.get("evaluation_key") != evaluation_key(payload)
            or row.get("index") != index or row.get("pair_id") != pair
            or row.get("seed") != payload["seed_schedule"][pair]
            or row.get("model_seat") != ("black" if index%2 else "white")):
        raise ValueError(f"Game resume identity/integrity mismatch: {path}")
    # Diagnostics are evidence, not optional caches.
    for key in ("raw_events", "short_events"):
        artifact = inside(payload["site"]["user_root"], row[key])
        if not artifact.is_file() or file_hash(artifact) != row["event_hashes"][key]:
            raise ValueError(f"Completed game evidence changed: {artifact}")
    return row


async def play_batch(payload, url, cards, output, *, indices=None):
    p = payload["protocol"]
    cfg = HarnessConfig.from_dict(p["harness"])
    games = []
    for pair_id, seed in enumerate(payload["seed_schedule"]):
        for side in ("white", "black"):
            index = pair_id * 2 + (side == "black")
            if indices is not None and index not in indices:
                continue
            game_id = f"{payload['job_id']}-{index:02d}"
            result_path = output / "game_results" / f"{index:02d}.json"
            if result_path.exists():
                games.append(checked_game(result_path, payload, index))
                continue
            game_root = output / "harness_runs" / game_id
            if game_root.exists():
                # Keep every incomplete trace; each retry starts a fresh, deterministic game.
                archive = output/"incomplete_games"
                archive.mkdir(exist_ok=True)
                number = sum(1 for _ in archive.glob(game_id+".*"))
                game_root.rename(archive/f"{game_id}.{number:04d}")
            store = TrajectoryStore(game_root)
            store.write_config({"id": game_id, "game": "chess", "model_seat": side,
                                "protocol": p, "candidate_id": payload["candidate_id"],
                                "skills_sha256": payload["skills_sha256"], "seed": seed,
                                "comparison_id": payload["comparison_id"]})
            asker = VllmAsker(url, p["model_name"], tokenizer_path=payload["model_path"],
                             temperature=p["temperature"], top_p=p["top_p"], top_k=p["top_k"],
                             presence_penalty=p["presence_penalty"], seed=seed, timeout_sec=cfg.request_timeout_sec)
            if payload.get("track", "player") == "model":
                student = HarnessSeat(side, asker, config=cfg, seed=seed, skill_cards=cards)
            else:
                student = CandidateSeat(side, asker, payload["candidate_source"]["root"],
                                        game_root/"candidate.log", seed=seed,
                                        python=sys.executable)
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
                if isinstance(student, CandidateSeat):
                    await student.close()
            if infrastructure_failure(summary):
                raise RuntimeError(f"Game {game_id} aborted: {summary.get('result_reason')}; see its saved trace")
            row = {"game_id": game_id, "pair_id": pair_id, "seed": seed, "model_seat": side,
                   "result": summary["result"], "result_reason": summary.get("result_reason"),
                   "seconds": time.monotonic()-start, "diagnostics": summary, "engine_nodes": nodes,
                   "raw_events": str(game_root / "events.jsonl"),
                   "short_events": str(game_root / "short_events.jsonl")}
            row.update(index=index, evaluation_key=evaluation_key(payload), event_hashes={
                key:file_hash(row[key]) for key in ("raw_events", "short_events")})
            row["integrity"] = digest(row)
            atomic_json(result_path, row)
            games.append(row)
    return games


def evaluate(payload):
    p, output = payload["protocol"], Path(payload["output"])
    baseline = payload.get("role") == "baseline" and p.get("purpose") == "baseline"
    validate_protocol(p, baseline=baseline)
    seeds = payload["seed_schedule"]
    if (not isinstance(seeds, list) or len(seeds) != p["games"]//2
            or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Seed schedule must contain one unique integer seed per white/black pair")
    verify_source(payload["source"])
    verify_source(payload["candidate_source"])
    # Model control uses the original reference skill library, never candidate cards.
    cards = None
    if payload.get("track") == "model":
        verify_source(payload["reference_source"])
        cards = validate_cards(read_json(Path(payload["reference_source"]["root"])/"src/llmgo/harness/skills/cards.json"))
    identity = model_identity(payload["model_path"])
    if payload.get("expected_model_sha256") and identity["sha256"] != payload["expected_model_sha256"]:
        raise ValueError("Candidate model changed since its registered checkpoint")
    if p["opponent"]["level"] not in range(1,9):
        raise ValueError("Unsupported official panel")
    actual_engine = engine_identity(p["engine"], p["opponent"])
    if actual_engine != payload["engine_identity"]:
        raise ValueError("Opponent binary, NNUE or options changed")
    helpers = runtime_helpers(payload["source"]["root"])
    gpus = helpers.visible_gpu_ids()
    if not gpus:
        raise RuntimeError("Official evaluation must run in a GPU job")
    count = payload.get("resources", {}).get("gpu_count", payload["site"]["gpu_count"])
    if type(count) is not int or not 1 <= count <= 8 or len(gpus) < count:
        raise RuntimeError(f"Evaluation requires 1-8 allocated GPUs; requested={count}, visible={len(gpus)}")
    # Pin the actual initial-model hash too, before any completed game can be reused.
    payload = {**payload, "expected_model_sha256":identity["sha256"]}
    games = asyncio.run(parallel_batch(payload, cards, output, helpers, gpus[:count]))
    metrics = {
        "schema_version": 1, "candidate_id": payload["candidate_id"],
        "comparison_id": payload["comparison_id"], "seed_schedule": payload["seed_schedule"],
        "opponent_identity": digest(actual_engine), "protocol_identity": digest(p),
        "track": payload.get("track", "player"),
        "candidate_source_identity": payload["candidate_source"]["sha256"],
        "reference_source_identity": payload.get("reference_source", {}).get("sha256"),
        "source_identity": payload["source"]["sha256"], "skills_sha256": payload["skills_sha256"],
        "model_identity": identity, "engine_identity": actual_engine, "protocol": p,
        "host": platform.node(), "games": games,
    }
    metrics["statistics"] = batch_stats(metrics, expected_games=p["games"])
    metrics["primary_score"] = metrics["statistics"]["score"]
    # Do not publish if candidate execution changed the protected source/engine.
    verify_source(payload["source"])
    if engine_identity(p["engine"], p["opponent"]) != actual_engine:
        raise ValueError("Evaluation engine changed during the batch")
    verify_source(payload["candidate_source"])
    if model_identity(payload["model_path"])["sha256"] != identity["sha256"]:
        raise ValueError("Student weights changed during evaluation")
    atomic_json(output / "metrics.json", metrics)
    return output / "metrics.json"


async def parallel_batch(payload, cards, output, helpers, gpus):
    """One owned vLLM replica per GPU, resumable games, bounded per-game retries."""
    output.mkdir(parents=True, exist_ok=True)
    expected = payload["protocol"]["games"]
    if payload.get("resume_from"):
        prior = inside(payload["site"]["user_root"], payload["resume_from"])
        skipped = []
        for index in range(expected):
            source = prior/"game_results"/f"{index:02d}.json"
            dest = output/"game_results"/source.name
            if source.is_file() and not dest.exists():
                legacy = read_json(source)
                if "integrity" not in legacy and "evaluation_key" not in legacy:
                    skipped.append({"index":index,"path":str(source),"reason":"legacy_unverified_receipt"})
                    continue
                if infrastructure_failure(legacy):
                    skipped.append({"index":index,"path":str(source),"reason":"infrastructure_failure"})
                    continue
                if legacy.get("evaluation_key") != evaluation_key(payload):
                    skipped.append({"index":index,"path":str(source),"reason":"evaluation_identity_changed"})
                    continue
                atomic_json(dest, checked_game(source, payload, index))
        if skipped:
            atomic_json(output/"resume-skipped.json", {"previous_output":str(prior), "games":skipped,
                "message":"旧棋局缺少有效回执或与当前评测不匹配；保留原证据，在新任务重新评测。"})
    completed = {}
    for index in range(expected):
        path = output/"game_results"/f"{index:02d}.json"
        if path.exists():
            if infrastructure_failure(read_json(path)):
                archive = output/"invalid_game_results"
                archive.mkdir(exist_ok=True)
                path.rename(archive/f"{index:02d}-{time.time_ns()}.json")
                continue
            completed[index] = checked_game(path, payload, index)
    queue = asyncio.Queue()
    for index in range(expected):
        if index not in completed:
            queue.put_nowait(index)
    ports = helpers._pick_ports(8000, len(gpus))
    slots = [{"slot":i, "gpu":gpu, "stage":"pending", "game_index":None, "attempt":0}
             for i,gpu in enumerate(gpus)]
    failures = []
    def progress():
        atomic_json(output/"progress.json", {"stage":"evaluating", "updated":utc(),
            "message":f"并行评测：{len(completed)}/{expected}局，{len(gpus)}个GPU副本",
            "completed_games":len(completed), "total_games":expected,
            "parallelism":len(gpus), "slots":slots, "failures":failures[-8:]})
    progress()
    async def worker(slot):
        state = slots[slot]
        service = None
        try:
            while True:
                try:
                    index = queue.get_nowait()
                except asyncio.QueueEmpty:
                    state.update(stage="completed", game_index=None)
                    progress()
                    return
                state.update(game_index=index)
                for attempt in range(1, 4):
                    state.update(attempt=attempt, stage="starting" if service is None else "playing")
                    progress()
                    try:
                        if service is None:
                            service = helpers.VllmServer(Path(payload["model_path"]), payload["protocol"],
                                output/f"gpu-{slot}"/f"vllm-{index:02d}-{attempt}.log", ports[slot], gpus[slot],
                                python=payload["site"].get("inference_python"))
                            service.start()
                            error = await asyncio.to_thread(helpers._wait_servers, [service], 1800)
                            if error:
                                raise RuntimeError(error)
                        state["stage"] = "playing"
                        progress()
                        rows = await play_batch(payload, service.base_url, cards, output, indices={index})
                        completed[index] = rows[0]
                        progress()
                        break
                    except Exception as exc:
                        detail = {"slot":slot,"game_index":index,"attempt":attempt,
                                  "error":str(exc),"traceback":traceback.format_exc(),"utc":utc()}
                        atomic_json(output/"attempts"/f"game-{index:02d}-{attempt}.json", detail)
                        failures.append({k:v for k,v in detail.items() if k != "traceback"})
                        state["stage"] = "retrying" if attempt < 3 else "failed"
                        progress()
                        if service is not None:
                            await asyncio.to_thread(service.stop)
                            service = None
                        if attempt == 3:
                            raise
        finally:
            if service is not None:
                await asyncio.to_thread(service.stop)
    results = await asyncio.gather(*(worker(i) for i in range(len(gpus))), return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        raise RuntimeError(f"Parallel evaluation incomplete; {len(completed)}/{expected} games retained. {errors[0]}") from errors[0]
    if sorted(completed) != list(range(expected)):
        raise RuntimeError("Refusing to merge an incomplete evaluation")
    return [completed[i] for i in range(expected)]
