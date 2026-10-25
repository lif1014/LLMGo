#!/usr/bin/env python3
"""One-GPU graph/eager diagnostic using the existing research-job interface.

The first batch includes lazy initialization; this is not warmed steady-state
throughput. Serving parameters here are explicit and may differ from a verl
rollout or the campaign's frozen referee. Each reply is retained in full and its
normalized final answer is checked independently of the visible reasoning.
"""
import asyncio
from dataclasses import asdict
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = 4
GENERATION_TIMEOUT_SEC = 2400
PROTOCOL_CHECK = "normalized-final-v1"


def check_answer(answer, legal_moves):
    """Apply the Harness final-answer rules without parsing merged reasoning."""
    from llmgo.harness.choices import parse_move, split_thinking
    final, _, split_error = split_thinking(answer.text, answer.reasoning)
    normalization_error = answer.normalization_error or split_error
    move = None
    if answer.error:
        failure = "request_error"
    elif normalization_error:
        failure = normalization_error
    elif answer.finish_reason == "length":
        failure = "length_without_choice"
    else:
        move = parse_move(final, set(legal_moves))
        failure = None if move else "invalid_choice"
    # Native separated reasoning has no literal </think> in the merged display.
    has_think_tag = "</think>" in answer.output or "</think>" in answer.text
    boundary = bool(final) and not (normalization_error or answer.error) and (
        bool(answer.reasoning) or has_think_tag
        or answer.boundary_source in {"native_reasoning", "content_tag", "token_ids"})
    return {"protocol_ok": move is not None, "uci": move, "failure_reason": failure,
            "normalization_error": normalization_error, "boundary_source": answer.boundary_source,
            "boundary_error": answer.boundary_error, "has_boundary": bool(boundary),
            "has_think_tag": has_think_tag}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(port, name, process, deadline):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                if name in response.read().decode("utf-8", "replace"):
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def build_server(python, model, name, port, log, gpu, eager, capture_sizes, mode):
    argv = [python, "-I", "-m", "vllm.entrypoints.cli.main", "serve", str(model),
            "--host", "127.0.0.1", "--port", str(port), "--served-model-name", name,
            "--language-model-only", "--reasoning-parser", "qwen3",
            "--tensor-parallel-size", "1", "--max-num-seqs", "8",
            "--max-model-len", "40960", "--gpu-memory-utilization", "0.90"]
    if eager:
        argv.append("--enforce-eager")
    else:
        compilation = {"cudagraph_mode": mode}
        if capture_sizes is not None:
            compilation["cudagraph_capture_sizes"] = capture_sizes
        argv += ["--compilation-config", json.dumps(compilation)]
    environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True, env=environment)
    return process, handle, argv


def memory_notes(log_path):
    notes = []
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if any(key in line for key in ("KV cache", "gpu_memory_utilization", "Memory profiling",
                                       "CUDA graph", "cudagraph", "OutOfMemory", "Error")):
            notes.append(line.strip()[:300])
    return notes[-12:]


async def run_mode(mode, model, name, python, gpu, output, prompts, protocol,
                    capture_sizes=None, cudagraph_mode="PIECEWISE"):
    from llmgo.harness.player import VllmAsker
    from llmgo.rsi.contracts import atomic_json
    output.mkdir(parents=True, exist_ok=True)
    port = free_port()
    log_path = output/f"server-{mode}.log"
    process, handle, argv = build_server(python, model, name, port, log_path, gpu,
                                         mode == "eager", capture_sizes, cudagraph_mode)
    row = {"mode": mode, "argv": argv, "port": port, "log": str(log_path), "prompts": len(prompts),
           "protocol_check": PROTOCOL_CHECK,
           "timing_scope": "first_batch_includes_lazy_initialization",
           "cudagraph_mode": None if mode == "eager" else cudagraph_mode,
           "cudagraph_capture_sizes": None if mode == "eager" else capture_sizes}
    started = time.monotonic()
    try:
        if not wait_ready(port, name, process, time.monotonic()+900):
            row.update(status="server_failed", startup_sec=time.monotonic()-started,
                       log_tail=memory_notes(log_path)[-6:])
            return row
        row["startup_sec"] = time.monotonic()-started
        asker = VllmAsker(f"http://127.0.0.1:{port}/v1", name, tokenizer_path=str(model),
            temperature=protocol["temperature"], top_p=protocol["top_p"], top_k=protocol["top_k"],
            presence_penalty=protocol["presence_penalty"], seed=protocol["seed"],
            timeout_sec=protocol["harness"]["request_timeout_sec"])

        async def ask(index, item):
            began = time.monotonic()
            answer = await asker(item["prompt"], protocol["max_tokens"])
            seconds = time.monotonic()-began
            checked = check_answer(answer, item["legal_moves"])
            response_path = output/"responses"/f"{mode}-{index:02d}.json"
            atomic_json(response_path, {"schema_version": 2, "protocol_check": PROTOCOL_CHECK,
                "id": item["id"], "prompt": item["prompt"], "legal_moves": item["legal_moves"],
                "seconds": seconds, "answer": asdict(answer), "check": checked})
            return {"id": item["id"], "seconds": seconds,
                "tokens": answer.tokens, "finish_reason": answer.finish_reason,
                "error": answer.error, "usage": answer.usage,
                "response_path": str(response_path), **checked}

        began = time.monotonic()
        results = await asyncio.wait_for(asyncio.gather(*(ask(i, item) for i, item in enumerate(prompts))),
            timeout=GENERATION_TIMEOUT_SEC)
        row["generation_sec"] = time.monotonic()-began
        tokens = sum(int(r["tokens"] or 0) for r in results)
        row.update(status="ok", requests=results, prompt_tokens_total=sum(
            int((r["usage"] or {}).get("prompt_tokens") or 0) for r in results),
            completion_tokens_total=tokens,
            tokens_per_sec=tokens/row["generation_sec"] if row["generation_sec"] else None,
            at_token_cap=sum(1 for r in results if int(r["tokens"] or 0) >= protocol["max_tokens"]),
            protocol_ok=sum(1 for r in results if r["protocol_ok"]),
            boundary_ok=sum(1 for r in results if r["has_boundary"]))
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}",
                   elapsed_sec=time.monotonic()-started)
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        handle.close()
        row["memory_notes"] = memory_notes(log_path)
    return row


def main():
    manifest = json.loads(Path(os.environ["LLMGO_SITE"]).read_text())
    site = manifest["payload"]["site"]
    python = site.get("inference_python") or sys.executable
    if Path(sys.executable).resolve() != Path(python).resolve():
        os.execve(python, [python, str(Path(__file__).resolve())], os.environ.copy())
    output = Path(os.environ["LLMGO_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    from llmgo.rsi.contracts import atomic_json, protocol as official_protocol, read_json, utc
    from llmgo.training.rollout import load_curriculum, prompt_for_record
    model = Path(os.environ["LLMGO_MODEL"])
    candidate = Path(os.environ["LLMGO_CANDIDATE"])
    cards = read_json(candidate/"skills.json") if (candidate/"skills.json").is_file() else []
    records = load_curriculum(candidate/"curriculum.jsonl")
    step = max(len(records)//(PROMPTS+1), 1)
    chosen = [records[index] for index in range(0, len(records), step)][:PROMPTS]
    prompts = []
    for record in chosen:
        built = prompt_for_record(record, cards)
        prompts.append({"id": built["position_id"][:12], "prompt": built["prompt"],
                        "legal_moves": built["legal_moves"]})
    settings = official_protocol({"skill": -9, "depth": 5, "movetime_sec": 0.05},
                                 site["engine"], seed=4170)
    devices = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x]
    result = {"schema_version": 2, "protocol_check": PROTOCOL_CHECK,
        "utc": utc(), "model": str(model), "gpu": devices[:1] or ["0"],
        "settings": {k: settings[k] for k in ("max_tokens", "model_name", "temperature", "top_p",
                                              "top_k", "presence_penalty", "seed")},
        "positions": [{"id": p["id"], "prompt_characters": len(p["prompt"])} for p in prompts],
        "modes": []}
    atomic_json(output/"graph_ab.json", result)
    recipe = read_json(candidate/"ppo.json").get("training", {}) if (candidate/"ppo.json").is_file() else {}
    capture_sizes = recipe.get("cudagraph_capture_sizes")
    result["capture_sizes"] = capture_sizes
    for mode in ("graph", "eager"):
        row = asyncio.run(run_mode(mode, model, settings["model_name"], python,
                                   (devices[:1] or ["0"])[0], output, prompts, settings,
                                   capture_sizes, "PIECEWISE"))
        result["modes"].append(row)
        atomic_json(output/"graph_ab.json", result)
    graph, eager = result["modes"]
    if graph.get("status") == "ok" and eager.get("status") == "ok":
        result["comparison"] = {"graph_tokens_per_sec": graph["tokens_per_sec"],
            "eager_tokens_per_sec": eager["tokens_per_sec"],
            "speedup": graph["tokens_per_sec"]/eager["tokens_per_sec"] if eager["tokens_per_sec"] else None,
            "graph_protocol_ok": graph["protocol_ok"], "eager_protocol_ok": eager["protocol_ok"],
            "graph_at_token_cap": graph["at_token_cap"], "eager_at_token_cap": eager["at_token_cap"]}
    atomic_json(output/"graph_ab.json", result)
    print(json.dumps({"status": [m.get("status") for m in result["modes"]],
                      "comparison": result.get("comparison")}))


if __name__ == "__main__":
    main()
