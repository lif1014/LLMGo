"""Small, versioned contracts shared by the controller and compute workers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1
REPO = Path(__file__).resolve().parents[3]
BT4_SHA256 = "e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618"
MODEL_NAME = "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        fd_dir = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd_dir)
        finally:
            os.close(fd_dir)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("ID must contain only letters, digits, underscores or hyphens")
    return value


def inside(root: Path | str, path: Path | str) -> Path:
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Write path outside user root: {path}")
    return path


def finite(value, name: str, low=0.0, high=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside allowed range")
    return float(value)


def load_site(path: str | Path) -> dict:
    site = read_json(path)
    if site.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported site schema")
    if (site["cluster"], site["resource_pool"], site["priority"]) != ("beijing", "model-autosft", "NORMAL"):
        raise ValueError("Only beijing / model-autosft / NORMAL is authorized")
    if site["user_root"] != "/user/lyh4170":
        raise ValueError("Remote writable root must be /user/lyh4170")
    for key in ("repo", "codex_home", "campaign_root", "worktree_root", "checkpoint_root", "reward_weights", "engine", "lc0_bin", "codex_bin"):
        inside(site["user_root"], site[key])
    if type(site.get("gpu_count")) is not int or site["gpu_count"] != 1:
        raise ValueError("This Worker supports exactly one GPU per job")
    if site.get("executor_mode", "embedded") not in {"embedded", "external"}:
        raise ValueError("executor_mode must be embedded or external")
    base_poll = finite(site.get("poll_seconds", 30), "poll_seconds", 1, 3600)
    finite(site.get("executor_max_poll_seconds", max(120, base_poll)), "executor_max_poll_seconds", base_poll, 3600)
    for key in ("cpu", "memory_gib"):
        if type(site.get(key)) is not int or site[key] < 1:
            raise ValueError(f"Invalid Worker {key}")
    return site


def worker_environment(site: dict) -> dict:
    root = Path(site["user_root"])
    env = os.environ.copy()
    # Use each variable for its documented purpose; never change HOME.
    paths = {
        "TMPDIR": root / "tmp", "XDG_CACHE_HOME": root / "cache",
        "HF_HOME": root / "cache/huggingface", "TORCH_HOME": root / "cache/torch",
        "TRITON_CACHE_DIR": root / "cache/triton", "RAY_TMPDIR": root / "tmp/ray",
        "WANDB_DIR": root / "cache/wandb",
    }
    for name, path in paths.items():
        inside(root, path).mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CCTL_NO_UPDATE_CHECK"] = "1"
    return env


def protocol(opponent: dict, engine: str, seed: int = 0) -> dict:
    return {
        "game": "chess", "framework": "harness", "enable_thinking": True,
        "games": 20, "white": 10, "black": 10, "max_plies": 200, "max_tokens": 32768,
        "model_name": "Qwen3.5-4B", "temperature": 0.9, "top_p": 0.95, "top_k": 20,
        "presence_penalty": 0.0, "seed": seed, "engine": engine,
        "harness": {"move_tokens": 32768, "attempts_per_choice": 2, "request_timeout_sec": 540.0,
                    "turn_timeout_sec": 1800.0, "skills_limit": 2, "history_plies": 12},
        "opponent": opponent, "skill": opponent["skill"], "depth": opponent.get("depth"),
        "movetime_sec": opponent.get("movetime_sec"), "nodes": opponent.get("nodes"),
    }


def validate_protocol(p: dict) -> None:
    required = {"game": "chess", "framework": "harness", "enable_thinking": True,
                "games": 20, "white": 10, "black": 10, "max_plies": 200, "max_tokens": 32768}
    for key, value in required.items():
        if p.get(key) != value or type(p.get(key)) is not type(value):
            raise ValueError(f"Official RSI protocol requires {key}={value!r}")
    from llmgo.harness.config import HarnessConfig
    if HarnessConfig.from_dict(p.get("harness")).as_dict() != protocol(p["opponent"], p["engine"])["harness"]:
        raise ValueError("RSI harness settings are frozen")


def model_identity(path: str | Path) -> dict:
    """Hash actual weights and tokenizer, not a mutable served-model alias."""
    root = Path(path)
    if not (root / "config.json").is_file():
        raise ValueError(f"Missing HF model: {root}")
    files = {}
    for p in sorted(root.iterdir()):
        if p.is_file() and (p.suffix in {".safetensors", ".bin", ".json", ".jinja", ".model", ".txt"}):
            files[p.name] = file_hash(p)
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Model has no weights; evaluation requires an exported full HF checkpoint")
    return {"sha256": digest(files), "files": files}
