"""Hashed reproducible snapshots. Research edits happen in a separate checkout."""
from __future__ import annotations
import shutil
from pathlib import Path
from .contracts import SCHEMA, atomic_json, digest, file_hash, inside, read_json

EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "runs", "engines",
            "base_model", "final_model", "models", "node_modules"}


def snapshot_files(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(x in EXCLUDED for x in relative.parts) or path.suffix==".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Source symlinks are not allowed: {path}")
        if path.is_file() and relative.as_posix() not in {"source-manifest.json", "context.json"}:
            yield relative, path


def source_snapshot(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(repo):
        raise ValueError("Snapshot destination cannot be inside its source tree")
    files = {}
    for relative,path in snapshot_files(repo):
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        files[relative.as_posix()] = file_hash(target)
    destination.mkdir(parents=True,exist_ok=True)
    result = {"files":files,"sha256":digest(files),"root":str(destination)}
    atomic_json(destination/"source-manifest.json",result)
    return result


def verify_source(source):
    root = Path(source["root"])
    if digest(source["files"]) != source["sha256"]:
        raise ValueError("Corrupt source manifest")
    observed = {r.as_posix():file_hash(p) for r,p in snapshot_files(root)}
    if observed != source["files"]:
        raise ValueError("Frozen source changed (content or file set)")


def write_manifest(path,payload):
    wrapper = {"schema_version":SCHEMA,"payload":payload,"sha256":digest(payload)}
    atomic_json(path,wrapper)
    return wrapper


def load_manifest(path,expected_sha=None):
    wrapper = read_json(path)
    if wrapper.get("schema_version")!=SCHEMA or digest(wrapper["payload"])!=wrapper["sha256"]:
        raise ValueError("Invalid manifest hash/schema")
    if expected_sha and expected_sha!=wrapper["sha256"]:
        raise ValueError("Manifest differs from submitted identity")
    payload = wrapper["payload"]
    root = payload["site"]["user_root"]
    inside(root,payload["output"])
    for key in ("source","candidate_source","reference_source"):
        if payload.get(key):
            inside(root,payload[key]["root"])
            verify_source(payload[key])
    if payload.get("skills_path"):
        inside(root,payload["skills_path"])
        if file_hash(payload["skills_path"])!=payload["skills_sha256"]:
            raise ValueError("Frozen candidate skills changed")
    return wrapper
