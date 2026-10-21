"""Immutable manifests and source snapshots; no git reset or implicit upload."""
from __future__ import annotations

import shutil
from pathlib import Path

from .contracts import SCHEMA, atomic_json, digest, file_hash, inside, read_json


def source_snapshot(repo: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    files = {}
    for folder in ("src", "scripts", "configs", ".agents", "docs"):
        base = repo / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlinks are not allowed: {path}")
            relative = path.relative_to(repo)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files[relative.as_posix()] = file_hash(target)
    result = {"files": files, "sha256": digest(files), "root": str(destination)}
    atomic_json(destination / "source-manifest.json", result)
    return result


def verify_source(source):
    root = Path(source["root"])
    if digest(source["files"]) != source["sha256"]:
        raise ValueError("Corrupt source manifest")
    for relative, expected in source["files"].items():
        path = inside(root, root / relative)
        if file_hash(path) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    # An extra Python/config file could change imports despite matching old hashes.
    observed = set()
    for folder in ("src", "scripts", "configs", ".agents", "docs"):
        for path in (root/folder).rglob("*"):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError("Frozen source contains a symlink")
            if path.is_file():
                observed.add(path.relative_to(root).as_posix())
    if observed != set(source["files"]):
        raise ValueError("Frozen source file set changed")


def write_manifest(path, payload):
    wrapper = {"schema_version": SCHEMA, "payload": payload, "sha256": digest(payload)}
    atomic_json(path, wrapper)
    return wrapper


def load_manifest(path, expected_sha=None):
    wrapper = read_json(path)
    if wrapper.get("schema_version") != SCHEMA or digest(wrapper["payload"]) != wrapper["sha256"]:
        raise ValueError("Invalid manifest hash/schema")
    if expected_sha and expected_sha != wrapper["sha256"]:
        raise ValueError("Manifest differs from submitted identity")
    payload = wrapper["payload"]
    root = payload["site"]["user_root"]
    for name in ("output", "skills_path"):
        inside(root, payload[name])
    inside(root, payload["source"]["root"])
    if file_hash(payload["skills_path"]) != payload["skills_sha256"]:
        raise ValueError("Frozen candidate skills changed")
    verify_source(payload["source"])
    return wrapper
