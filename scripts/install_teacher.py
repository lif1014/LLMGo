#!/usr/bin/env python3
"""Verify and install bundled LC0/BT4 into the existing site paths, without overwriting other files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from llmgo.rsi.contracts import BT4_SHA256, MODEL_NAME, inside, load_site


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_assets(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported teacher asset manifest")
    files = manifest["files"]
    required = {"lc0", MODEL_NAME, "lc0-source.tar.gz", "build-result.json", "uci-result.json"}
    if not required.issubset(files):
        raise ValueError("Incomplete LC0/BT4 bundle")
    if files[MODEL_NAME]["sha256"] != BT4_SHA256:
        raise ValueError("Teacher weights differ from the initial BT4-it332 recipe")
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != set(files) | {"manifest.json"}:
        raise ValueError("Teacher bundle file set changed")
    for name, record in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe teacher asset path")
        path = root / relative
        if path.is_symlink():
            raise ValueError("Teacher asset symlinks are not allowed")
        path = inside(root, path)
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Teacher asset identity mismatch: " + name)
    receipt = json.loads((root / "build-result.json").read_text())
    if not receipt.get("ok") or not receipt.get("cuda_backend_compiled"):
        raise ValueError("LC0 CUDA build did not pass its build checks")
    if receipt["sha256"] != files["lc0"]["sha256"]:
        raise ValueError("LC0 build receipt differs from bundled binary")
    uci = json.loads((root / "uci-result.json").read_text())
    if (not uci.get("ok") or not uci.get("uci_handshake") or not uci.get("native_wdl_option")
            or uci.get("engine_sha256") != files["lc0"]["sha256"]):
        raise ValueError("LC0 UCI/WDL interface check differs from bundled binary")
    return manifest


def check_destination(source, target):
    if target.exists() and (not target.is_file() or sha(source) != sha(target)):
        raise FileExistsError(f"Refusing to overwrite different existing asset: {target}")


def install_file(source, target, mode):
    check_destination(source, target)
    if target.exists():
        if mode & 0o111 and not os.access(target, os.X_OK):
            raise PermissionError(f"Existing LC0 is not executable; inspect its permissions: {target}")
        return "already_present"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".llmgo-teacher-", dir=target.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as stream:
            shutil.copyfileobj(stream, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(mode)
        # Link atomically fails if another process creates the destination.
        try:
            os.link(temporary, target)
        except FileExistsError:
            check_destination(source, target)
    finally:
        temporary.unlink(missing_ok=True)
    return "installed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=REPO / "teacher_assets")
    parser.add_argument("--site", type=Path, default=REPO / "configs/rsi/site.beijing.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = verify_assets(args.assets)
    if args.verify_only:
        print(json.dumps({"verified": True, "gpu_tested": manifest.get("gpu_tested", False)}))
        return
    site = load_site(args.site)
    root = Path(args.assets).resolve()
    destinations = [(root / "lc0", inside(site["user_root"], site["lc0_bin"]), 0o755),
                    (root / MODEL_NAME, inside(site["user_root"], site["reward_weights"]), 0o644)]
    for source, target, mode in destinations:
        check_destination(source, target)
        if target.exists() and mode & 0o111 and not os.access(target, os.X_OK):
            raise PermissionError(f"Existing LC0 is not executable: {target}")
    result = {str(target): install_file(source, target, mode) for source, target, mode in destinations}
    print(json.dumps({"assets": result, "gpu_tested": manifest.get("gpu_tested", False),
                      "note": "Run doctor --worker --ppo and a real LC0 inference check in a GPU job."}))


if __name__ == "__main__":
    main()
