"""Teacher deployment must preserve asset identity and unrelated existing files."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from llmgo.rsi.artifacts import source_snapshot


def installer():
    path = Path(__file__).resolve().parents[1] / "scripts/install_teacher.py"
    spec = importlib.util.spec_from_file_location("install_teacher_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_bundle_rejects_tampered_binary_or_other_network(tmp_path, monkeypatch):
    tool = installer()
    weights = b"fixture BT4"
    monkeypatch.setattr(tool, "BT4_SHA256", hashlib.sha256(weights).hexdigest())
    files = {"lc0": b"fixture LC0", tool.MODEL_NAME: weights,
             "lc0-source.tar.gz": b"matching source archive"}
    receipt = {"ok": True, "cuda_backend_compiled": True,
               "sha256": hashlib.sha256(files["lc0"]).hexdigest(), "gpu_tested": False}
    files["build-result.json"] = json.dumps(receipt).encode()
    files["uci-result.json"] = json.dumps({"ok": True, "uci_handshake": True,
        "native_wdl_option": True, "engine_sha256": receipt["sha256"]}).encode()
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    manifest = {"schema_version": 1, "files": {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in files.items()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert tool.verify_assets(tmp_path) == manifest
    (tmp_path / "lc0").write_bytes(b"corrupted LC0")
    with pytest.raises(ValueError, match="identity mismatch"):
        tool.verify_assets(tmp_path)
    manifest["files"][tool.MODEL_NAME]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differ from"):
        tool.verify_assets(tmp_path)


def test_install_is_idempotent_and_refuses_existing_different_file(tmp_path):
    tool = installer()
    source = tmp_path / "bundle/lc0"
    source.parent.mkdir()
    source.write_bytes(b"verified LC0 binary")
    target = tmp_path / "tools/bin/lc0"
    assert tool.install_file(source, target, 0o755) == "installed"
    identity = target.stat()
    assert tool.install_file(source, target, 0o755) == "already_present"
    assert target.stat().st_mtime_ns == identity.st_mtime_ns
    target.write_bytes(b"another task's LC0")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tool.install_file(source, target, 0o755)
    assert target.read_bytes() == b"another task's LC0"


def test_teacher_assets_are_not_copied_into_each_research_round(tmp_path):
    repo = tmp_path / "repo"
    (repo / "teacher_assets").mkdir(parents=True)
    (repo / "teacher_assets/BT4.pb.gz").write_bytes(b"large external teacher asset")
    (repo / "research.py").write_text("pass\n")
    result = source_snapshot(repo, tmp_path / "snapshot")
    assert set(result["files"]) == {"research.py"}
    assert not (tmp_path / "snapshot/teacher_assets").exists()
