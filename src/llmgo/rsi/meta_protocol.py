"""Explicit research completion and classified, recoverable Meta failures."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from .artifacts import snapshot_files
from .contracts import REPO, atomic_json, digest, file_hash, inside, utc


class MetaFailure(RuntimeError):
    def __init__(self, message, *, code="transport", retryable=True):
        super().__init__(message)
        self.code, self.retryable = code, retryable


def service_failure(message, *, code="transport"):
    """Do not retry credential/configuration failures as transient outages."""
    text = str(message)
    permanent = re.search(r"\b(?:401|403)\b|unauthorized|invalid[_ ]api[_ ]key|"
                          r"authentication|permission denied|model.{0,40}not found|"
                          r"insufficient[_ ]quota", text, re.IGNORECASE)
    return MetaFailure(text, code="authentication_or_config" if permanent else code,
                       retryable=not bool(permanent))


def result_object(value):
    if (not isinstance(value, dict) or set(value) != {"status", "summary", "proposal_path"}
            or not isinstance(value["summary"], str) or not value["summary"].strip()):
        raise ValueError("Expected status, nonempty summary and proposal_path")
    if value["status"] == "ready":
        if isinstance(value["proposal_path"], str) and value["proposal_path"].strip():
            return value
    elif value["status"] == "needs_input" and value["proposal_path"] is None:
        return value
    raise ValueError("ready requires proposal_path; needs_input requires null")


def strict_object(text):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate JSON key")
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def workspace_hash(workspace):
    return digest({relative.as_posix(): file_hash(path) for relative, path in snapshot_files(Path(workspace))})


class ResearchHandshake:
    """A request-scoped commit, outside the editable source snapshot.

    The existence of proposal.json is never evidence of completion. Only a
    deliberate commit (or the existing explicit final JSON) authorizes prepare.
    """
    def __init__(self, campaign_root, workspace, request_id):
        if not re.fullmatch(r"[0-9a-f]{64}", request_id):
            raise ValueError("Invalid research request ID")
        self.workspace = Path(workspace).resolve()
        self.folder = Path(campaign_root).resolve()/"meta-requests"/request_id
        self.request_id = request_id
        self.request_path = self.folder/"request.json"
        self.result_path = self.folder/"result.json"

    def publish(self):
        command = [sys.executable, str(REPO/"scripts/finish_research.py"), "--request", str(self.request_path)]
        atomic_json(self.request_path, {"schema_version": 1, "request_id": self.request_id,
            "workspace": str(self.workspace), "result_path": str(self.result_path), "finish_command": command})
        return command

    @classmethod
    def from_request(cls, request_path):
        path = Path(request_path).resolve()
        request = strict_object(path.read_text())
        handshake = cls(path.parents[2], request["workspace"], request["request_id"])
        if (request.get("schema_version") != 1 or path != handshake.request_path
                or request.get("result_path") != str(handshake.result_path)):
            raise ValueError("Research request path/identity mismatch")
        # Both the campaign and workspace must be below the same user root.
        # Controller enforces user_root when publishing; no arbitrary result_path is used here.
        return handshake

    def commit(self, answer):
        answer = result_object(answer.copy())
        proposal_sha = source_sha = None
        if answer["status"] == "ready":
            path = Path(answer["proposal_path"])
            path = inside(self.workspace, path if path.is_absolute() else self.workspace/path)
            if path.parent != self.workspace:
                raise ValueError("Use a proposal in the registered workspace root")
            proposal = strict_object(path.read_text())
            if not isinstance(proposal, dict) or not proposal.get("hypothesis"):
                raise ValueError("Write the completed proposal before committing")
            if proposal["hypothesis"] == "Describe the next experiment":
                raise ValueError("The initial proposal template is not a completed experiment")
            answer["proposal_path"] = str(path)
            proposal_sha, source_sha = file_hash(path), workspace_hash(self.workspace)
        atomic_json(self.result_path, {"schema_version": 1, "request_id": self.request_id,
            "workspace": str(self.workspace), "answer": answer, "proposal_sha256": proposal_sha,
            "workspace_sha256": source_sha, "committed_at": utc()})
        return answer

    def read(self):
        if not self.result_path.exists():
            return None
        if self.result_path.is_symlink() or self.result_path.stat().st_size > 65536:
            raise ValueError("Invalid research result file")
        result = strict_object(self.result_path.read_text())
        if (result.get("schema_version") != 1 or result.get("request_id") != self.request_id
                or result.get("workspace") != str(self.workspace)):
            raise ValueError("Stale or mismatched research completion")
        answer = result_object(result.get("answer"))
        if answer["status"] == "ready":
            path = inside(self.workspace, answer["proposal_path"])
            if (path.parent != self.workspace or result.get("proposal_sha256") != file_hash(path)
                    or result.get("workspace_sha256") != workspace_hash(self.workspace)):
                raise ValueError("Research files changed after completion; commit again after finishing edits")
        return answer
