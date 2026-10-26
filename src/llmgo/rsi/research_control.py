"""Persistent operator hold, independent of campaign state and its writer lock."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from .contracts import REPO, atomic_json, identifier, inside, read_json, utc


class ResearchPaused(RuntimeError):
    def __init__(self, control):
        super().__init__(control.get("reason") or "Research paused by operator")
        self.control = control


def read_control(root):
    path = Path(root)/"research-control.json"
    if not path.exists():
        return {"schema_version": 1, "paused": False, "revision": 0}
    if path.is_symlink() or path.stat().st_size > 16384:
        raise ValueError("Invalid research control file; research must not start")
    value = read_json(path)
    if (value.get("schema_version") != 1 or type(value.get("paused")) is not bool
            or type(value.get("revision")) is not int or value["revision"] < 1):
        raise ValueError("Invalid research control identity; research must not start")
    return value


def set_control(root, paused, reason):
    root = Path(root)
    if not (root/"campaign.json").is_file():
        raise ValueError("Unknown campaign; no control file written")
    if type(paused) is not bool or not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
        raise ValueError("Use a boolean hold and a nonempty reason of at most 4000 characters")
    with (root/"research-control.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = read_control(root)
        if old["paused"] == paused and old.get("reason") == reason:
            return old
        value = {"schema_version": 1, "paused": paused, "revision": old["revision"]+1,
                 "reason": reason, "updated": utc(), "source": "operator"}
        atomic_json(root/"research-control.json", value)
        return value


def require_research(root):
    control = read_control(root)
    if control["paused"]:
        raise ResearchPaused(control)
    return control


def launcher_campaign(workspace, site=None):
    """Only gate app-server launches in this site's campaign workspaces."""
    if site is None:
        path = REPO/"configs/rsi/site.beijing.json"
        if not path.is_file():
            return None
        site = read_json(path)
    workspace = Path(workspace).resolve()
    base = inside(site["user_root"], site["worktree_root"]).resolve()
    if not workspace.is_relative_to(base):
        return None
    parts = workspace.relative_to(base).parts
    if len(parts) < 2:
        return None
    root = inside(site["user_root"], Path(site["campaign_root"])/identifier(parts[0]))
    return root if (root/"campaign.json").is_file() else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pause/resume research while preserving existing jobs")
    parser.add_argument("action", choices=("pause", "resume", "status"))
    parser.add_argument("--site", default=str(REPO/"configs/rsi/site.beijing.json"))
    parser.add_argument("--campaign", default="chess-autonomous-v2")
    parser.add_argument("--reason", default="Operator maintenance: PPO protocol advantage repair")
    args = parser.parse_args(argv)
    site = read_json(args.site)
    root = inside(site["user_root"], Path(site["campaign_root"])/identifier(args.campaign))
    control = read_control(root) if args.action == "status" else set_control(root, args.action == "pause", args.reason)
    print(json.dumps({"campaign": str(root), "research_control": control,
                      "existing_cluster_jobs_stopped": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
