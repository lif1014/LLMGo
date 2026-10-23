"""Codex isolation policy shared by research turns and connection checks."""


def sandbox_mode(site):
    # This deployment's server enforces the user's writable directory.
    # Missing keys in existing campaign snapshots use the same deployment mode.
    mode = site.get("meta_sandbox_mode", "external")
    if mode not in {"external", "workspace-write"}:
        raise ValueError("meta_sandbox_mode must be external or workspace-write")
    return mode


def thread_sandbox(site):
    # thread/start accepts SandboxMode, not the externalSandbox policy variant.
    return "danger-full-access" if sandbox_mode(site) == "external" else "workspace-write"


def turn_sandbox(site):
    if sandbox_mode(site) == "external":
        return {"type": "externalSandbox", "networkAccess": "enabled"}
    return {"type": "workspaceWrite", "writableRoots": [site["user_root"]],
            "networkAccess": True, "excludeTmpdirEnvVar": True, "excludeSlashTmp": True}
