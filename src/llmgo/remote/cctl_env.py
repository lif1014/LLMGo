"""Resolve cctl's server URL without changing shared profiles or credentials."""
from __future__ import annotations

import os
from pathlib import Path
import re
from urllib.parse import urlsplit


def cctl_environment(environment=None, *, config_root=None):
    env = dict(os.environ if environment is None else environment)
    env["CCTL_NO_UPDATE_CHECK"] = "1"
    server = env.get("CCTL_SERVER", "").strip()
    if not server:
        root = Path(config_root) if config_root is not None else Path.home()/".cybertronctl"
        current = root/"current-profile"
        profile = env.get("CCTL_PROFILE") or (current.read_text().strip() if current.is_file() else "modelbest") or "modelbest"
        if "/" in profile or "\\" in profile or profile in {".",".."}:
            raise ValueError("Invalid CCTL_PROFILE name")
        path = root/"profiles"/profile/"server"
        if not path.is_file() and profile == "modelbest":
            path = root/"server"
        if path.is_file():
            server = path.read_text().strip()
    if server:
        # This legacy bare alias also has an expired TLS certificate. Use the
        # verified canonical endpoint of the same Cybertron deployment. Explicit
        # http(s) URLs remain operator choices and are never silently replaced.
        if server.rstrip("/") == "cybertron-hcloud.modelbest.co":
            server = "https://cybertron.modelbest.co"
        elif "://" not in server:
            server = "https://"+server
        parsed = urlsplit(server)
        if (parsed.scheme not in {"http","https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or any(c.isspace() for c in server)):
            raise ValueError("CCTL_SERVER must be an http(s) server URL without embedded credentials")
        env["CCTL_SERVER"] = server.rstrip("/")
    return env


def cctl_failure_detail(result, environment):
    message = (result.stderr or result.stdout or "No diagnostic output").strip()
    for name,value in environment.items():
        if value and re.search(r"TOKEN|KEY|SECRET|PASSWORD",name,re.I):
            message = message.replace(value,"[REDACTED]")
    message = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", message)
    message = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)((?:token|api[_-]?key|password|secret)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,}]+", r"\1[REDACTED]", message)
    return f"cctl exited {result.returncode}: {message[:2400]}"
