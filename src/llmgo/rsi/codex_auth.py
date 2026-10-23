"""Resolve Codex provider credentials without persisting their values."""
from __future__ import annotations

from pathlib import Path
import tomllib


def resolve_auth(config, environment):
    env = dict(environment)
    provider = config.get("model_provider", "openai")
    settings = config.get("model_providers", {}).get(provider, {})
    key_name = settings.get("env_key") or "OPENAI_API_KEY"
    inline = settings.get("experimental_bearer_token") or ""
    if not isinstance(inline, str):
        raise ValueError("experimental_bearer_token must be a string")
    secrets = tuple(dict.fromkeys(value for value in (
        inline, env.get(key_name, ""), env.get("OPENAI_API_KEY", "")
    ) if value))
    if inline:
        # Keep the test and RSI consistent even if the old env_key line remains.
        # Only the Codex child environment changes; nothing is exported globally.
        env[key_name] = inline
        if settings.get("requires_openai_auth"):
            env["OPENAI_API_KEY"] = inline
        source = "config.experimental_bearer_token"
    else:
        source = "environment" if env.get(key_name) else "missing"
    return env, {"api_key_env": key_name, "auth_source": source,
                 "api_key_present": source != "missing"}, secrets


def runtime_auth(site, environment):
    path = Path(site["codex_home"]) / "config.toml"
    if not path.is_file():
        return dict(environment), ()
    env, _, secrets = resolve_auth(tomllib.loads(path.read_text()), environment)
    return env, secrets


def scrub_known(value, secrets):
    if isinstance(value, dict):
        return {key: scrub_known(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_known(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in sorted(set(secrets), key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value
