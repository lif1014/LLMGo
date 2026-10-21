"""Cross-component regression checks found during the complete pipeline review."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from llmgo.rsi.evaluator import runtime_helpers
from llmgo.rsi.contracts import REPO


def test_vllm_readiness_bypasses_external_proxy(monkeypatch, tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"Qwen3.5-4B"}]}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, "http://127.0.0.1:1")
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.setenv(name, "")
        helper = runtime_helpers(REPO)
        service = helper.VllmServer(Path("unchanged-model-path"), {"model_name": "Qwen3.5-4B"},
                                   tmp_path/"service.log", server.server_port, "0")
        assert service.ready_now() is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_doctor_codex_login_is_bounded_uses_own_config_and_hides_output(tmp_path, monkeypatch):
    import os
    import subprocess
    from llmgo.rsi.cli import doctor
    calls = []
    site = {"user_root":str(tmp_path), "repo":str(tmp_path/"repo"), "base_model":str(tmp_path/"model"),
            "engine":str(tmp_path/"fairy"), "codex_home":str(tmp_path/".codex-rsi"),
            "codex_bin":str(tmp_path/"codex"), "cctl_bin":"cctl"}
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "SENSITIVE_PROVIDER_INFORMATION", "SENSITIVE_TOKEN")
    monkeypatch.setattr(subprocess, "run", run)
    doctor(site)
    assert calls == []
    result = doctor(site, codex_auth=True)
    assert calls[0][0] == [site["codex_bin"], "login", "status"]
    assert calls[0][1]["timeout"] == 30
    assert calls[0][1]["env"]["CODEX_HOME"] == site["codex_home"]
    assert calls[0][1]["env"].get("HOME") == os.environ.get("HOME")
    assert "SENSITIVE" not in str(result)
    assert result["inference_tested"] is False
    assert next(c for c in result["checks"] if c["name"]=="codex_login_status")["ok"]
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30)
    monkeypatch.setattr(subprocess, "run", timeout)
    result = doctor(site, codex_auth=True)
    assert not next(c for c in result["checks"] if c["name"]=="codex_login_status")["ok"]


def test_doctor_cli_exit_status_does_not_report_failed_readiness_as_success(monkeypatch, capsys):
    import json
    from llmgo.rsi import cli
    monkeypatch.setattr(cli, "load_site", lambda path: {})
    monkeypatch.setattr(cli, "doctor", lambda *a, **k: {"ok": False, "checks": []})
    assert cli.main(["doctor"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    monkeypatch.setattr(cli, "doctor", lambda *a, **k: {"ok": True, "checks": []})
    assert cli.main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
