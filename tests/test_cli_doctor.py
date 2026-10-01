"""`doctor` and `completion`: what they check, what they say, and that a key is never printed."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest
from cli_support import demo_config, invoke

from ragbench.cli.doctor import INFO, OK, WARN, environment_checks
from ragbench.models import cost as pricing

SENTINEL = "sk-SENTINEL-do-not-print-0123456789abcdef"


def _by_name(checks):
    return {check.name: check for check in checks}


def test_doctor_never_prints_an_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL + "-anthropic")
    config = demo_config(tmp_path, ["bm25"])
    for args in ([], ["--json"], ["--config", str(config)], ["--config", str(config), "--json"]):
        result = invoke("doctor", "--results-dir", str(tmp_path / "results"), "--cache-dir", str(tmp_path / "cache"), *args)
        assert result.exit_code == 0, result.output
        combined = result.output + result.stdout + result.stderr
        assert SENTINEL not in combined and "SENTINEL" not in combined
        assert "OPENAI_API_KEY" in combined and "set" in combined


def test_doctor_reports_python_packages_extras_keys_folders_and_prices(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("ragbench.cli.doctor.has_api_key", lambda name: False)
    result = invoke("doctor", "--json", "--results-dir", str(tmp_path / "r"), "--cache-dir", str(tmp_path / "c"))
    payload = json.loads(result.stdout)
    checks = {(c["group"], c["name"]): c for c in payload["checks"]}
    assert payload["ok"] is True and result.exit_code == 0
    assert checks[("Environment", "Python")]["status"] == OK and checks[("Packages", "required")]["status"] == OK
    assert checks[("Credentials", "OPENAI_API_KEY")]["status"] == INFO and checks[("Credentials", "OPENAI_API_KEY")]["detail"] == "not set"
    extras = {name: c for (group, name), c in checks.items() if group == "Optional extras"}
    assert set(extras) >= {"anthropic", "pdf", "docx", "chroma", "faiss", "qdrant", "local"}
    for name, check in extras.items():
        assert check["status"] in (OK, INFO) and (check["status"] == OK or check["fix"] == f"pip install 'ragbench[{name}]'")  # an absent extra says how to get it
    assert checks[("Prices", "price table")]["detail"].startswith(f"as of {pricing.PRICING_AS_OF}")
    assert any(group == "Folders" for group, _ in checks)


def test_the_price_table_is_flagged_after_90_days():
    as_of = date.fromisoformat(pricing.PRICING_AS_OF)
    fresh = _by_name(environment_checks(Path("results"), Path(".ragbench_cache"), today=as_of + timedelta(days=90)))
    stale = _by_name(environment_checks(Path("results"), Path(".ragbench_cache"), today=as_of + timedelta(days=91)))
    assert fresh["price table"].status == OK
    assert stale["price table"].status == WARN and "91 days" in stale["price table"].detail and "pricing:" in (stale["price table"].fix or "")


def test_a_results_folder_that_cannot_be_written_fails_the_check_and_the_command(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can write anywhere")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result = invoke("doctor", "--results-dir", str(locked / "results"), "--cache-dir", str(tmp_path / "c"))
        assert result.exit_code == 1 and "FAIL" in result.output and "not writable" in result.output
        payload = json.loads(invoke("doctor", "--json", "--results-dir", str(locked), "--cache-dir", str(tmp_path / "c")).stdout)
        assert payload["ok"] is False
    finally:
        locked.chmod(0o700)


def test_a_missing_results_folder_is_fine_when_it_can_be_created(tmp_path):
    folders = {c.name: c for c in environment_checks(tmp_path / "new" / "results", tmp_path / "cache") if c.group == "Folders"}
    result = next(c for name, c in folders.items() if name.startswith("results"))
    assert result.status == OK and "will be created" in result.detail


def test_the_cache_size_is_reported(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache.sqlite3").write_bytes(b"x" * 3000)
    folders = {c.name: c for c in environment_checks(tmp_path / "r", cache)}
    entry = next(c for name, c in folders.items() if name.startswith("cache"))
    assert "2.9 KiB" in entry.detail


def test_doctor_checks_every_model_in_the_config_against_the_price_table(monkeypatch, tmp_path):
    monkeypatch.setattr("ragbench.cli.doctor.has_api_key", lambda name: False)
    config = demo_config(tmp_path, ["vector"], evaluation={"max_workers": 1, "judge_model": "mystery-judge-9"})
    payload = json.loads(invoke("doctor", "--json", "--config", str(config), "--results-dir", str(tmp_path), "--cache-dir", str(tmp_path / "c")).stdout)
    models = {c["name"]: c for c in payload["checks"] if c["group"] == "Models in the config"}
    unknown = next(c for name, c in models.items() if name.startswith("mystery-judge-9"))
    assert unknown["status"] == WARN and "no price" in unknown["detail"] and "pricing:" in unknown["fix"]
    known = [c for name, c in models.items() if not name.startswith("mystery") and name != "credentials"]
    assert known and all(c["status"] == OK for c in known)
    assert payload["ok"] is True  # a warning is not a failure


def test_a_price_in_the_config_clears_the_warning_and_leaves_no_trace(monkeypatch, tmp_path):
    monkeypatch.setattr("ragbench.cli.doctor.has_api_key", lambda name: False)
    config = demo_config(tmp_path, ["bm25"], evaluation={"max_workers": 1, "judge_model": "mystery-judge-9"}, pricing={"mystery-judge-9": {"input": 1.0, "output": 2.0}})
    payload = json.loads(invoke("doctor", "--json", "--config", str(config), "--results-dir", str(tmp_path), "--cache-dir", str(tmp_path / "c")).stdout)
    entry = next(c for c in payload["checks"] if c["name"].startswith("mystery-judge-9"))
    assert entry["status"] == OK
    assert not pricing.price_known("mystery-judge-9")  # the override lived only for the check


def test_a_missing_credential_for_a_configured_model_is_a_warning(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("ragbench.models.refs.has_api_key", lambda name: False)
    config = demo_config(tmp_path, ["vector"])
    payload = json.loads(invoke("doctor", "--json", "--config", str(config), "--results-dir", str(tmp_path), "--cache-dir", str(tmp_path / "c")).stdout)
    entry = next(c for c in payload["checks"] if c["name"] == "credentials")
    assert entry["status"] == WARN and "OPENAI_API_KEY" in entry["detail"]
    assert SENTINEL not in json.dumps(payload)


def test_doctor_with_a_broken_config_reports_the_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x}\ndataset: {documents_path: d, questions_path: q}\nsystems: [{type: nope}]\n")
    result = invoke("doctor", "--config", str(bad), "--results-dir", str(tmp_path))
    assert result.exit_code == 2 and "Invalid config" in result.output


# -- completion --------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_show_prints_a_script_for_the_shell(shell):
    result = invoke("completion", "show", "--shell", shell)
    assert result.exit_code == 0 and "_RAGBENCH_COMPLETE" in result.output and "ragbench" in result.output


def test_completion_needs_to_know_the_shell(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/tcsh")
    result = invoke("completion", "show")
    assert result.exit_code == 2 and "--shell" in result.output
    monkeypatch.setenv("SHELL", "/bin/zsh")
    assert invoke("completion", "show").exit_code == 0  # detected from $SHELL


def test_completion_install_writes_into_the_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = invoke("completion", "install", "--shell", "bash")
    assert result.exit_code == 0 and "Installed bash completion" in result.output
    installed = list(tmp_path.rglob("ragbench*"))
    assert installed and "_RAGBENCH_COMPLETE" in installed[0].read_text(encoding="utf-8")
