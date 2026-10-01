"""The CI workflow is valid YAML, runs what the project promises, and only refers to things that exist (a command, an extra, a file)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import typer.main
import yaml

from ragbench.cli import app

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
CI = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps() -> list[tuple[str, dict]]:
    return [(job, step) for job, body in CI["jobs"].items() for step in body["steps"]]


def _commands() -> list[str]:
    return [text for _, step in _steps() for text in [step.get("run", "")] if text]


def test_the_workflow_has_every_job_the_project_relies_on():
    assert set(CI["jobs"]) >= {"lint", "docs", "test", "extras", "smoke", "build", "audit"}
    assert CI["jobs"]["test"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12", "3.13", "3.14"]
    assert CI["jobs"]["audit"]["continue-on-error"] is True  # advisory: a new advisory must not block unrelated work
    assert CI["jobs"]["test"]["continue-on-error"] == "${{ matrix.python-version == '3.14' }}"
    assert CI["permissions"] == {"contents": "read"}


def test_every_step_is_a_checkout_setup_or_a_command_and_actions_are_pinned_to_a_major_version():
    for job, step in _steps():
        assert ("uses" in step) != ("run" in step), f"{job}: a step either uses an action or runs a command: {step}"
        if "uses" in step:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+", step["uses"]), f"{job}: unpinned action {step['uses']}"


def test_the_gate_commands_are_in_the_workflow():
    text = "\n".join(_commands())
    for needed in ("ruff check .", "mypy", "pytest --cov", "generate_docs.py --check", "audit_docs.py", "twine check --strict", "pip-audit", "python -m build", "ragbench auto"):
        assert needed in text, f"CI does not run `{needed}`"


def test_coverage_has_a_floor_and_ci_measures_it():
    floor = PYPROJECT["tool"]["coverage"]["report"]["fail_under"]
    assert floor >= 85
    assert "--cov" in "\n".join(_commands())


def test_every_extra_ci_installs_exists():
    extras = set(PYPROJECT["project"]["optional-dependencies"])
    for command in _commands():
        for group in re.findall(r'pip install[^\n]*?"\.?\[([\w,-]+)\]"', command):
            assert set(group.split(",")) <= extras, f"CI installs an extra that does not exist: {group}"


def test_every_ragbench_command_ci_runs_exists():
    root = typer.main.get_command(app)
    names = set(root.commands) | {"--version"}
    for command in _commands():
        for line in command.splitlines():
            match = re.match(r"\s*(?:/tmp/clean/bin/)?ragbench\s+([\w-]+)", line)
            if match:
                assert match.group(1) in names, f"CI runs `ragbench {match.group(1)}`, which is not a command"


def test_files_the_workflow_names_exist():
    for command in _commands():
        for path in re.findall(r"(?:configs|data/demo|scripts)/[\w./-]+", command):
            assert (ROOT / path.rstrip(".")).exists(), f"CI refers to {path}, which does not exist"
    assert (ROOT / "scripts" / "generate_docs.py").exists()
