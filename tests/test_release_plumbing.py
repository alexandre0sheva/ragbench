"""Releasing: the tag, the changelog and the version agree; the release workflow publishes only what the package checks passed, without a stored token; the image is small and non-root."""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
release_notes = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["release_notes"] = release_notes
spec.loader.exec_module(release_notes)  # type: ignore[union-attr]

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- Something not released yet.

## [0.2.0] - 2026-06-12

### Added

- Two.

### Fixed

- Two again.

## [0.1.0] - 2026-05-09

- One.
"""


def _workflow(name: str) -> dict:
    # PyYAML reads the key `on` as the boolean True; the tests look it up as "on".
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    loaded["on"] = loaded.pop(True)
    return loaded


def test_the_changelog_section_is_what_sits_between_its_heading_and_the_next():
    section = release_notes.changelog_section(CHANGELOG, "0.2.0")
    assert section.startswith("### Added") and section.endswith("- Two again.")
    assert "One." not in section and "Something not released" not in section
    assert release_notes.changelog_section(CHANGELOG, "0.1.0") == "- One."


@pytest.mark.parametrize("version", ["0.3.0", "0.2", "Unreleased"])
def test_a_version_without_a_changelog_section_is_refused(version):
    with pytest.raises(release_notes.ReleaseError, match="CHANGELOG"):
        release_notes.changelog_section(CHANGELOG, version)


def test_an_empty_changelog_section_is_refused():
    with pytest.raises(release_notes.ReleaseError, match="empty"):
        release_notes.changelog_section("## [Unreleased]\n\n## [0.4.0] - 2026-11-01\n\n## [0.3.0] - 2026-10-01\n\n- x\n", "0.4.0")


def test_the_tag_must_be_v_plus_the_package_version():
    release_notes.check_tag("v0.4.0", "0.4.0")
    for bad in ("0.4.0", "v0.4.1", "v0.4.0rc1", "v0.4"):
        with pytest.raises(release_notes.ReleaseError, match="tag"):
            release_notes.check_tag(bad, "0.4.0")


def test_the_version_comes_from_pyproject_and_the_citation_file_agrees():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert release_notes.package_version(ROOT) == pyproject["project"]["version"]
    assert release_notes.citation_version(ROOT) == pyproject["project"]["version"]


def test_the_command_writes_the_notes_and_fails_on_a_mismatch(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.2.0"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text('version: "0.2.0"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    out = tmp_path / "notes.md"
    assert release_notes.main(["v0.2.0", "--root", str(tmp_path), "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("### Added")
    assert release_notes.main(["v0.2.1", "--root", str(tmp_path), "--output", str(out)]) == 1
    assert "tag" in capsys.readouterr().err
    (tmp_path / "CITATION.cff").write_text('version: "0.1.0"\n', encoding="utf-8")
    assert release_notes.main(["v0.2.0", "--root", str(tmp_path), "--output", str(out)]) == 1
    assert "CITATION.cff" in capsys.readouterr().err


def test_every_released_version_in_the_repository_has_notes():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release_notes.changelog_section(text, "0.3.0")


def test_the_package_checks_are_one_reusable_workflow_that_ci_and_the_release_both_call():
    package = _workflow("package.yml")
    assert "workflow_call" in package["on"]
    commands = "\n".join(step.get("run", "") for step in package["jobs"]["package"]["steps"])
    for needed in ("python -m build", "twine check --strict", "ragbench/_data/demo/questions.jsonl", "ragbench/_data/hard/questions.jsonl", "/tmp/clean/bin/pip install dist/*.whl", "ragbench demo"):
        assert needed in commands, f"the package checks do not run `{needed}`"
    uploads = [step for step in package["jobs"]["package"]["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert uploads and uploads[0]["with"]["name"] == "dist", "the release publishes the files that passed the checks, so they must be uploaded"
    assert _workflow("ci.yml")["jobs"]["build"]["uses"] == "./.github/workflows/package.yml"
    assert _workflow("release.yml")["jobs"]["package"]["uses"] == "./.github/workflows/package.yml"


def test_the_release_runs_on_version_tags_only_and_is_read_only_by_default():
    release = _workflow("release.yml")
    assert release["on"] == {"push": {"tags": ["v*"]}}
    assert release["permissions"] == {"contents": "read"}


def test_publishing_uses_trusted_publishing_and_needs_the_checks_and_the_tag():
    jobs = _workflow("release.yml")["jobs"]
    publish = jobs["publish-pypi"]
    assert set(publish["needs"]) >= {"package", "verify"}
    assert publish["permissions"]["id-token"] == "write"  # OIDC: no token is stored anywhere
    assert publish["environment"]["name"] == "pypi"
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert not re.search(r"password|PYPI_API_TOKEN|TWINE_", text, re.I), "a token has no place in a trusted-publishing workflow"
    assert set(jobs["github-release"]["needs"]) >= {"publish-pypi", "verify"}
    assert jobs["github-release"]["permissions"] == {"contents": "write"}
    assert [name for name, job in jobs.items() if "id-token" in job.get("permissions", {})] == ["publish-pypi"]


def test_the_verify_job_checks_the_tag_against_the_version_and_the_changelog():
    verify = _workflow("release.yml")["jobs"]["verify"]
    commands = "\n".join(step.get("run", "") for step in verify["steps"])
    assert "scripts/release_notes.py" in commands and "$TAG" in commands
    assert verify["steps"][-2]["env"]["TAG"] == "${{ github.ref_name }}"


def test_release_workflow_actions_are_pinned_to_a_major_version():
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8") + (WORKFLOWS / "package.yml").read_text(encoding="utf-8")
    for used in re.findall(r"uses:\s*(\S+)", text):
        assert used.startswith("./") or re.fullmatch(r"[\w.-]+/[\w.-]+@(v\d+|release/v1)", used), f"unpinned: {used}"


def test_the_release_workflow_names_only_files_that_exist():
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    for path in re.findall(r"(?:scripts|\.github/workflows)/[\w./-]+", text):
        assert (ROOT / path).exists(), f"release.yml refers to {path}, which does not exist"


def test_ci_builds_the_image_and_checks_it_runs_without_a_key_as_a_non_root_user():
    docker = _workflow("ci.yml")["jobs"]["docker"]
    commands = "\n".join(step.get("run", "") for step in docker["steps"])
    assert "docker build" in commands and "demo" in commands and "--mock" in commands
    assert "id -u" in commands, "the job must assert the container does not run as root"


def test_the_dockerfile_is_slim_non_root_and_has_the_cli_as_its_entrypoint():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM python:\d+\.\d+-slim", text, re.M)
    assert 'ENTRYPOINT ["ragbench"]' in text
    user = re.findall(r"^USER\s+(\S+)", text, re.M)
    assert user and user[-1] not in {"root", "0"}
    instructions = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert not [line for line in instructions if "API_KEY" in line or ".env" in line], "no key or .env file may be baked into the image"


def test_the_docker_context_leaves_secrets_and_scratch_out():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for name in (".env", ".env.*", ".git", ".venv", "results", ".ragbench_cache", "dist", "build"):
        assert name in ignored


def test_the_checklist_and_the_readme_describe_the_release_path():
    checklist = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")
    for needed in ("trusted publisher", "git tag", "release.yml"):
        assert needed.lower() in checklist.lower(), f"release-checklist.md does not mention {needed}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install ragbench" in readme and "uvx ragbench demo" in readme
    assert "Not to be confused with" in readme
