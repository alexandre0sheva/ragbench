"""The docs site: the repository's Markdown staged for mkdocs-material, links rewritten, and the Leaderboard page generated from committed results."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/alexandre0sheva/ragbench"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["build_site"] = module  # dataclasses look their module up
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def builder():
    return load_builder()


@pytest.fixture(scope="module")
def staged(builder, tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("site_src")
    builder.stage(ROOT, dest)
    return dest


# -- rewriting links ------------------------------------------------------------------------------------------------------------------------


def test_links_between_staged_files_stay_relative_and_follow_a_file_that_moved(builder):
    files = {PurePosixPath("README.md"): PurePosixPath("index.md"), PurePosixPath("docs/cli.md"): PurePosixPath("cli.md"), PurePosixPath("CHANGELOG.md"): PurePosixPath("CHANGELOG.md")}
    readme = builder.rewrite("[cli](docs/cli.md#ragbench-run) and ![x](docs/cli.md)", PurePosixPath("README.md"), files, ROOT)
    assert readme == "[cli](cli.md#ragbench-run) and ![x](cli.md)"
    doc = builder.rewrite("[log](../CHANGELOG.md), [self](#top), [web](https://example.com/a.md), [mail](mailto:a@b.c)", PurePosixPath("docs/cli.md"), files, ROOT)
    assert doc == "[log](CHANGELOG.md), [self](#top), [web](https://example.com/a.md), [mail](mailto:a@b.c)"


def test_a_link_to_a_file_the_site_does_not_carry_goes_to_the_repository_on_github(builder):
    files = {PurePosixPath("docs/extending.md"): PurePosixPath("extending.md")}
    text = builder.rewrite("[cfg](../configs/all.yaml) [src](../src/ragbench/) [anchor](../configs/all.yaml#L3)", PurePosixPath("docs/extending.md"), files, ROOT)
    assert f"]({REPO_URL}/blob/main/configs/all.yaml)" in text
    assert f"]({REPO_URL}/tree/main/src/ragbench)" in text  # a directory is a tree, not a blob
    assert f"]({REPO_URL}/blob/main/configs/all.yaml#L3)" in text


def test_html_attributes_are_rewritten_like_links_and_code_is_left_alone(builder):
    files = {PurePosixPath("README.md"): PurePosixPath("index.md"), PurePosixPath("docs/assets/a.png"): PurePosixPath("assets/a.png")}
    text = '<img src="docs/assets/a.png" width="49%">\n```\n[not a link](docs/assets/a.png)\n```\n`[inline](docs/assets/a.png)`\n'
    out = builder.rewrite(text, PurePosixPath("README.md"), files, ROOT)
    assert 'src="assets/a.png"' in out and "[not a link](docs/assets/a.png)" in out and "`[inline](docs/assets/a.png)`" in out


def test_a_link_to_nothing_is_left_as_written_so_the_build_can_report_it(builder):
    assert builder.rewrite("[gone](nope.md)", PurePosixPath("docs/cli.md"), {}, ROOT) == "[gone](nope.md)"


# -- staging the real repository ----------------------------------------------------------------------------------------------------------


def test_the_readme_is_the_home_page_and_the_plans_are_not_published(staged):
    assert (staged / "index.md").read_text(encoding="utf-8").lstrip().startswith("#") or "RAGBench" in (staged / "index.md").read_text(encoding="utf-8")
    assert not (staged / "README.md").exists() and not (staged / "superpowers").exists()
    for page in ("cli.md", "methodology.md", "observability.md", "CHANGELOG.md", "CONTRIBUTING.md", "assets/report-desktop-light.png", "benchmarks/example-run/metrics_summary.csv"):
        assert (staged / page).is_file(), page


def test_the_leaderboard_page_and_its_interactive_twin_are_generated_from_the_committed_runs(staged):
    page = (staged / "leaderboard.md").read_text(encoding="utf-8")
    assert page.startswith("# Leaderboard") and "`example-run`" in page and "bm25_default" in page
    interactive = (staged / "leaderboard-explorer" / "index.html").read_text(encoding="utf-8")
    assert 'id="site-data"' in interactive and "bm25_default" in interactive
    assert "(leaderboard-explorer/index.html)" in page


def test_every_relative_link_in_the_staged_site_points_at_a_file_it_carries(staged):
    link = re.compile(r"\]\(([^)\s]+)")
    broken = []
    for page in staged.rglob("*.md"):
        if "benchmarks" in page.relative_to(staged).parts[:1] and page.name != "README.md":
            continue  # the result files of committed runs are data, not documentation
        in_fence = False
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            if in_fence:
                continue
            for target in link.findall(line):
                if re.match(r"([a-z][a-z0-9+.-]*:|#|/)", target):
                    continue
                path = target.split("#")[0].split("?")[0]
                if path and not (page.parent / path).exists():
                    broken.append(f"{page.relative_to(staged)}:{number}: {target}")
    assert not broken, broken


def test_the_nav_lists_only_pages_the_site_has_and_every_guide_has_a_place(staged):
    class Loader(yaml.SafeLoader):
        """mkdocs.yml names Python objects (`!!python/name:`) that only mkdocs may import; here they are read as nothing."""

    Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=Loader)  # noqa: S506 (a SafeLoader subclass)

    def pages(nav):
        for item in nav:
            for value in item.values() if isinstance(item, dict) else [item]:
                if isinstance(value, list):
                    yield from pages(value)
                else:
                    yield value

    listed = set(pages(config["nav"]))
    assert all((staged / page).is_file() for page in listed), sorted(p for p in listed if not (staged / p).is_file())
    guides = {p.name for p in staged.glob("*.md")}
    assert guides <= listed, f"pages the site carries but the nav forgets: {sorted(guides - listed)}"


@pytest.mark.slow
def test_mkdocs_builds_the_site_in_strict_mode(builder, tmp_path):
    pytest.importorskip("mkdocs")
    pytest.importorskip("material")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py"), "--out", str(tmp_path / "out")], capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert (tmp_path / "out" / "index.html").is_file() and (tmp_path / "out" / "leaderboard-explorer" / "index.html").is_file()
