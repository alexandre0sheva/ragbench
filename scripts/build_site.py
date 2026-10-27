"""Build the documentation site (mkdocs-material) from the repository's Markdown.

    python scripts/build_site.py                 # stage build/site_src, then `mkdocs build --strict` into build/site
    python scripts/build_site.py --stage-only    # only stage (what the tests and a `mkdocs serve` start from)
    python scripts/build_site.py --out DIR       # build somewhere else

The Markdown stays where GitHub renders it. The site is built from a staged copy: README.md becomes the home page, `docs/` is the root, and every
relative link is rewritten to follow the file it points at, or to the repository on GitHub when the site does not carry the file (configs,
examples, source). The Leaderboard page (`leaderboard.md`) and its filterable twin (`leaderboard-explorer/index.html`) are generated from the result files
committed under `docs/benchmarks/*-run/`, so a new run updates them by regenerating, and CI rebuilds them on every push.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "site_src"
OUT = ROOT / "build" / "site"
UNPUBLISHED = ("docs/superpowers",)  # working plans, not documentation
ROOT_PAGES = {"README.md": "index.md", "CONTRIBUTING.md": "CONTRIBUTING.md", "CHANGELOG.md": "CHANGELOG.md"}
_LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))|((?:href|src)=\")([^\"]+)(\")")
_SCHEME = re.compile(r"([a-z][a-z0-9+.-]*:|#|/)", re.I)


LEADERBOARD_GUIDE = (
    "**Looking for the current benchmark?** It is [MultiHop-RAG](benchmarks/multihop-rag.md), with its analysis; "
    "[Where the results are](finding-results.md) says which run below is which and how to read them."
)

def repository_url() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return str(project["urls"]["Repository"]).rstrip("/")


def site_files(root: Path) -> dict[PurePosixPath, PurePosixPath]:
    """Repository path -> path on the site, for every file the site carries."""
    files = {PurePosixPath(source): PurePosixPath(target) for source, target in ROOT_PAGES.items() if (root / source).is_file()}
    for path in sorted((root / "docs").rglob("*")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_file() and not any(str(relative).startswith(prefix) for prefix in UNPUBLISHED) and "__pycache__" not in relative.parts:
            files[relative] = relative.relative_to("docs")
    return files


def _rewrite_target(target: str, source: PurePosixPath, files: dict[PurePosixPath, PurePosixPath], root: Path, url: str) -> str:
    if _SCHEME.match(target):
        return target
    path, hash_, anchor = target.partition("#")
    path = path.split("?")[0]
    if not path:
        return target
    resolved = PurePosixPath(posixpath.normpath(posixpath.join(source.parent.as_posix(), path)))
    if resolved.parts[:1] == ("..",):
        return target
    if resolved in files:
        relative = posixpath.relpath(files[resolved].as_posix(), files[source].parent.as_posix())
        return relative + hash_ + anchor
    on_disk = root / resolved
    if on_disk.is_file():
        return f"{url}/blob/main/{resolved.as_posix()}{hash_}{anchor}"
    if on_disk.is_dir():
        return f"{url}/tree/main/{resolved.as_posix()}{hash_}{anchor}"
    return target  # points at nothing: left as written, so the strict build names it


def rewrite(text: str, source: PurePosixPath, files: dict[PurePosixPath, PurePosixPath], root: Path) -> str:
    """`text` (the Markdown of repository file `source`) with every relative link pointing at where the site keeps the file, or at GitHub."""
    url = repository_url()
    out: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        # leave inline code alone: only text outside `...` spans is rewritten
        parts = re.split(r"(`[^`\n]*`)", line)
        for index in range(0, len(parts), 2):
            def replace(match: re.Match[str]) -> str:
                if match.group(1):
                    return match.group(1) + _rewrite_target(match.group(2), source, files, root, url) + match.group(3)
                return match.group(4) + _rewrite_target(match.group(5), source, files, root, url) + match.group(6)

            parts[index] = _LINK.sub(replace, parts[index])
        out.append("".join(parts))
    return "".join(out)


def stage(root: Path, dest: Path) -> None:
    """Copy the site's files into `dest` (emptied first), rewrite the Markdown, and generate the Leaderboard."""
    from ragbench.reporting.site import leaderboard_markdown, write_site

    if dest.exists():
        shutil.rmtree(dest)
    files = site_files(root)
    for source, target in files.items():
        destination = dest / target.as_posix()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".md" and not any(part.endswith("-run") for part in source.parts):  # a run folder's reports are copied as written
            destination.write_text(rewrite((root / source).read_text(encoding="utf-8"), source, files, root), encoding="utf-8")
        else:
            shutil.copyfile(root / source, destination)  # images, and the result files of committed runs, as they are
    benchmarks = root / "docs" / "benchmarks"
    leaderboard = leaderboard_markdown(benchmarks)
    if benchmarks.is_dir():  # under the title: where the current benchmark is, since this page lists every run's table alike
        leaderboard = leaderboard.replace("\n\n", f"\n\n{LEADERBOARD_GUIDE}\n\n", 1)
        leaderboard += "\n[Filter and sort every system of every run](leaderboard-explorer/index.html){ .md-button }\n"
    (dest / "leaderboard.md").write_text(leaderboard, encoding="utf-8")
    if benchmarks.is_dir() and any(benchmarks.glob("*-run")):
        write_site(benchmarks, dest / "leaderboard-explorer", title="RAGBench leaderboard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage-only", action="store_true", help="Only stage the sources; do not run mkdocs.")
    parser.add_argument("--out", type=Path, default=OUT, help="Where mkdocs writes the site (default: build/site).")
    args = parser.parse_args()
    stage(ROOT, STAGE)
    print(f"Staged {sum(1 for p in STAGE.rglob('*') if p.is_file())} files in {STAGE.relative_to(ROOT)}")
    if args.stage_only:
        return 0
    command = [sys.executable, "-m", "mkdocs", "build", "--strict", "--config-file", str(ROOT / "mkdocs.yml"), "--site-dir", str(args.out.resolve())]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
