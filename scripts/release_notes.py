"""Check that a release tag, the package version and the changelog agree, and extract the release notes.

    python scripts/release_notes.py v0.4.0 --output release_notes.md

Run by `.github/workflows/release.yml` before anything is published. The tag must be `v` plus the version in `pyproject.toml`, `CITATION.cff` must
carry the same version, and `CHANGELOG.md` must have a non-empty `## [<version>]` section, which becomes the body of the GitHub release.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"\d+\.\d+\.\d+")


class ReleaseError(Exception):
    """The tag, the version files and the changelog do not describe the same release."""


def package_version(root: Path) -> str:
    return str(tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])


def citation_version(root: Path) -> str:
    match = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', (root / "CITATION.cff").read_text(encoding="utf-8"), re.M)
    if not match:
        raise ReleaseError("CITATION.cff has no version")
    return match.group(1)


def check_tag(tag: str, version: str) -> None:
    if tag != f"v{version}":
        raise ReleaseError(f"the tag {tag!r} does not match the package version: expected 'v{version}' (pyproject.toml says {version})")


def changelog_section(text: str, version: str) -> str:
    """The body of `## [version]` in a Keep a Changelog file, without its heading."""
    heading = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n", text, re.M) if VERSION.fullmatch(version) else None
    if heading is None:
        raise ReleaseError(f"CHANGELOG.md has no section for version {version!r} (expected a '## [{version}] - YYYY-MM-DD' heading)")
    following = re.search(r"^## ", text[heading.end() :], re.M)
    body = text[heading.end() : heading.end() + following.start() if following else len(text)].strip()
    if not body:
        raise ReleaseError(f"the CHANGELOG.md section for {version} is empty")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("tag", help="the release tag, for example v0.4.0")
    parser.add_argument("--root", type=Path, default=ROOT, help="the repository root (default: this checkout)")
    parser.add_argument("--output", type=Path, help="write the notes here instead of to standard output")
    args = parser.parse_args(argv)
    try:
        version = package_version(args.root)
        check_tag(args.tag, version)
        if citation_version(args.root) != version:
            raise ReleaseError(f"CITATION.cff says {citation_version(args.root)} but pyproject.toml says {version}")
        notes = changelog_section((args.root / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    except ReleaseError as error:
        print(f"release check failed: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.write_text(notes + "\n", encoding="utf-8")
    else:
        print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
