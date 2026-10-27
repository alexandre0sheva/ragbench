"""Create or update the repository's issue labels from `.github/labels.yml` (a maintainer task; needs the GitHub CLI, `gh`, signed in).

    python scripts/sync_labels.py            # print the `gh label create` commands, change nothing
    python scripts/sync_labels.py --apply    # run them (`--force` updates a label that exists)
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml

LABELS = Path(__file__).resolve().parents[1] / ".github" / "labels.yml"


def commands() -> list[list[str]]:
    labels = yaml.safe_load(LABELS.read_text(encoding="utf-8"))
    return [["gh", "label", "create", label["name"], "--color", label["color"], "--description", label["description"], "--force"] for label in labels]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="run the commands instead of printing them")
    args = parser.parse_args()
    for command in commands():
        if args.apply:
            subprocess.run(command, check=True)
        else:
            print(shlex.join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
