"""Build hook: ship `data/demo` and `data/hard` inside the wheel as `ragbench/_data/<suite>` (the CLI's `demo` command reads them via importlib.resources).

`data/<suite>` stays the single source of truth in the repository; a symlink under `src/` would not survive every packaging tool, so the
files are copied into the build tree instead. Everything else is configured in pyproject.toml.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

SUITES = ("demo", "hard")  # ragbench.datasets.demo_generator.SUITES


class BuildPyWithDemoData(build_py):
    def run(self) -> None:
        super().run()
        for suite in SUITES:
            source = Path(__file__).resolve().parent / "data" / suite
            if not source.is_dir():  # e.g. building from a tree that was stripped of data
                continue
            target = Path(self.build_lib) / "ragbench" / "_data" / suite
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


setup(cmdclass={"build_py": BuildPyWithDemoData})
