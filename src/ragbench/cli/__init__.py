"""The `ragbench` command line (`ragbench.cli:app` is the entry point). One module per group of commands; `app.py` assembles them."""

from __future__ import annotations

from ragbench.cli.app import app
from ragbench.cli.tools import parse_chunker_spec

__all__ = ["app", "parse_chunker_spec"]
