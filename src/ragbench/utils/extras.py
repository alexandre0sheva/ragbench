"""Helpers for optional dependencies (`pip install 'ragbench[<extra>]'`)."""

from __future__ import annotations

import importlib.util

from ragbench.models.errors import MissingExtraError


def extra_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return True  # already imported: `find_spec` cannot answer for a module without a spec (e.g. a stand-in)


def require_extra(module: str, what: str, extra: str, package: str | None = None) -> None:
    """Raise `MissingExtraError` (an `ImportError`) naming the extra to install when `module` cannot be imported.

    `package` is the distribution name when it differs from the import name (`faiss` is installed as `faiss-cpu`).
    """
    if not extra_installed(module):
        raise MissingExtraError(f"{what} needs the optional `{package or module}` package. Install it with: pip install 'ragbench[{extra}]'")
