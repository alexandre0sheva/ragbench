from __future__ import annotations

import os
from pathlib import Path

_LOADED_ENV_PATHS: set[Path] = set()


def load_project_env(start: Path | None = None) -> None:
    """Load a local .env file without requiring python-dotenv.

    Existing process environment values win over .env values. This keeps shell
    overrides predictable while allowing `ragbench run` to work from a repo that
    contains OPENAI_API_KEY in .env.
    """
    env_path = find_env_file(start or Path.cwd())
    if not env_path or env_path in _LOADED_ENV_PATHS:
        return
    _LOADED_ENV_PATHS.add(env_path)
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def find_env_file(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def has_openai_key() -> bool:
    load_project_env()
    return bool(os.environ.get("OPENAI_API_KEY"))
