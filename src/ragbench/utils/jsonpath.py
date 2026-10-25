"""The small JSONPath subset that maps an external system's JSON reply: `$`, `.name`, `['name']`, `[n]` and `[*]`.

No filters, slices or recursive descent: a reply's fields are addressed by name and position, and anything more is better done
in the system's own response than in a mapping string.
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"\.(?P<name>[A-Za-z_][A-Za-z0-9_\-]*)|\[(?P<index>-?\d+)\]|\[(?P<star>\*)\]|\['(?P<quoted>[^']*)'\]|\[\"(?P<dquoted>[^\"]*)\"\]")


class JsonPathError(ValueError):
    pass


def parse(path: str) -> list[str | int | None]:
    """Steps of `path`: a field name, a list index, or None for `[*]`. Raises `JsonPathError` when it is not in the supported subset."""
    if not path.startswith("$"):
        raise JsonPathError(f"JSONPath {path!r} must start with '$' (for example $.answer or $.contexts[*].text)")
    steps: list[str | int | None] = []
    position = 1
    while position < len(path):
        match = _TOKEN.match(path, position)
        if match is None:
            raise JsonPathError(f"JSONPath {path!r}: cannot read {path[position:]!r} (supported: .name, ['name'], [0], [*])")
        if match["index"] is not None:
            steps.append(int(match["index"]))
        elif match["star"]:
            steps.append(None)
        else:
            steps.append(match["name"] or match["quoted"] or match["dquoted"] or "")
        position = match.end()
    return steps


def find(data: Any, path: str) -> list[Any]:
    """Every value `path` selects in `data` (an empty list when nothing matches)."""
    found = [data]
    for step in parse(path):
        selected: list[Any] = []
        for value in found:
            if step is None:
                selected += list(value.values()) if isinstance(value, dict) else list(value) if isinstance(value, list) else []
            elif isinstance(step, int):
                if isinstance(value, list) and -len(value) <= step < len(value):
                    selected.append(value[step])
            elif isinstance(value, dict) and step in value:
                selected.append(value[step])
        found = selected
    return found


def first(data: Any, path: str | None) -> Any:
    """The first value `path` selects, or None (also when `path` is None)."""
    if path is None:
        return None
    values = find(data, path)
    return values[0] if values else None
