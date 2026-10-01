from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

DATASET_FILES = ("questions.jsonl", "qrels.jsonl")


def demo_source_dir() -> Path:
    """Directory holding the bundled demo dataset (`docs/`, `questions.jsonl`, `qrels.jsonl`).

    An installed wheel carries it as package data (`ragbench/_data/demo`, copied from `data/demo` at build time by `setup.py`);
    a source checkout reads `data/demo` directly, which stays the single source of truth.
    """
    packaged = Path(str(resources.files("ragbench") / "_data" / "demo"))
    if packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[3] / "data" / "demo"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("The bundled demo dataset is missing from this installation (expected ragbench/_data/demo or data/demo).")


def write_demo_dataset(base_dir: Path = Path("data/demo"), overwrite: bool = False) -> dict[str, int]:
    """Write the bundled demo dataset under `base_dir`, or verify it when it is already there.

    Missing files are written. A file that exists and differs from the bundled copy is reported in `modified` and left
    untouched unless `overwrite` is true, so edits to a local copy are never lost silently.
    """
    source = demo_source_dir()
    files = sorted((source / "docs").glob("*.md")) + [source / name for name in DATASET_FILES]
    written = modified = 0
    for path in files:
        target = base_dir / path.relative_to(source)
        if target.exists():
            if target.read_bytes() == path.read_bytes():
                continue
            if not overwrite:
                modified += 1
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        written += 1
    return {
        "documents": len(files) - len(DATASET_FILES),
        "questions": _line_count(source / "questions.jsonl"),
        "qrels": _line_count(source / "qrels.jsonl"),
        "written": written,
        "modified": modified,
    }


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
