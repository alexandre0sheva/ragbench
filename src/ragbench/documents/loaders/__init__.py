"""Document loading. Importing this package registers every built-in loader in `LOADERS` (one per file extension).

A loader is `loader(path, context) -> list[Document]`; one file may yield several documents (one per CSV row). Formats:
text (`.txt .md .markdown .rst`), HTML, PDF (`ragbench[pdf]`), Word (`ragbench[docx]`), and tables / records
(`.csv .tsv .json .jsonl`). See docs/dataset-format.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ragbench.config.schema import DatasetConfig, TabularConfig
from ragbench.documents.loaders import docx as _docx  # noqa: F401  (importing each module registers its loaders)
from ragbench.documents.loaders import html as _html  # noqa: F401
from ragbench.documents.loaders import pdf as _pdf  # noqa: F401
from ragbench.documents.loaders import tabular as _tabular  # noqa: F401
from ragbench.documents.loaders import text as _text  # noqa: F401
from ragbench.documents.loaders.registry import DocumentLoadError, IgnoreRules, LoadContext, matches_any
from ragbench.documents.schema import Document
from ragbench.models.errors import MissingExtraError
from ragbench.registry import LOADERS

LOADERS.load_entry_points("ragbench.loaders")

IGNORE_FILE = ".ragbenchignore"


def supported_extensions() -> set[str]:
    return set(LOADERS.all_names())


def _candidate_files(root: Path, include: list[str] | None, exclude: list[str] | None) -> list[Path]:
    extensions = supported_extensions()
    rules = IgnoreRules()
    ignore_file = root / IGNORE_FILE
    if ignore_file.is_file():
        rules = IgnoreRules(ignore_file.read_text(encoding="utf-8", errors="replace").splitlines())
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(root).as_posix()
        if rules.ignores(relative) or (include and not matches_any(relative, include)) or (exclude and matches_any(relative, exclude)):
            continue
        files.append(path)
    return files


def load_documents(
    path: Path,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_file_mb: float = 25,
    on_error: Literal["raise", "skip"] = "raise",
    tabular: TabularConfig | None = None,
    warnings: list[str] | None = None,
) -> list[Document]:
    """Load every supported document under `path` (a folder, or one file).

    Files matching `.ragbenchignore` (in the folder) or `exclude`, or not matching `include`, are left out. A file that cannot
    be loaded raises `DocumentLoadError` (`on_error="raise"`) or is skipped with a message appended to `warnings`
    (`"skip"`); a missing optional dependency always raises, since no file is at fault. Decoding problems are reported
    through `warnings` without skipping the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Documents path not found: {path}")
    sink = warnings if warnings is not None else []
    if path.is_file():
        files, root = [path], path.parent
    else:
        files, root = _candidate_files(path, include, exclude), path
    context = LoadContext(root=root, tabular=tabular or TabularConfig(), warn=sink.append)
    documents: list[Document] = []
    for file_path in files:
        try:
            size_mb = file_path.stat().st_size / 1_000_000
            if size_mb > max_file_mb:
                raise DocumentLoadError(f"{file_path.name}: {size_mb:.1f} MB exceeds dataset.max_file_mb ({max_file_mb:g}); raise the limit or exclude the file")
            documents.extend(LOADERS.get(file_path.suffix.lower())(file_path, context))
        except MissingExtraError:
            raise
        except Exception as exc:  # noqa: BLE001  (any parser failure means "this file is bad")
            message = str(exc) if isinstance(exc, DocumentLoadError) else f"{file_path.name}: could not be loaded ({type(exc).__name__}: {exc})"
            if on_error == "raise":
                raise DocumentLoadError(message) from exc
            sink.append(f"skipped {message}")
    if not documents:
        raise ValueError(f"No supported documents found under {path}. Supported: {sorted(supported_extensions())}")
    return documents


def load_dataset_documents(dataset: DatasetConfig, warnings: list[str] | None = None) -> list[Document]:
    """`load_documents` with the options of a config's `dataset:` section."""
    return load_documents(
        dataset.documents_path,
        include=dataset.include,
        exclude=dataset.exclude,
        max_file_mb=dataset.max_file_mb,
        on_error=dataset.on_error,
        tabular=dataset.tabular,
        warnings=warnings,
    )


__all__ = ["DocumentLoadError", "load_dataset_documents", "load_documents", "supported_extensions"]
