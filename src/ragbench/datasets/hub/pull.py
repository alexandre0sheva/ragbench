"""`ragbench datasets pull`: download (with consent), verify, convert, optionally take a stratified subset, and write a dataset folder."""

from __future__ import annotations

import json
import shutil
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ragbench.datasets.hub.convert import Converted
from ragbench.datasets.hub.download import cache_root, fetch_files, missing_files
from ragbench.datasets.hub.readers import READERS
from ragbench.datasets.hub.registry import HubDataset, PullError
from ragbench.datasets.hub.subset import SubsetError, stratified_subset
from ragbench.errors import CommandFailed
from ragbench.utils.jsonl import write_jsonl

SOURCE_NAME, META_NAME = "SOURCE.md", "hub.json"


class Declined(CommandFailed):
    """The user said no to the download."""

    def __init__(self) -> None:
        super().__init__("Cancelled. Nothing was downloaded.")


@dataclass
class PullResult:
    output: Path
    name: str
    n_questions: int
    n_documents: int
    n_qrels: int
    downloaded_bytes: int
    subset: dict[str, int] | None
    warnings: list[str] = field(default_factory=list)


def human_size(size: int) -> str:
    if size >= 1 << 20:
        return f"{size / (1 << 20):.1f} MB"
    return f"{max(1, round(size / 1024))} KB"


def _consent_message(entry: HubDataset, missing: list) -> str:
    hosts = sorted({urllib.parse.urlparse(item.url).netloc for item in missing})
    files = ", ".join(f"{item.name} ({human_size(item.size)})" for item in missing)
    return f"Download {human_size(sum(item.size for item in missing))} from {', '.join(hosts)}: {files}? Licence of {entry.title}: {entry.license}."


def _prepare_output(output: Path, force: bool) -> None:
    existing = [name for name in ("docs", "questions.jsonl", "qrels.jsonl", META_NAME) if (output / name).exists()]
    if existing and not force:
        raise PullError(f"{output} already holds {', '.join(existing)}. Use --force to replace an earlier pull, or pick another --output.")
    if force and (output / "docs").exists():
        shutil.rmtree(output / "docs")  # an earlier, larger pull would leave documents the new questions never use


def _keep_documents(converted: Converted, questions: list[dict], shared_corpus: bool) -> list[str]:
    if shared_corpus:
        return list(converted.documents)
    wanted = {doc_id for question in questions for doc_id in converted.contexts[question["id"]]}
    return [doc_id for doc_id in converted.documents if doc_id in wanted]


def _source_text(entry: HubDataset, subset: dict[str, int] | None, counts: tuple[int, int]) -> str:
    lines = [
        f"# {entry.title}",
        "",
        entry.description,
        "",
        f"- Source: {entry.source_url}",
        f"- Revision: `{entry.revision}`",
        f"- License: [{entry.license}]({entry.license_url})",
        f"- Questions: {counts[0]}, documents: {counts[1]}" + (f" (stratified subset of {subset['questions']}, seed {subset['seed']})" if subset else ""),
        "",
        "## Citation",
        "",
        entry.citation,
    ]
    if entry.notes:
        lines += ["", "## Terms and caveats", "", entry.notes]
    lines += ["", "Pulled with `ragbench datasets pull`; the files were checked against the checksums pinned in RAGBench.", ""]
    return "\n".join(lines)


def pull(
    entry: HubDataset,
    output: Path,
    *,
    subset: int | None = None,
    seed: int = 42,
    force: bool = False,
    cache_dir: Path | None = None,
    confirm: Callable[[str], bool],
    on_download: Callable[[str], None] | None = None,
) -> PullResult:
    """Pull `entry` into `output`. `confirm(message)` is asked before anything is downloaded and must return True to go on."""
    cache = cache_dir or cache_root()
    _prepare_output(output, force)
    missing = missing_files(entry, cache)
    if missing and not confirm(_consent_message(entry, missing)):
        raise Declined()
    paths = fetch_files(entry, cache, on_file=(lambda item: on_download(item.name)) if on_download else None)
    converted = READERS[entry.reader](entry, paths)
    questions = converted.questions
    chosen: dict[str, int] | None = None
    if subset is not None:
        try:
            questions = stratified_subset(questions, subset, seed=seed)
        except SubsetError as exc:
            raise PullError(str(exc)) from exc
        chosen = {"questions": len(questions), "seed": seed} if len(questions) < len(converted.questions) else None
    kept_ids = {question["id"] for question in questions}
    doc_ids = _keep_documents(converted, questions, entry.shared_corpus)
    qrels = [row for row in converted.qrels if row["query_id"] in kept_ids]

    docs_dir = output / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc_id in doc_ids:
        document = converted.documents[doc_id]
        (docs_dir / f"{doc_id}.md").write_text(f"# {document.title}\n\n{document.text}\n" if document.title else f"{document.text}\n", encoding="utf-8")
    write_jsonl(output / "questions.jsonl", questions)
    write_jsonl(output / "qrels.jsonl", qrels)
    (output / SOURCE_NAME).write_text(_source_text(entry, chosen, (len(questions), len(doc_ids))), encoding="utf-8")
    meta = {
        "dataset": entry.name,
        "revision": entry.revision,
        "license": entry.license,
        "subset": chosen,
        "files": [{"name": item.name, "url": item.url, "size": item.size, "sha256": item.sha256, "git_sha1": item.git_sha1} for item in entry.files],
    }
    (output / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return PullResult(output, entry.name, len(questions), len(doc_ids), len(qrels), sum(item.size for item in missing), chosen, converted.warnings)
