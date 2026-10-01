"""Audit the documentation for the things that go stale or contradict each other.

    python scripts/audit_docs.py            # print findings, exit 1 if there are any (also run by tests/test_docs_audit.py)

Each piece of knowledge has one owner (see the table in CONTRIBUTING.md); everything else links to it. The checks:

* **links**: every relative Markdown link points at a file that exists, and every `#anchor` at a heading that exists;
* **option tables**: a table with `Option` and `Default` columns lives only where options are owned (`configuration.md`, the generated `systems.md`, `tools.md`);
* **system descriptions**: a table row that describes a system (`` | `name` | description | ...``) lives only in the generated `systems.md` and the README's generated block;
* **metric definitions**: a sentence that defines a metric ("MRR is the ...") lives only in `methodology.md`;
* **counts**: no document states how many systems, chunkers or tools exist (the number changes; the generated tables are the list; small numbers in
  comparisons such as "the two systems" are fine);
* **removed models**: nothing outside `CHANGELOG.md` names a model the price table no longer has.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTION_OWNERS = {"docs/configuration.md", "docs/systems.md", "docs/tools.md", "docs/cli.md"}
SYSTEM_OWNERS = {"docs/systems.md"}
METRIC_OWNER = "docs/methodology.md"
GENERATED_README = re.compile(r"<!-- systems:start -->.*?<!-- systems:end -->", re.S)
REMOVED_MODELS = re.compile(r"\b(gpt-5\.4[\w.-]*|gpt-4o[\w.-]*|gpt-4\.1[\w.-]*|gpt-3\.5[\w.-]*)\b")
COUNTS = re.compile(r"\b(?:all |the )?(?:six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty) (?:RAG |built-in |registered |bundled |original )?(?:systems|chunkers|tools|rerankers)\b", re.I)
METRICS = (
    r"Recall(?:@\w+)?", r"Precision(?:@\w+)?", r"MRR(?:@\w+)?", r"nDCG(?:@\w+)?", r"Hit(?:@\w+| rate)", r"faithfulness", r"correctness", r"completeness", r"citation quality",
    r"token F1", r"exact match", r"keyword recall", r"context recall", r"context precision", r"answer score", r"cost per correct",
)
DEFINES = re.compile(r"\b(?:" + "|".join(METRICS) + r")\b`?\**\s*(?:\([^)]*\)\s*)?(?:is|are|measures|means|counts|=|:)\s+(?:the|a|an|how|whether|share|fraction|one|\d)", re.I)


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.text}"


def documents() -> list[Path]:
    return sorted([ROOT / "README.md", ROOT / "CONTRIBUTING.md", *(ROOT / "docs").glob("*.md")])


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:  # a file outside the repository (the tests plant violations in temporary ones)
        return path.name


def _prose(text: str) -> list[tuple[int, str]]:
    """The lines of `text` outside fenced code blocks and generated-marker blocks, with their line numbers."""
    out: list[tuple[int, str]] = []
    fenced = False
    generated = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if "<!-- systems:start -->" in line or "<!-- tools:start -->" in line:
            generated = True
        if not fenced and not generated:
            out.append((number, line))
        if "<!-- systems:end -->" in line or "<!-- tools:end -->" in line:
            generated = False
    return out


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens (backticks and links are unwrapped first)."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    found: dict[str, int] = {}
    result: set[str] = set()
    for _, line in _prose(path.read_text(encoding="utf-8")):
        match = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            base = slug(match.group(1))
            count = found.get(base, 0)
            found[base] = count + 1
            result.add(base if count == 0 else f"{base}-{count}")
    return result


def check_links(path: Path) -> list[Finding]:
    findings = []
    own = _rel(path)
    for number, line in _prose(path.read_text(encoding="utf-8")):
        for target in re.findall(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)", line):
            if re.match(r"[a-z]+:", target) or target.startswith("mailto:"):
                continue
            file_part, _, anchor = target.partition("#")
            destination = path if not file_part else (path.parent / file_part).resolve()
            if file_part and not destination.exists():
                findings.append(Finding("link", own, number, f"{target} does not exist"))
            elif anchor and destination.suffix == ".md" and anchor not in anchors(destination):
                findings.append(Finding("anchor", own, number, f"{target}: no heading '{anchor}' in {destination.name}"))
    return findings


def check_option_tables(path: Path) -> list[Finding]:
    if _rel(path) in OPTION_OWNERS:
        return []
    findings = []
    for number, line in _prose(path.read_text(encoding="utf-8")):
        cells = [cell.strip().lower() for cell in line.strip().strip("|").split("|")] if line.strip().startswith("|") else []
        if any(c in ("option", "options") for c in cells) and any(c == "default" for c in cells):
            findings.append(Finding("option-table", _rel(path), number, "an option table belongs in docs/configuration.md (or docs/systems.md); link to it instead"))
    return findings


def check_system_rows(path: Path, system_names: set[str]) -> list[Finding]:
    if _rel(path) in SYSTEM_OWNERS:
        return []
    findings = []
    text = path.read_text(encoding="utf-8")
    if path.name == "README.md":
        text = GENERATED_README.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    for number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"\|\s*\[?`(\w+)`\]?(?:\([^)]*\))?\s*\|(.+)\|", line)
        if match and match.group(1) in system_names and len(match.group(2).split("|")) >= 2:
            findings.append(Finding("system-row", _rel(path), number, f"`{match.group(1)}` is described here; docs/systems.md owns system descriptions"))
    return findings


def check_metric_definitions(path: Path) -> list[Finding]:
    if _rel(path) in (METRIC_OWNER, "docs/cli.md"):
        return []
    findings = []
    for number, line in _prose(path.read_text(encoding="utf-8")):
        if line.lstrip().startswith("|"):
            continue
        match = DEFINES.search(line)
        if match:
            findings.append(Finding("metric-definition", _rel(path), number, f"defines a metric ('{match.group(0)}…'); docs/methodology.md owns metric definitions"))
    return findings


def check_counts(path: Path) -> list[Finding]:
    if _rel(path) in ("docs/cli.md",):
        return []
    findings = []
    for number, line in _prose(path.read_text(encoding="utf-8")):
        match = COUNTS.search(line)
        if match:
            findings.append(Finding("count", _rel(path), number, f"states a count ('{match.group(0)}'); say nothing about how many, the generated tables are the list"))
    return findings


def check_removed_models(path: Path) -> list[Finding]:
    findings = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = REMOVED_MODELS.search(line)
        if match:
            findings.append(Finding("removed-model", _rel(path), number, f"names {match.group(0)}, which is not in the price table"))
    return findings


def audit() -> list[Finding]:
    sys.path.insert(0, str(ROOT / "src"))
    import ragbench.rag_systems  # noqa: F401  (registers the systems)
    from ragbench.registry import SYSTEMS

    names = set(SYSTEMS.names())
    findings: list[Finding] = []
    for path in documents():
        findings += check_links(path)
        findings += check_option_tables(path)
        findings += check_system_rows(path, names)
        findings += check_metric_definitions(path)
        findings += check_counts(path)
        findings += check_removed_models(path)
    return findings


def main() -> int:
    findings = audit()
    for finding in findings:
        print(finding)
    print(f"{len(findings)} finding(s) in {len(documents())} documents")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
