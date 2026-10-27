"""The published benchmark numbers are generated from the committed result files, never typed: check that they still match.

Every `docs/benchmarks/<name>.md` is rendered from `scripts/benchmark_templates/<name>.md.j2` and the run folders next to it
(`docs/benchmarks/<name>-run/`, and `<name>-harnesses-run/`, `<name>-models-run/` and `<name>-optimize/` when they exist); the README block between
`<!-- results:<name>:start -->` and `<!-- results:<name>:end -->` is rendered from `<name>.readme.md.j2`. A page or block that
differs from its rendering, or a page under docs/benchmarks/ with no template, fails the check, so no number can drift from the CSVs.

    python scripts/check_results.py            # check (CI)
    python scripts/check_results.py --write    # regenerate the pages and the README block after publishing a run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Path("scripts") / "benchmark_templates"
BENCHMARKS = Path("docs") / "benchmarks"


def _number(value: str) -> Any:
    if value in ("", None):
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


class Run:
    """A published run folder, read for a template: its rows by system, its significance tests and its JSON files."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.rows = read_csv(path / "metrics_summary.csv")
        self.significance = read_csv(path / "significance.csv")
        self.summary = _json(path / "run_summary.json")
        self.manifest = _json(path / "run_manifest.json")
        self.recommendation = _json(path / "recommendation.json")
        self.stats = _json(path / "stats.json")
        self.runtime = read_csv(path / "system_runtime.csv")
        self.hub = _json(path / "hub.json")
        self.patterns = _json(path / "patterns.json")
        self.corrections = _json(path / "corrections.json")
        config = path / "config.yaml"
        self.config = yaml.safe_load(config.read_text(encoding="utf-8")) if config.is_file() else {}

    def row(self, system: str) -> dict[str, Any]:
        for row in self.rows:
            if row["system"] == system:
                return row
        raise KeyError(f"{self.name}: no system {system!r} in metrics_summary.csv")

    def by(self, metric: str, *, reverse: bool = True) -> list[dict[str, Any]]:
        """Rows sorted by `metric` (missing values last), ties by name."""
        present = sorted((row for row in self.rows if row.get(metric) is not None), key=lambda row: row["system"])
        present.sort(key=lambda row: row[metric], reverse=reverse)
        return present + sorted((row for row in self.rows if row.get(metric) is None), key=lambda row: row["system"])

    def sweep(self, base: str, key: str = "generator") -> list[str]:
        """The values a sweep gave `base`, in the order of metrics_summary.csv: `base[generator=X]` rows give X."""
        prefix = f"{base}[{key}="
        return [row["system"][len(prefix) : -1] for row in self.rows if row["system"].startswith(prefix) and row["system"].endswith("]")]

    def test(self, system: str, baseline: str, metric: str = "answer_score") -> dict[str, Any] | None:
        for row in self.significance:
            if row["system"] == system and row["baseline"] == baseline and row["metric"] == metric:
                return row
        return None


def _ranks(values: list[float]) -> list[float]:
    """1-based ranks, ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start : end + 1]:
            ranks[i] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def rank_correlation(rows: list[dict[str, Any]], x: str, y: str) -> float:
    """Spearman's rank correlation of two columns over the rows that have both."""
    pairs = [(row[x], row[y]) for row in rows if row.get(x) is not None and row.get(y) is not None]
    xs, ys = zip(*pairs, strict=True)
    return statistics.correlation(_ranks(list(xs)), _ranks(list(ys)))


def score(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def ci(row: dict[str, Any], metric: str, digits: int = 2) -> str:
    """`4.81 [4.71, 4.90]`: the value and its 95% interval from metrics_summary.csv."""
    lo, hi = row.get(f"{metric}_ci_lo"), row.get(f"{metric}_ci_hi")
    interval = f" [{lo:.{digits}f}, {hi:.{digits}f}]" if lo is not None and hi is not None else ""
    return score(row.get(metric), digits) + interval


def money(value: Any, digits: int = 5) -> str:
    return "—" if value is None else f"${value:.{digits}f}"


def signed(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{value:+.{digits}f}".replace("-", "−")


def ms(value: Any) -> str:
    return "—" if value is None else f"{value:.0f} ms"


def codes(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def extract(key: Any, mapping: Any) -> Any:
    """`names | map("extract", scores)`: each name's value in a mapping."""
    return mapping[key]


FILTERS: dict[str, Callable[..., Any]] = {"score": score, "money": money, "signed": signed, "ms": ms, "codes": codes, "extract": extract}


def context(root: Path, name: str) -> dict[str, Any]:
    benchmarks = root / BENCHMARKS
    folders = {"run": f"{name}-run", "harnesses": f"{name}-harnesses-run", "models": f"{name}-models-run", "optimize": f"{name}-optimize"}
    found: dict[str, Any] = {key: Run(benchmarks / folder) if (benchmarks / folder).is_dir() else None for key, folder in folders.items()}
    optimize = benchmarks / folders["optimize"]
    found["optimization_md"] = (optimize / "optimization.md").read_text(encoding="utf-8") if (optimize / "optimization.md").is_file() else ""
    found["ledger"] = _json(optimize / "ledger.json")
    return {"name": name, "ci": ci, "rank_correlation": rank_correlation, **found}


def _environment(root: Path) -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(root / TEMPLATES), undefined=jinja2.StrictUndefined, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True
    )
    env.filters.update(FILTERS)
    env.tests["startingwith"] = lambda value, prefix: str(value).startswith(prefix)
    env.globals["section"] = _section
    return env


def _section(markdown: str, heading: str) -> str:
    """The body of `## heading` in a Markdown file (up to the next `## `), for quoting a committed report verbatim."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", markdown, flags=re.S | re.M)
    return match.group(1).strip() if match else ""


def _readme_block(name: str) -> re.Pattern[str]:
    return re.compile(rf"(<!-- results:{re.escape(name)}:start -->\n)(.*?)(<!-- results:{re.escape(name)}:end -->)", flags=re.S)


def rendered(root: Path) -> dict[Path, str]:
    """What every generated file must contain: the benchmark pages, and the README with each results block filled in."""
    env = _environment(root)
    out: dict[Path, str] = {}
    readme = root / "README.md"
    readme_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    for template in sorted((root / TEMPLATES).glob("*.md.j2")):
        name = template.name.removesuffix(".md.j2")
        if name.endswith(".readme"):
            name = name.removesuffix(".readme")
            block = env.get_template(template.name).render(**context(root, name))
            pattern = _readme_block(name)
            if not pattern.search(readme_text):
                raise SystemExit(f"README.md has no `<!-- results:{name}:start -->` / `<!-- results:{name}:end -->` block for {template.name}")
            readme_text = pattern.sub(lambda match, block=block: match.group(1) + block + match.group(3), readme_text)
            out[readme] = readme_text
        else:
            out[root / BENCHMARKS / f"{name}.md"] = env.get_template(template.name).render(**context(root, name))
    return out


def problems(root: Path) -> list[str]:
    found = []
    expected = rendered(root)
    for path, text in expected.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text:
            found.append(f"{path.relative_to(root)} does not match the committed result files: run `python scripts/check_results.py --write`")
    for page in sorted((root / BENCHMARKS).glob("*.md")):
        if page not in expected:
            found.append(f"{page.relative_to(root)} has no template in {TEMPLATES}: its numbers would be typed by hand")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="regenerate the pages and README blocks instead of checking them")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.write:
        for path, text in rendered(args.root).items():
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(args.root)}")
        return 0
    found = problems(args.root)
    for problem in found:
        print(problem, file=sys.stderr)
    if not found:
        print("published results match the committed result files")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
