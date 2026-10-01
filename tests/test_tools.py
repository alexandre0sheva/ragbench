"""The tools framework: built-in tools, the safe runner (`ToolBox`), custom tools, config validation, budgets and evaluator metrics."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Annotated

import pandas as pd
import pytest
from fake_systems import RankedFakeSystem, question, write_experiment
from pydantic import Field

from ragbench.agents import AgentBudget, AgentLoop, Continue
from ragbench.config.schema import ExperimentConfig, ToolsConfig
from ragbench.datasets.schema import Question
from ragbench.documents.schema import Document, RetrievedChunk
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems import SYSTEM_REGISTRY
from ragbench.rag_systems.base import RetrievalResult
from ragbench.rag_systems.trace import Tracer, activate
from ragbench.registry import TOOLS
from ragbench.tools import CallableTool, CorpusView, CustomToolError, ToolBox, ToolContext, ToolResult, resolve_tools
from ragbench.tools.calculator import calculate
from ragbench.tools.corpus import regex_problem
from ragbench.utils.jsonl import read_jsonl

NOW = datetime.fromisoformat("2026-03-15")

DOCS = [
    Document(doc_id="doc_a", path="a.md", title="Pricing", text="# Pricing\nHarborShield costs $200 per month.\nThe marine module is extra.\nContact sales for ID-1234 quotes."),
    Document(doc_id="doc_b", path="b.md", title="Roadmap", text="# Roadmap\nClaimPilot ships in Q3.\nHarborShield 2 follows in Q4."),
    Document(doc_id="doc_c", path="c.md", title="Long", text="x" * 5000),
]


def _ctx(documents=DOCS, **kwargs) -> ToolContext:
    return ToolContext(corpus=CorpusView(documents), now=NOW, **kwargs)


def _box(*refs, **settings) -> ToolBox:
    return ToolBox.from_refs(list(refs), ToolsConfig(**settings))


def _call(tool: str, args: dict, ctx: ToolContext | None = None, box: ToolBox | None = None) -> ToolResult:
    return (box or _box(tool)).call(tool, args, ctx or _ctx())


# --- calculator -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1200 - 950) / 950 * 100", pytest.approx(26.3157894737)),
        ("2 ** 10", 1024),
        ("2 ^ 10", 1024),
        ("17 % 5", 2),
        ("17 // 5", 3),
        ("-(3 + 4) * 2", -14),
        ("round(10 / 3, 2)", 3.33),
        ("round(2.5)", 2),
        ("min(4, 2, 9)", 2),
        ("max([1, 5, 3])", 5),
        ("sum(1, 2, 3.5)", 6.5),
        ("sum([10, 20, 30]) / 3", 20),
        ("abs(-7.5)", 7.5),
        ("$1,200 * 2", 2400),
        ("1,000,000 / 4", 250000),
        ("12 × 3", 36),
        ("(100 ÷ 8) =", 12.5),
        ("2 * pi", pytest.approx(6.283185307)),
        ("10 ** -2", 0.01),
        ("0.1 + 0.2", pytest.approx(0.3)),
    ],
)
def test_calculator_evaluates_plain_arithmetic(expression, expected):
    result = _call("calculator", {"expression": expression})
    assert result.error is None and result.data == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "__import__('os')",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('x=1')",
        "(1).__class__",
        "(1).__class__.__bases__[0].__subclasses__()",
        "'a' * 3",
        "'abc'",
        "[x for x in range(3)]",
        "lambda: 1",
        "x = 5",
        "a + 1",
        "True + 1",
        "None",
        "1 if 1 else 2",
        "1 < 2",
        "~5",
        "5 << 2",
        "5 & 3",
        "abs",
        "abs(1, 2)",
        "abs()",
        "round(1, 2, 3)",
        "sum('abc')",
        "min(key=abs)",
        "f'{1}'",
        "{1: 2}",
        "1 +",
        "",
        "(" * 300 + "1" + ")" * 300,
        "1 + " * 150 + "1",
    ],
)
def test_calculator_rejects_everything_but_arithmetic_cleanly(expression):
    result = _call("calculator", {"expression": expression})
    assert result.error and result.text.startswith("Error:") and result.data is None


@pytest.mark.parametrize("expression", ["9**9**9", "9 ** 1001", "2 ** 100000", "10 ** 5000", "(10 ** 400) ** 30", "7 ** 7 ** 7"])
def test_calculator_caps_huge_results(expression):
    started = time.perf_counter()
    result = _call("calculator", {"expression": expression})
    assert result.error and ("too large" in result.error)
    assert time.perf_counter() - started < 1


@pytest.mark.parametrize("expression", ["1 / 0", "5 % 0", "5 // 0", "0 ** -1", "(-8) ** 0.5", "1e308 * 10"])
def test_calculator_reports_math_errors_as_results(expression):
    assert _call("calculator", {"expression": expression}).error


def test_calculator_big_but_allowed_powers_work():
    assert _call("calculator", {"expression": "2 ** 1000"}).data == 2**1000
    assert _call("calculator", {"expression": "1 ** 99999999"}).data == 1


def test_calculator_rejects_overlong_expressions_and_bad_argument_shapes():
    assert "too long" in _call("calculator", {"expression": "1+" * 300 + "1"}).error
    assert _call("calculator", {}).error.startswith("invalid arguments")
    assert _call("calculator", {"expression": 5}).error.startswith("invalid arguments")
    assert _call("calculator", {"expression": "1", "extra": 1}).data == 1  # unknown keys from a sloppy model are ignored


def test_calculate_function_raises_a_dedicated_error():
    from ragbench.tools.calculator import CalculatorError

    with pytest.raises(CalculatorError):
        calculate("__import__('os')")
    assert calculate("3 * 4") == 12


# --- date_calc ------------------------------------------------------------------------------------------------------------


def test_date_calc_uses_the_frozen_date_for_today():
    assert "2026-03-15" in _call("date_calc", {"operation": "today"}).text
    other = ToolContext(corpus=CorpusView([]), now=datetime.fromisoformat("2030-07-04"))
    assert "2030-07-04" in _call("date_calc", {"operation": "today"}, other).text
    # deterministic: the same call twice gives the same answer
    first = _call("date_calc", {"operation": "diff", "date": "2026-01-01", "other_date": "today"})
    assert first.data == _call("date_calc", {"operation": "diff", "date": "2026-01-01", "other_date": "today"}).data
    assert first.data["days"] == 73


@pytest.mark.parametrize(
    ("args", "days", "calendar"),
    [
        ({"date": "2025-01-01", "other_date": "2025-01-31"}, 30, (0, 0, 30)),
        ({"date": "2025-01-31", "other_date": "2025-03-15"}, 43, (0, 1, 15)),
        ({"date": "2024-02-29", "other_date": "2025-02-28"}, 365, (0, 11, 30)),
        ({"date": "2020-06-15", "other_date": "2025-06-15"}, 1826, (5, 0, 0)),
        ({"date": "2025-03-15", "other_date": "2025-01-31"}, -43, (0, 1, 15)),
        ({"date": "March 5, 2025", "other_date": "5 Apr 2025"}, 31, (0, 1, 0)),
        ({"date": "2025-03-05T10:30:00Z", "other_date": "2025-03-06"}, 1, (0, 0, 1)),
        ({"date": "1st of"[:0] + "March 1st, 2025", "other_date": "03/15/2025"}, 14, (0, 0, 14)),
    ],
)
def test_date_diff(args, days, calendar):
    result = _call("date_calc", {"operation": "diff", **args})
    assert result.error is None
    assert result.data["days"] == days
    assert (result.data["years"], result.data["months"], result.data["remaining_days"]) == calendar


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"date": "2025-01-31", "months": 1}, "2025-02-28"),
        ({"date": "2024-01-31", "months": 1}, "2024-02-29"),
        ({"date": "2025-03-31", "months": -1}, "2025-02-28"),
        ({"date": "2025-12-15", "months": 2}, "2026-02-15"),
        ({"date": "2025-02-28", "years": 1, "days": 1}, "2026-03-01"),
        ({"date": "2025-01-01", "weeks": 2, "days": -1}, "2025-01-14"),
        ({"months": 1}, "2026-04-15"),  # no base date: today
        ({"date": "2025-01-01", "days": -366}, "2024-01-01"),  # 2024 is a leap year
        ({"date": "2025-01-01", "days": -367}, "2023-12-31"),
    ],
)
def test_date_add(args, expected):
    result = _call("date_calc", {"operation": "add", **args})
    assert result.error is None and result.data == expected


def test_date_weekday_and_errors():
    assert _call("date_calc", {"operation": "weekday", "date": "2026-03-15"}).data == "Sunday"
    for bad in [
        {"operation": "diff", "date": "2025-01-01"},
        {"operation": "diff", "date": "not a date", "other_date": "2025-01-01"},
        {"operation": "weekday"},
        {"operation": "add", "date": "2025-13-45"},
        {"operation": "add", "date": "2025-01-01", "years": 10**12},
        {"operation": "add", "date": "9999-12-31", "days": 1},
        {"operation": "frobnicate"},
    ]:
        assert _call("date_calc", bad).error, bad


# --- corpus tools ---------------------------------------------------------------------------------------------------------


def test_list_documents_pages_and_filters():
    listed = _call("list_documents", {})
    assert listed.text.splitlines()[0] == "doc_a | Pricing | 85 chars" or listed.text.startswith("doc_a | Pricing")
    assert [d["doc_id"] for d in listed.data] == ["doc_a", "doc_b", "doc_c"]
    assert [d["doc_id"] for d in _call("list_documents", {"limit": 1, "offset": 1}).data] == ["doc_b"]
    assert "use offset=1" in _call("list_documents", {"limit": 1}).text
    assert [d["doc_id"] for d in _call("list_documents", {"contains": "ROAD"}).data] == ["doc_b"]
    assert "No documents" in _call("list_documents", {"contains": "zzz"}).text
    assert _call("list_documents", {"limit": 0}).error


def test_read_document_windows_and_bounds():
    first = _call("read_document", {"doc_id": "doc_a", "length": 20})
    assert first.data == {"doc_id": "doc_a", "start": 0, "end": 20, "total": len(DOCS[0].text)}
    assert "characters 0-20 of" in first.text and "use start=20" in first.text
    last = _call("read_document", {"doc_id": "doc_a", "start": len(DOCS[0].text) - 5})
    assert last.text.endswith("[end of document]") and last.data["end"] == len(DOCS[0].text)
    assert _call("read_document", {"doc_id": "doc_a", "start": 10_000}).error.startswith("start=10000 is past the end")
    assert _call("read_document", {"doc_id": "doc_a", "length": 0}).error
    assert _call("read_document", {"doc_id": "doc_a", "length": 8001}).error
    assert _call("read_document", {"doc_id": "doc_a", "start": -1}).error
    window = _call("read_document", {"doc_id": "doc_c", "start": 100, "length": 8000})
    assert window.data["end"] == 5000 - 0 and len(window.text) < 5100  # clipped to the document, not to the request
    assert _call("read_document", {"doc_id": "DOC_B"}).data["doc_id"] == "doc_b"  # a unique case-insensitive match is accepted
    missing = _call("read_document", {"doc_id": "doc_x"})
    assert missing.error and "did you mean" in missing.error and "doc_a" in missing.error


def test_corpus_grep_finds_words_with_context_and_line_numbers():
    result = _call("corpus_grep", {"pattern": "harborshield", "context_lines": 1})
    assert [(h["doc_id"], h["line"]) for h in result.data] == [("doc_a", 2), ("doc_b", 3)]
    assert "doc_a:2: HarborShield costs $200 per month." in result.text and "doc_a:1- # Pricing" in result.text
    assert _call("corpus_grep", {"pattern": "harborshield", "case_sensitive": True}).data == []
    assert [h["line"] for h in _call("corpus_grep", {"pattern": "marine extra"}).data] == [3]  # words are ANDed on one line
    assert [h["doc_id"] for h in _call("corpus_grep", {"pattern": "HarborShield", "doc_id": "doc_b"}).data] == ["doc_b"]
    assert _call("corpus_grep", {"pattern": "x", "doc_id": "nope"}).error
    assert _call("corpus_grep", {"pattern": "   "}).error


def test_corpus_grep_caps_the_number_of_results():
    many = [Document(doc_id=f"d{i}", path=f"{i}.md", title=str(i), text="needle\n" * 30) for i in range(10)]
    capped = _call("corpus_grep", {"pattern": "needle", "max_results": 7}, _ctx(many))
    assert len(capped.data) == 7 and "stopped at 7" in capped.text
    box = _box({"name": "corpus_grep", "max_results": 3})
    assert len(box.call("corpus_grep", {"pattern": "needle", "max_results": 50}, _ctx(many)).data) == 3  # the configured cap wins
    assert _call("corpus_grep", {"pattern": "needle", "max_results": 51}).error  # above the hard limit
    huge = _call("corpus_grep", {"pattern": "needle"}, _ctx(many), _box("corpus_grep", max_output_chars=200))
    assert huge.truncated and len(huge.text) <= 200


def test_corpus_grep_regex_mode():
    result = _call("corpus_grep", {"pattern": r"ID-\d{4}", "regex": True})
    assert [h["doc_id"] for h in result.data] == ["doc_a"]
    assert _call("corpus_grep", {"pattern": r"q[34]\b", "regex": True}).data[0]["doc_id"] == "doc_b"
    assert "invalid regular expression" in _call("corpus_grep", {"pattern": "(unclosed", "regex": True}).error
    # without `regex`, metacharacters are plain text
    assert _call("corpus_grep", {"pattern": "(a+)+$"}).data == []


@pytest.mark.parametrize(
    "pattern",
    ["(a+)+$", "(a*)*b", "(a|aa)+c", "(.*)*x", "(.*a){20}", "^(([a-z])+.)+[A-Z]([a-z])+$", "(a+){2,}", "(\\w+\\s?)*$", "(x+x+)+y", "((a+))+", "(a|b|ab)*c"],
)
def test_corpus_grep_refuses_catastrophic_regexes_instantly(pattern):
    # `(a+)+$` against 'aaaa...b' would run for years: the pattern must be refused before any matching happens
    docs = [Document(doc_id="d", path="d.md", title="d", text="a" * 60 + "b")]
    started = time.perf_counter()
    result = _call("corpus_grep", {"pattern": pattern, "regex": True}, _ctx(docs))
    assert time.perf_counter() - started < 1
    assert result.error and ("exponential" in result.error or "supported" in result.error)


@pytest.mark.parametrize(("pattern", "ok"), [("ID-\\d+", True), ("(foo|bar)baz", True), ("(ab)+c", True), ("[a-z]+@[a-z]+\\.com", True), ("a+b+", True), ("(\\d+)-(\\d+)", True), ("[(+*]+", True), ("x{2,5}", True), ("(a|b)?c", True), ("\\(a+\\)+", True), ("(a+)\\1", False), ("a" * 201, False)])
def test_regex_problem_accepts_ordinary_patterns(pattern, ok):
    assert (regex_problem(pattern) is None) is ok


def test_corpus_grep_long_lines_are_only_scanned_to_a_limit():
    docs = [Document(doc_id="d", path="d.md", title="d", text="x" * 5000 + " needle")]
    assert _call("corpus_grep", {"pattern": "needle"}, _ctx(docs)).data == []  # the match is beyond the scanned prefix


def test_lookup_table_matches_rows_and_selects_columns():
    rows = [
        Document(doc_id="prices.csv#1", path="prices.csv", title="prices row 1", text="product: HarborShield\nprice: 200\nunit: month", metadata={"row": 1}),
        Document(doc_id="prices.csv#2", path="prices.csv", title="prices row 2", text="product: ClaimPilot\nprice: 350\nunit: month", metadata={"row": 2}),
        DOCS[0],
    ]
    ctx = _ctx(rows)
    found = _call("lookup_table", {"where": {"Product": " harborshield "}, "select": ["price"]}, ctx)
    assert found.data == [{"price": "200"}] and "prices.csv#1" in found.text
    assert len(_call("lookup_table", {"where": {"unit": "month"}}, ctx).data) == 2
    assert _call("lookup_table", {"where": {"product": "Nope"}}, ctx).data == []
    assert "Columns:" in _call("lookup_table", {"where": {"product": "Nope"}}, ctx).text
    assert _call("lookup_table", {"where": {"unit": "month"}, "source": "other.csv"}, ctx).data == []
    assert "no table rows" in _call("lookup_table", {"where": {"a": "b"}}).error  # a corpus without tables


# --- search ---------------------------------------------------------------------------------------------------------------


def test_search_wraps_the_systems_retriever_and_reports_its_cost():
    calls = []

    def retriever(query: str, top_k: int) -> RetrievalResult:
        calls.append((query, top_k))
        chunks = [RetrievedChunk(chunk_id=f"doc_a::chunk::{i}", doc_id="doc_a", text=f"passage {i}", score=1.0 / (i + 1), rank=i + 1) for i in range(top_k)]
        return RetrievalResult(question=query, chunks=chunks, cost=CostBreakdown(embedding_cost=0.001))

    ctx = _ctx(retrievers={"search": retriever})
    result = _call("search", {"query": "price?", "top_k": 2}, ctx)
    assert calls == [("price?", 2)] and result.cost_usd == pytest.approx(0.001)
    assert result.text.startswith("[doc_a | 0] passage 0") and len(result.data) == 2
    assert _call("search", {"query": "x"}).error == "search is not available for this system"
    assert _call("search", {"query": "x", "top_k": 99}, ctx).error


# --- the runner: timeouts, exceptions, size, tracing ------------------------------------------------------------------------


def _custom(function, name="custom", **options) -> ToolBox:
    return ToolBox([CallableTool(name, function, **options)], ToolsConfig(timeout_s=options.pop("box_timeout", 10)))


def test_a_tool_that_hangs_returns_an_error_instead_of_blocking():
    def hang(seconds: float) -> str:
        """Sleeps."""
        time.sleep(seconds)
        return "done"

    box = ToolBox([CallableTool("hang", hang)], ToolsConfig(timeout_s=0.2))
    started = time.perf_counter()
    result = box.call("hang", {"seconds": 5}, _ctx())
    assert time.perf_counter() - started < 2
    assert result.error and "timed out after 0.2 s" in result.error
    assert box.call("hang", {"seconds": 0}, _ctx()).text == "done"  # the box keeps working afterwards


def test_a_tool_can_set_its_own_timeout():
    def slow() -> str:
        """Slow."""
        time.sleep(0.4)
        return "ok"

    box = ToolBox([CallableTool("slow", slow, timeout_s=2)], ToolsConfig(timeout_s=0.05))
    assert box.call("slow", {}, _ctx()).text == "ok"


def test_tool_exceptions_never_propagate():
    def boom(x: int) -> str:
        """Always fails."""
        raise RuntimeError(f"kaboom {x}")

    result = _custom(boom).call("custom", {"x": 3}, _ctx())
    assert result.error == "custom failed: RuntimeError: kaboom 3" and result.text.startswith("Error:")

    class Broken:
        spec = TOOLS.get("calculator").spec
        side_effects = "none"

        def run(self, args, ctx):
            return "not a ToolResult"

    assert "returned str" in ToolBox([Broken()]).call("calculator", {}, _ctx()).error  # type: ignore[list-item]


def test_unknown_tools_and_malformed_arguments_are_error_results():
    box = _box("calculator", "date_calc")
    unknown = box.call("calculater", {}, _ctx())
    assert unknown.error and "unknown tool 'calculater'" in unknown.error and "calculator, date_calc" in unknown.error
    assert box.call("calculator", '{"expression": "2+2"}', _ctx()).data == 4  # arguments as a JSON string (some providers)
    assert box.call("calculator", "", _ctx()).error.startswith("invalid arguments")
    assert box.call("calculator", "not json", _ctx()).error == "the arguments must be a JSON object"
    assert box.call("calculator", [1, 2], _ctx()).error == "the arguments must be a JSON object"
    assert box.call("calculator", None, _ctx()).error.startswith("invalid arguments")


def test_output_is_capped_and_flagged_truncated():
    box = _box("read_document", max_output_chars=300)
    result = box.call("read_document", {"doc_id": "doc_c", "length": 4000}, _ctx())
    assert result.truncated and len(result.text) == 300 and result.text.endswith("[output truncated]")
    short = box.call("read_document", {"doc_id": "doc_a", "length": 50}, _ctx())
    assert not short.truncated


def test_every_call_is_a_traced_tool_step_with_args_cost_and_errors():
    box = _box("calculator", "corpus_grep")
    tracer = Tracer()
    with activate(tracer):
        box.call("calculator", {"expression": "6 * 7"}, _ctx(), tracer)
        box.call("calculator", {"expression": "__import__('os')"}, _ctx(), tracer)
        box.call("nope", {}, _ctx(), tracer)
        box.call("corpus_grep", {"pattern": "needle"}, _ctx([Document(doc_id="d", path="d", title="d", text="needle " * 2000)]), tracer)
    ok, bad, unknown, big = tracer.steps
    assert [s.kind for s in tracer.steps] == ["tool"] * 4 and [s.name for s in tracer.steps] == ["calculator", "calculator", "nope", "corpus_grep"]
    assert json.loads(ok.input_preview) == {"expression": "6 * 7"} and ok.output_preview == "6 * 7 = 42" and "error" not in ok.metadata
    assert ok.latency_ms > 0 or ok.latency_ms == 0
    assert bad.metadata["error"] and unknown.metadata["error"].startswith("unknown tool")
    assert big.metadata["truncated"] is True and len(big.output_preview) <= 300  # the 14,000-character matching line was cut to the 4,000-character cap
    assert ok.metadata["truncated"] is False
    assert all(step.cost.total_cost == 0 for step in tracer.steps)


def test_tool_cost_lands_in_the_step_as_tool_cost():
    def priced(query: str) -> ToolResult:
        """A paid lookup."""
        return ToolResult(text="found", cost_usd=0.25)

    tracer = Tracer()
    result = _custom(priced).call("custom", {"query": "x"}, _ctx(), tracer)
    assert result.cost_usd == 0.25 and tracer.steps[0].cost.tool_cost == 0.25 and tracer.steps[0].cost.total_cost == 0.25


def test_tool_result_latency_is_measured():
    def nap() -> str:
        """Naps."""
        time.sleep(0.05)
        return "z"

    assert _custom(nap).call("custom", {}, _ctx()).latency_ms >= 40


def test_concurrent_questions_have_their_own_context_state():
    box = _box("calculator")
    seen: dict[int, str] = {}

    def question_worker(index: int) -> None:
        ctx = ToolContext.for_question(CorpusView(DOCS))
        ctx.state["me"] = index
        for _ in range(50):
            assert box.call("calculator", {"expression": f"{index} + 1"}, ctx).data == index + 1
        seen[index] = str(ctx.state["me"])

    threads = [threading.Thread(target=question_worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {i: str(i) for i in range(8)}
    assert ToolContext.for_question(CorpusView([])).state is not ToolContext.for_question(CorpusView([])).state


def test_for_question_uses_the_runs_frozen_date():
    from ragbench.runtime.context import RuntimeContext, activate_runtime

    with activate_runtime(RuntimeContext(tools_now=datetime.fromisoformat("2031-02-03"), tools=ToolsConfig(allow_network=True))):
        ctx = ToolContext.for_question(CorpusView([]))
    assert ctx.today.isoformat() == "2031-02-03" and ctx.allow_network is True
    assert ToolContext.for_question(CorpusView([])).allow_network is False


def test_openai_schemas_describe_every_tool():
    schemas = _box("calculator", "date_calc", "search").schemas()
    assert [s["function"]["name"] for s in schemas] == ["calculator", "date_calc", "search"]
    for schema in schemas:
        assert schema["type"] == "function" and schema["function"]["description"]
        parameters = schema["function"]["parameters"]
        assert parameters["type"] == "object" and parameters["properties"] and "title" not in parameters
    assert _box("calculator").schemas()[0]["function"]["parameters"]["required"] == ["expression"]


# --- custom tools ---------------------------------------------------------------------------------------------------------


def lookup_price(product: str, quantity: int = 1, currency: Annotated[str, Field(description="ISO currency code")] = "USD") -> str:
    """Price of a product.

    Longer explanation that must not reach the model.
    """
    return f"{quantity} x {product} = {200 * quantity} {currency}"


def needs_ctx(term: str, ctx: ToolContext) -> dict:
    """Count documents mentioning a term."""
    return {"count": sum(term.lower() in d.text.lower() for d in ctx.corpus.documents), "today": ctx.today.isoformat()}


def untyped(a, b: int) -> str:
    """No hint on a."""
    return "x"


def variadic(*items: int) -> str:
    """Variadic."""
    return "x"


def test_custom_tool_schema_comes_from_type_hints_and_the_docstring():
    tool = CallableTool("lookup_price", lookup_price)
    assert tool.spec.description == "Price of a product."
    parameters = tool.spec.parameters
    assert parameters["required"] == ["product"]
    assert parameters["properties"]["quantity"] == {"type": "integer", "default": 1}
    assert parameters["properties"]["currency"]["description"] == "ISO currency code"
    assert parameters["properties"]["product"] == {"type": "string"}
    result = ToolBox([tool]).call("lookup_price", {"product": "HarborShield", "quantity": "3"}, _ctx())  # "3" is coerced like a model might send it
    assert result.text == "3 x HarborShield = 600 USD"
    assert ToolBox([tool]).call("lookup_price", {"quantity": 2}, _ctx()).error.startswith("invalid arguments (product")
    assert ToolBox([tool]).call("lookup_price", {"product": "x", "quantity": "many"}, _ctx()).error.startswith("invalid arguments")
    assert tool.spec.openai_schema()["function"]["name"] == "lookup_price"


def test_custom_tools_can_receive_the_context_and_return_structured_data():
    tool = CallableTool("count_docs", needs_ctx, description="Overridden description.")
    assert tool.spec.description == "Overridden description."
    assert "ctx" not in tool.spec.parameters["properties"]
    result = ToolBox([tool]).call("count_docs", {"term": "harborshield"}, _ctx())
    assert result.data == {"count": 2, "today": "2026-03-15"} and json.loads(result.text) == result.data


def test_custom_tool_loading_errors_are_specific():
    with pytest.raises(CustomToolError, match="parameter 'a' needs a type hint"):
        CallableTool("t", untyped)
    with pytest.raises(CustomToolError, match=r"\*args"):
        CallableTool("t", variadic)
    with pytest.raises(CustomToolError, match="1-64 letters"):
        CallableTool("bad name!", lookup_price)


def test_tools_resolve_from_refs_with_options_and_custom_paths():
    tools = resolve_tools(["calculator", {"name": "corpus_grep", "max_results": 5}, {"name": "price", "path": "test_tools:lookup_price"}])
    assert [t.spec.name for t in tools] == ["calculator", "corpus_grep", "price"]
    assert tools[1].options.max_results == 5
    with pytest.raises(ValueError, match="Did you mean 'calculator'"):
        resolve_tools(["calculater"])
    with pytest.raises(ValueError, match="max_resultz"):
        resolve_tools([{"name": "corpus_grep", "max_resultz": 5}])
    with pytest.raises(ValueError, match="listed twice"):
        resolve_tools(["calculator", "calculator"])
    with pytest.raises(ValueError, match="name of a built-in"):
        resolve_tools([{"name": "calculator", "path": "test_tools:lookup_price"}])
    with pytest.raises(ValueError, match="package.module:function"):
        resolve_tools([{"name": "x", "path": "nocolon"}])
    with pytest.raises(ValueError, match="cannot import 'no_such_module_xyz'"):
        resolve_tools([{"name": "x", "path": "no_such_module_xyz:f"}])
    with pytest.raises(ValueError, match="no attribute 'missing'"):
        resolve_tools([{"name": "x", "path": "test_tools:missing"}])
    with pytest.raises(ValueError, match="unknown option"):
        resolve_tools([{"name": "x", "path": "test_tools:lookup_price", "colour": "red"}])


def test_tools_with_side_effects_need_explicit_permission():
    ref = {"name": "fetch", "path": "test_tools:lookup_price", "side_effects": "network"}
    with pytest.raises(ValueError, match="list it under `tools.allow`"):
        resolve_tools([ref], ToolsConfig())
    with pytest.raises(ValueError, match="tools.allow_network"):
        resolve_tools([ref], ToolsConfig(allow=["fetch"]))
    assert resolve_tools([ref], ToolsConfig(allow=["fetch"], allow_network=True))[0].side_effects == "network"
    fs = {"name": "save", "path": "test_tools:lookup_price", "side_effects": "filesystem"}
    assert resolve_tools([fs], ToolsConfig(allow=["save"]))  # filesystem tools need `allow` but not `allow_network`
    with pytest.raises(ValueError, match="side_effects must be one of"):
        resolve_tools([{"name": "x", "path": "test_tools:lookup_price", "side_effects": "launch-missiles"}])
    assert all(tool.side_effects == "none" for tool in resolve_tools(list(TOOLS.names())))  # every built-in is side-effect free


def test_no_network_tool_is_built_in():
    assert not {"http_get", "web_search", "fetch_url"} & set(TOOLS.names())
    assert set(TOOLS.names()) == {"calculator", "date_calc", "list_documents", "read_document", "corpus_grep", "search", "lookup_table"}


# --- configuration --------------------------------------------------------------------------------------------------------


class ToolingFake(RankedFakeSystem):
    """A fake system with no spec (so it takes any `tools:`) that uses its tools on every question."""

    def __init__(self, config, force_mock=False):
        super().__init__(config, force_mock=force_mock)
        self.box = ToolBox.from_refs(config.tools)
        self.corpus = CorpusView([])

    def ingest(self, documents):
        self.corpus = CorpusView(documents)
        return super().ingest(documents)

    def fetch_context(self, question, top_k=None):
        ctx = ToolContext.for_question(self.corpus)
        if "compute" in question and "calculator" in self.box.tools:
            self.box.call("calculator", {"expression": "19 * 21"}, ctx, self.trace)
        if "broken" in question and "calculator" in self.box.tools:
            self.box.call("calculator", {"expression": "__import__('os')"}, ctx, self.trace)
        if "grep" in question and "corpus_grep" in self.box.tools:
            self.box.call("corpus_grep", {"pattern": "topic"}, ctx, self.trace)
        return super().fetch_context(question, top_k)


def _config(systems, **top) -> dict:
    return {
        "run": {"name": "t", "output_dir": "out"},
        "dataset": {"documents_path": "docs", "questions_path": "q.jsonl"},
        "systems": systems,
        **top,
    }


def test_a_system_that_does_not_use_tools_rejects_a_tools_list():
    with pytest.raises(ValueError, match="does not use tools"):
        ExperimentConfig.model_validate(_config([{"type": "bm25", "tools": ["calculator"]}]))
    assert ExperimentConfig.model_validate(_config([{"type": "bm25"}])).systems[0].tools == []


def test_tool_names_are_checked_when_the_config_loads(monkeypatch):
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_tools", ToolingFake)
    ok = ExperimentConfig.model_validate(_config([{"type": "fake_tools", "tools": ["calculator", {"name": "corpus_grep", "max_results": 3}]}]))
    assert len(ok.systems[0].tools) == 2
    with pytest.raises(ValueError, match=r"systems\[0\]\.tools: tool 'calculater'.*Did you mean 'calculator'"):
        ExperimentConfig.model_validate(_config([{"type": "fake_tools", "tools": ["calculater"]}]))
    with pytest.raises(ValueError, match="must be a tool name or a mapping with a `name`"):
        ExperimentConfig.model_validate(_config([{"type": "fake_tools", "tools": [{"path": "x:y"}]}]))
    with pytest.raises(ValueError, match="tools.allow"):
        ExperimentConfig.model_validate(_config([{"type": "fake_tools", "tools": [{"name": "fetch", "path": "test_tools:lookup_price", "side_effects": "network"}]}]))
    allowed = _config([{"type": "fake_tools", "tools": [{"name": "fetch", "path": "test_tools:lookup_price", "side_effects": "network"}]}], tools={"allow": ["fetch"], "allow_network": True})
    assert ExperimentConfig.model_validate(allowed).tools.allow == ["fetch"]


def test_the_tools_section_and_tools_now_are_validated():
    cfg = ExperimentConfig.model_validate(_config([{"type": "bm25"}], tools={"timeout_s": 3, "max_output_chars": 500}, evaluation={"tools_now": "2027-05-06"}))
    assert cfg.tools.timeout_s == 3 and cfg.evaluation.tools_now == "2027-05-06"
    assert ExperimentConfig.model_validate(_config([{"type": "bm25"}])).evaluation.tools_now == "2026-01-01"
    for bad in ({"tools": {"timeout_s": 0}}, {"tools": {"max_output_chars": 10}}, {"tools": {"alow": []}}, {"evaluation": {"tools_now": "next tuesday"}}):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate(_config([{"type": "bm25"}], **bad))


def test_requires_tools_is_validated_against_the_registry():
    assert Question(id="q", question="?", requires_tools=["calculator", "date_calc"]).requires_tools == ["calculator", "date_calc"]
    with pytest.raises(ValueError, match="Did you mean 'calculator'"):
        Question(id="q", question="?", requires_tools=["calculatr"])
    with pytest.raises(ValueError, match="Unknown tool"):
        Question.model_validate({"id": "q", "question": "?", "requires_tools": ["web_search"]})


def test_the_generated_tools_doc_covers_every_builtin():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "tools.md").read_text(encoding="utf-8")
    for name in TOOLS.names():
        assert f"`{name}`" in text


# --- the agent budget -----------------------------------------------------------------------------------------------------


def test_the_agent_loop_stops_at_the_tool_call_budget():
    box = _box("calculator")
    calls = []

    def policy(state):
        calls.append(state.iteration)
        box.call("calculator", {"expression": "1+1"}, _ctx(), state.tracer)
        return Continue("again")

    tracer = Tracer()
    state = AgentLoop(AgentBudget(max_steps=10, max_tool_calls=3), tracer).run(policy, question="q")
    assert calls == [1, 2, 3] and state.tool_calls == 3
    assert state.termination == "budget" and state.metadata()["budget_exhausted"] is True


def test_a_policy_sees_the_tool_budget_before_calling_a_tool():
    state_seen = []

    def policy(state):
        state_seen.append(state.over_budget())
        with state.tracer.step("tool", "calculator"):
            pass
        return Continue("x")

    AgentLoop(AgentBudget(max_steps=5, max_tool_calls=2), Tracer()).run(policy, question="q")
    assert state_seen == [False, False]
    assert AgentBudget(max_tool_calls=0).exhausted(0, 0, 0) is True  # a budget of zero tool calls is already spent
    assert AgentBudget(max_tool_calls=2).exhausted(0, 0, 1) is False
    with pytest.raises(ValueError):
        AgentBudget(max_tool_calls=-1)


# --- evaluator metrics ----------------------------------------------------------------------------------------------------


def test_the_evaluator_aggregates_tool_use(tmp_path, monkeypatch):
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_tools", ToolingFake)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_plain", RankedFakeSystem)
    questions = [
        {**question("q1", "please compute it"), "requires_tools": ["calculator"]},
        {**question("q2", "compute and grep this"), "requires_tools": ["calculator", "corpus_grep"]},
        {**question("q3", "no tools needed here")},
        {**question("q4", "broken request to compute"), "requires_tools": ["calculator"]},
        {**question("q5", "grep only"), "requires_tools": ["calculator"]},
    ]
    systems = [
        {"type": "fake_tools", "name": "with_tools", "retrieval": {"top_k": 3}, "tools": ["calculator", "corpus_grep", "date_calc"]},
        {"type": "fake_tools", "name": "without_tools", "retrieval": {"top_k": 3}},
        {"type": "fake_plain", "name": "plain", "retrieval": {"top_k": 3}},
    ]
    config = write_experiment(tmp_path, questions, systems)
    run_dir = run_benchmark(config, force_mock=True, max_workers=2)

    rows = read_jsonl(run_dir / "per_question_results.jsonl")
    mine = {r["question_id"]: r for r in rows if r["system"] == "with_tools"}
    assert mine["q1"]["requires_tools"] == ["calculator"] and mine["q3"]["requires_tools"] == []
    assert [s["name"] for s in mine["q2"]["steps"] if s["kind"] == "tool"] == ["calculator", "corpus_grep"]
    q1_step = next(s for s in mine["q1"]["steps"] if s["kind"] == "tool")
    assert q1_step["name"] == "calculator" and json.loads(q1_step["input_preview"]) == {"expression": "19 * 21"} and q1_step["output_preview"] == "19 * 21 = 399"

    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    # tool calls: q1 1, q2 2, q3 0, q4 2 (one fine, one an error), q5 1 (grep) -> 6 calls over 5 questions, 1 of them an error
    assert summary.loc["with_tools", "avg_tool_calls"] == pytest.approx(1.2)
    assert summary.loc["with_tools", "tool_error_rate"] == pytest.approx(1 / 6)
    # requires_tools: q1 yes (calculator), q2 yes (both), q4 yes (calculator was called, even though it failed), q5 no -> 3 of 4
    assert summary.loc["with_tools", "required_tool_used_rate"] == pytest.approx(0.75)
    assert pd.isna(summary.loc["without_tools", "avg_tool_calls"])  # offered no tools and called none
    assert pd.isna(summary.loc["plain", ["avg_tool_calls", "tool_error_rate", "required_tool_used_rate"]]).all()

    usage = pd.read_csv(run_dir / "tool_usage.csv").set_index(["system", "tool"])
    assert usage.loc[("with_tools", "calculator"), "calls"] == 4 and usage.loc[("with_tools", "calculator"), "errors"] == 1
    assert usage.loc[("with_tools", "calculator"), "questions_using"] == 3
    assert usage.loc[("with_tools", "corpus_grep"), "calls"] == 2
    assert usage.loc[("with_tools", "date_calc"), "calls"] == 0  # offered but never used still gets a row
    assert pd.isna(usage.loc[("with_tools", "date_calc"), "error_rate"])
    assert "plain" not in usage.index.get_level_values("system")
    assert usage.loc[("with_tools", "calculator"), "error_rate"] == pytest.approx(0.25)


def test_no_tool_usage_file_for_a_run_without_tools(tmp_path, monkeypatch):
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_plain", RankedFakeSystem)
    config = write_experiment(tmp_path, [question("q1", "hello")], [{"type": "fake_plain", "name": "plain", "retrieval": {"top_k": 3}}])
    run_dir = run_benchmark(config, force_mock=True, max_workers=1)
    assert not (run_dir / "tool_usage.csv").exists()
    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    assert "avg_tool_calls" not in summary.columns


def test_the_run_uses_the_configured_frozen_date(tmp_path, monkeypatch):
    seen = []

    class DateFake(ToolingFake):
        def fetch_context(self, question, top_k=None):
            seen.append(ToolContext.for_question(self.corpus).today.isoformat())
            return super().fetch_context(question, top_k)

    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_dates", DateFake)
    config = write_experiment(tmp_path, [question("q1", "hello")], [{"type": "fake_dates", "name": "d", "retrieval": {"top_k": 3}}], evaluation={"tools_now": "2028-02-29"})
    run_benchmark(config, force_mock=True, max_workers=1)
    assert seen == ["2028-02-29"]
