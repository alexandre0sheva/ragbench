"""Tool types: what a tool is, what it returns, and the per-question context it runs in.

A tool is a small, deterministic, side-effect-free (unless it says otherwise) function an agent can call: it takes a JSON
object of arguments and returns text for the model to read. Tools never raise to their caller; a failure is a `ToolResult` with `error` set.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from ragbench.documents.schema import Document

if TYPE_CHECKING:
    from ragbench.rag_systems.base import RetrievalResult

SideEffects = Literal["none", "network", "filesystem"]
DEFAULT_TOOLS_NOW = "2026-01-01"
# A retriever a tool may call: `retriever(query, top_k)`. It must not record trace steps itself; the tool call is the step.
Retriever = Callable[[str, int], "RetrievalResult"]


class ToolSpec(BaseModel):
    """What the model is told about a tool; `parameters` is a JSON schema of the arguments object."""

    name: str
    description: str
    parameters: dict[str, Any]

    def openai_schema(self) -> dict[str, Any]:
        """The OpenAI function-calling form `LLM.generate(tools=...)` takes."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolResult(BaseModel):
    text: str = ""  # what the model reads
    data: Any = None  # structured form of the same answer, for code and tests
    error: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    truncated: bool = False  # `text` was cut to `tools.max_output_chars`

    @classmethod
    def ok(cls, text: str, data: Any = None, cost_usd: float = 0.0) -> ToolResult:
        return cls(text=text, data=data, cost_usd=cost_usd)

    @classmethod
    def fail(cls, message: str) -> ToolResult:
        return cls(text=f"Error: {message}", error=message)


class CorpusView:
    """Read-only view of the corpus for tools. Shared by every question, so it must never be mutated."""

    def __init__(self, documents: Sequence[Document]):
        self.documents: list[Document] = list(documents)
        self._by_id = {document.doc_id: document for document in self.documents}

    def get(self, doc_id: str) -> Document | None:
        """The document with this id; failing that, the only one whose id matches ignoring case."""
        if doc_id in self._by_id:
            return self._by_id[doc_id]
        folded = [document for document in self.documents if document.doc_id.lower() == doc_id.strip().lower()]
        return folded[0] if len(folded) == 1 else None

    def suggest(self, doc_id: str) -> list[str]:
        return difflib.get_close_matches(doc_id, list(self._by_id), n=3, cutoff=0.5)


@dataclass
class ToolContext:
    """Everything a tool call may use. Build one per question: `state` is that question's private scratch space,
    so concurrent questions never share it (`corpus` and `retrievers` are read-only and shared)."""

    corpus: CorpusView
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    now: datetime = field(default_factory=lambda: datetime.fromisoformat(DEFAULT_TOOLS_NOW))  # frozen for the run, so date tools are deterministic
    allow_network: bool = False
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def today(self) -> date:
        return self.now.date()

    @classmethod
    def for_question(cls, corpus: CorpusView, retrievers: dict[str, Retriever] | None = None) -> ToolContext:
        """A fresh context for one question, with the run's frozen date and network permission (`evaluation.tools_now`, `tools.allow_network`)."""
        from ragbench.runtime.context import current_runtime

        runtime = current_runtime()
        return cls(corpus=corpus, retrievers=dict(retrievers or {}), now=runtime.tools_now, allow_network=runtime.tools.allow_network)


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    @property
    def side_effects(self) -> SideEffects: ...

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


class NoOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseTool:
    """Base for built-in tools: declare `Args` (a Pydantic model; its JSON schema is the tool's parameters), `spec`, and `_run`.

    `Options` are the knobs a config can set on the tool (`tools: [{name: corpus_grep, max_results: 10}]`).
    """

    spec: ClassVar[ToolSpec]
    side_effects: ClassVar[SideEffects] = "none"
    Options: ClassVar[type[BaseModel]] = NoOptions
    Args: ClassVar[type[BaseModel]]
    timeout_s: float | None = None  # None: use `tools.timeout_s`

    def __init__(self, **options: Any):
        self.options = self.Options.model_validate(options)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            parsed = self.Args.model_validate(args)
        except ValidationError as exc:
            return ToolResult.fail(describe_validation_error(exc))
        return self._run(parsed, ctx)

    def _run(self, args: Any, ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


def describe_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"])
        parts.append(f"{where}: {error['msg']}" if where else error["msg"])
    return "invalid arguments (" + "; ".join(parts) + ")"


def make_spec(name: str, description: str, args: type[BaseModel]) -> ToolSpec:
    """A `ToolSpec` whose parameters are the JSON schema of `args`, without Pydantic's `title` noise."""

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: clean(value) for key, value in node.items() if not (key == "title" and isinstance(value, str))}
        if isinstance(node, list):
            return [clean(item) for item in node]
        return node

    return ToolSpec(name=name, description=description, parameters=clean(args.model_json_schema()))
