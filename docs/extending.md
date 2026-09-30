# Extending RAGBench

Systems are discovered through a registry and describe themselves, so adding one is: write the class, declare its options, register it, and regenerate the docs.

## 1. Declare the options

Options are a Pydantic model that forbids unknown keys, so a typo in a config fails at load time with a suggestion. Reuse the building blocks in `src/ragbench/rag_systems/options.py` (`TopKOptions`, `VectorOptions`, `FusionOptions`, …) and give every field a `description` — it becomes the option table in [systems.md](systems.md).

```python
from pydantic import Field

from ragbench.rag_systems.options import TopKOptions, VectorOptions


class MyOptions(VectorOptions, TopKOptions):
    my_threshold: float = Field(default=0.3, ge=0, le=1, description="Drop candidates scoring below this.")
```

## 2. Write the class

Create a file under `src/ragbench/rag_systems/` and inherit from `BaseRAGSystem`. Register it with `@SYSTEMS.register(...)` and describe it with a `SystemSpec`:

```python
from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker, build_embedder, build_vector_index
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS


@SYSTEMS.register("my_rag")
class MyRAGSystem(BaseRAGSystem):
    """One-line description of the strategy."""

    spec = SystemSpec(
        type="my_rag",
        title="My RAG",
        summary="One line for the README table",
        best_for="When this strategy wins",
        cost_profile="low",        # free | low | medium | high
        latency_profile="fast",    # fast | medium | slow
        requires_llm=False,        # does retrieval itself call an LLM?
        agentic=False,             # does it run a multi-step loop?
        options=MyOptions,
    )
    options: MyOptions             # validated copy of the config's `retrieval:` section

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.store = build_vector_index(self.embedding_model, self.options, self.name)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        # build whatever indexes you need; report cost on IngestionResult.cost
        ...

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        ...
```

Use the shared builders (`build_chunker`, `build_embedder`, `build_vector_index`, `build_reranker`) rather than constructing chunkers, embedding models or stores yourself — they apply the project defaults and mock mode. Read tunables from `self.options`, never from raw dicts.

The `top_k` passed to `fetch_context` is the retrieval **depth**: return that many ranked chunks (the evaluator scores retrieval on all of them), while only the first `context_k` are given to the generator. Make sure any internal candidate pool is at least `top_k` deep.

### Record steps

`answer_question()` traces every question. Wrap each piece of work in `self.trace.step(kind, name)` so the report can show where time and money go (kinds: `retrieve`, `rerank`, `llm`, `tool`, `embed`, `route`, `grade`, `generate`; the final `generate` step is recorded for you):

```python
with self.trace.step("retrieve", "vector_search", top_k=k) as step:
    step.set_input(question)
    result = self.store.search(question, top_k=k)
    step.set_chunks(result.chunks, result.cost)   # or step.set_llm(llm_result, messages, cost=...) / step.set_cost(...)
```

Step costs must add up to the cost you return. If they do not, `answer_question` appends a visible `untracked` step with the difference, so a gap shows up in the stage columns instead of hiding. A system that records no steps still gets a coarse `fetch_context` step. Outside `answer_question` (for example a direct `fetch_context` call) `self.trace` is a silent no-op.

`BaseRAGSystem.answer_question()` already wires `fetch_context` into the configured LLM, tracks cost, and returns an `AnswerResult`. Override it only when your system needs custom generation (query rewriting, multi-hop).

Keep the system **dataset-agnostic and mock-safe**: it must work on any user-supplied corpus and run without an API key (use the builders and `self.llm`, which fall back to mocks). `HyDERAG` (`hyde_rag.py`) is a compact example of a system that adds an LLM step while staying mock-safe.

## 3. Register the import

Add the import to `src/ragbench/rag_systems/__init__.py` so the module loads (and its decorator runs) with the package. Then regenerate the docs, which derive from your `SystemSpec` and option model:

```bash
python scripts/generate_docs.py
```

CI (`--check`) and `tests/test_docs_fresh.py` fail when the generated files are stale.

## 4. Add a config and run it

Copy one of the `configs/*.yaml` files and change the `systems:` block (the schema is in [configuration.md](configuration.md)), then:

```bash
ragbench run --config configs/my_rag.yaml
ragbench compare --config configs/all.yaml   # head-to-head with the others
```

Add a test under `tests/` that exercises ingestion and retrieval against the demo dataset in mock mode (no API key). See `tests/test_bm25_system.py` for a minimal example.

## Chunkers and rerankers

Both use the same mechanism: decorate the class with `@CHUNKERS.register("name", aliases=(...))` (in `documents/chunkers.py`) or `@RERANKERS.register("name")` (in `models/rerankers.py`). Names are then accepted in configs (`chunker.type`, `retrieval.reranker`) and unknown names are rejected with a did-you-mean suggestion.

## Plugins from other packages

A separate package can contribute components without touching this repository by declaring entry points:

```toml
[project.entry-points."ragbench.systems"]
my_rag = "my_package.rag:MyRAGSystem"

[project.entry-points."ragbench.chunkers"]
my_chunker = "my_package.chunking:MyChunker"

[project.entry-points."ragbench.rerankers"]
my_reranker = "my_package.rerank:MyReranker"
```

Plugins are loaded when `ragbench.rag_systems` is imported. A plugin that fails to import is skipped with a warning, and a name that is already taken is ignored. Give a plugin system a `spec` to get config validation and documentation; without one its `retrieval:` section is accepted as-is.
