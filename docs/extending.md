# Adding a new RAG system

RAGBench is designed to compare retrieval architectures, so adding a new one is a five-step process.

## 1. Create the class

Create a file under `src/ragbench/rag_systems/` (e.g. `my_rag.py`) and inherit from `BaseRAGSystem`:

```python
from ragbench.rag_systems.base import (
    BaseRAGSystem,
    IngestionResult,
    RetrievalResult,
)
from ragbench.documents.schema import Document


class MyRAGSystem(BaseRAGSystem):
    """One-line description of the strategy."""

    def ingest(self, documents: list[Document]) -> IngestionResult:
        # build whatever indexes you need; track cost via self.cost_tracker
        ...

    def fetch_context(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        # return a RetrievalResult with retrieved chunks + retrieved doc ids
        ...
```

## 2. Reuse `answer_question()`

`BaseRAGSystem.answer_question()` already wires `fetch_context` into the configured LLM, tracks cost, and returns an `AnswerResult`. Override it only when your system needs custom generation (see `LLMHeavyRAGSystem` for an example with query rewriting).

## 3. Register the class

Add an import + entry in `src/ragbench/rag_systems/__init__.py` so the config loader can resolve it by name:

```python
from .my_rag import MyRAGSystem

REGISTRY = {
    ...,
    "my_rag": MyRAGSystem,
}
```

## 4. Add a config

Create `configs/my_rag.yaml` by copying one of the existing ones and changing the `systems:` block. The config schema is documented in [configuration.md](configuration.md).

## 5. Run it

```bash
ragbench run --config configs/my_rag.yaml
ragbench compare --config configs/all.yaml   # head-to-head with the others
```

If you intend to contribute the system back upstream, add a test under `tests/` that exercises ingestion and retrieval against the demo dataset in mock mode (no API key required). See `tests/test_bm25_system.py` for a minimal example.
