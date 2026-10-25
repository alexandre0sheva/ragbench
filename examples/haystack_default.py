"""Haystack out of the box: the pipeline its RAG tutorial builds, benchmarked as `python_callable`.

    systems:
      - type: python_callable
        name: haystack_default
        retrieval:
          target: examples/haystack_default.py:answer
          ingest_target: examples/haystack_default.py:ingest
          model: gpt-6-luna

Needs `pip install 'ragbench[haystack]'`. The tutorial's defaults: a `DocumentSplitter` (200 words, no overlap), a document embedder into an
`InMemoryDocumentStore`, and a query pipeline of text embedder, `InMemoryEmbeddingRetriever`, `PromptBuilder` and `OpenAIGenerator`. The model and the
embeddings are the built-in systems' (`gpt-6-luna`, `text-embedding-3-small`); the retrieval depth is RAGBench's `top_k`, so the systems are compared at the same depth.

With no key, or without `RAGBENCH_REFERENCE_LIVE=1`, it runs on RAGBench's mock models (see `ragbench.adapters.offline`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from haystack import Document, Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from ragbench.adapters import offline
from ragbench.adapters.haystack import from_pipeline
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL

TEMPLATE = """Given the following information, answer the question.

Context:
{% for document in documents %}{{ document.content }}
{% endfor %}

Question: {{ question }}"""


@component
class OfflineTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, Any]:
        return {"embedding": offline.embed([text])[0]}


@component
class OfflineDocumentEmbedder:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        vectors = offline.embed([document.content or "" for document in documents])
        return {"documents": [replace(document, embedding=vector) for document, vector in zip(documents, vectors, strict=True)]}


@component
class OfflineGenerator:
    @component.output_types(replies=list[str], meta=list[dict[str, Any]])
    def run(self, prompt: str) -> dict[str, Any]:
        text, prompt_tokens, completion_tokens = offline.complete(prompt)
        return {"replies": [text], "meta": [{"model": "ragbench-offline", "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}]}


_pipeline: Pipeline | None = None


def ingest(documents: list[Any]) -> None:
    global _pipeline
    if offline.live():
        from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
        from haystack.components.generators import OpenAIGenerator

        document_embedder: Any = OpenAIDocumentEmbedder(model=DEFAULT_EMBEDDING_MODEL)
        text_embedder: Any = OpenAITextEmbedder(model=DEFAULT_EMBEDDING_MODEL)
        generator: Any = OpenAIGenerator(model=DEFAULT_GENERATOR_MODEL, generation_kwargs={"temperature": 0})
    else:
        document_embedder, text_embedder, generator = OfflineDocumentEmbedder(), OfflineTextEmbedder(), OfflineGenerator()
    store = InMemoryDocumentStore()
    chunks = DocumentSplitter().run([Document(content=doc.text, meta={"file_path": doc.path}) for doc in documents])["documents"]
    store.write_documents(document_embedder.run(chunks)["documents"])

    pipeline = Pipeline()
    pipeline.add_component("embedder", text_embedder)
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(store))
    pipeline.add_component("prompt_builder", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"]))
    pipeline.add_component("llm", generator)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    _pipeline = pipeline


def _inputs(question: str, top_k: int) -> dict[str, Any]:
    return {"embedder": {"text": question}, "retriever": {"top_k": top_k}, "prompt_builder": {"question": question}}


def answer(question: str, top_k: int) -> dict[str, Any]:
    assert _pipeline is not None, "ingest() runs first"
    return from_pipeline(_pipeline, _inputs, model=DEFAULT_GENERATOR_MODEL)(question, top_k)
