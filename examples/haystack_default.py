"""Haystack out of the box: the pipeline its RAG tutorial builds, benchmarked as `python_callable`.

    systems:
      - type: python_callable
        name: haystack_default
        retrieval:
          target: examples/haystack_default.py:answer
          ingest_target: examples/haystack_default.py:ingest
          model: gpt-6-luna

Needs `pip install 'ragbench[haystack]'`. The tutorial's defaults: a `DocumentSplitter` (200 words, no overlap), a document embedder into an
`InMemoryDocumentStore`, and a query pipeline of text embedder, `InMemoryEmbeddingRetriever`, `ChatPromptBuilder` and `OpenAIChatGenerator` (Haystack 3
dropped the plain `OpenAIGenerator`). The model and the embeddings are the built-in systems' (`gpt-6-luna`, `text-embedding-3-small`); the retrieval depth is RAGBench's `top_k`, so the systems are compared at the same depth.

With no key, or without `RAGBENCH_REFERENCE_LIVE=1`, it runs on RAGBench's mock models (see `ragbench.adapters.offline`). Live, the generator is Haystack's own
`OpenAIChatGenerator`; the embedders call RAGBench's client (the same `text-embedding-3-small` vectors), so indexing a large corpus honours the run's
`limits.tokens_per_minute`, retries and embedding cache.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from ragbench.adapters import offline
from ragbench.adapters.haystack import from_pipeline
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL
from ragbench.models.embeddings import create_embedding_model

TEMPLATE = """Given the following information, answer the question.

Context:
{% for document in documents %}{{ document.content }}
{% endfor %}

Question: {{ question }}"""
PROMPT = [ChatMessage.from_user(TEMPLATE)]


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
class RagbenchTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, Any]:
        return {"embedding": _ragbench_embed([text])[0]}


@component
class RagbenchDocumentEmbedder:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        vectors = _ragbench_embed([document.content or "" for document in documents])
        return {"documents": [replace(document, embedding=vector) for document, vector in zip(documents, vectors, strict=True)]}


def _ragbench_embed(texts: list[str]) -> list[list[float]]:
    """`text-embedding-3-small` through RAGBench's client (throttled, retried and cached like the built-in systems' embeddings)."""
    return [[float(x) for x in row] for row in create_embedding_model(DEFAULT_EMBEDDING_MODEL).embed_texts(texts).vectors]


@component
class OfflineChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        text, prompt_tokens, completion_tokens = offline.complete("\n\n".join(message.text or "" for message in messages))
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        return {"replies": [ChatMessage.from_assistant(text, meta={"model": "ragbench-offline", "usage": usage})]}


_pipeline: Pipeline | None = None


def ingest(documents: list[Any]) -> None:
    global _pipeline
    if offline.live():
        from haystack.components.generators.chat import OpenAIChatGenerator

        document_embedder: Any = RagbenchDocumentEmbedder()
        text_embedder: Any = RagbenchTextEmbedder()
        # No `temperature: 0`: GPT-6 models accept only their default and reject 0 (RAGBench's own client drops it the same way).
        generator: Any = OpenAIChatGenerator(model=DEFAULT_GENERATOR_MODEL)
    else:
        document_embedder, text_embedder, generator = OfflineDocumentEmbedder(), OfflineTextEmbedder(), OfflineChatGenerator()
    store = InMemoryDocumentStore()
    chunks = DocumentSplitter().run([Document(content=doc.text, meta={"file_path": doc.path}) for doc in documents])["documents"]
    store.write_documents(document_embedder.run(chunks)["documents"])

    pipeline = Pipeline()
    pipeline.add_component("embedder", text_embedder)
    pipeline.add_component("retriever", InMemoryEmbeddingRetriever(store))
    pipeline.add_component("prompt_builder", ChatPromptBuilder(template=PROMPT, required_variables=["documents", "question"]))
    pipeline.add_component("llm", generator)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    _pipeline = pipeline


def _inputs(question: str, top_k: int) -> dict[str, Any]:
    return {"embedder": {"text": question}, "retriever": {"top_k": top_k}, "prompt_builder": {"question": question}}


def answer(question: str, top_k: int) -> dict[str, Any]:
    assert _pipeline is not None, "ingest() runs first"
    return from_pipeline(_pipeline, _inputs, model=DEFAULT_GENERATOR_MODEL)(question, top_k)
