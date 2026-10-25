"""LlamaIndex out of the box: the pipeline its starter tutorial builds, benchmarked as `python_callable`.

    systems:
      - type: python_callable
        name: llamaindex_default
        retrieval:
          target: examples/llamaindex_default.py:answer
          ingest_target: examples/llamaindex_default.py:ingest
          model: gpt-6-luna

Needs `pip install 'ragbench[llamaindex]'`. The starter's defaults: `VectorStoreIndex.from_documents` (a `SentenceSplitter` of 1024 tokens with 200 overlap, an
in-memory vector store) and `index.as_query_engine()` with LlamaIndex's own question-answering and refine prompts. The model and the embeddings are the built-in
systems' (`gpt-6-luna`, `text-embedding-3-small`); the retrieval depth is RAGBench's `top_k` (LlamaIndex's own default is 2), so the systems are compared at the same
depth. The file path stays out of the text LlamaIndex embeds and shows the model (its readers would add it), so no system reads more than the document says.

With no key, or without `RAGBENCH_REFERENCE_LIVE=1`, it runs on RAGBench's mock models (see `ragbench.adapters.offline`). Live, LlamaIndex talks to OpenAI through
RAGBench's own client, wrapped in LlamaIndex's `CustomLLM` (a chat model, like its OpenAI integration) and `BaseEmbedding`: `llama-index-llms-openai` requires
`openai<3`, so it cannot be installed next to RAGBench. The pipeline, its splitter and its prompts are LlamaIndex's either way.
"""

from __future__ import annotations

import re
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

from ragbench.adapters import offline
from ragbench.adapters.llamaindex import from_query_engine
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL
from ragbench.models.embeddings import create_embedding_model
from ragbench.models.llms import create_llm


class OfflineEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> list[float]:
        return offline.embed([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return offline.embed([text])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class OfflineLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="ragbench-offline")

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        # LlamaIndex's question-answering template: "Context information is below.\n---...---\n<context>\n---...---\n...Query: <question>\nAnswer: "
        context = re.search(r"-{5,}\n(.*?)\n-{5,}", prompt, flags=re.S)
        question = re.search(r"Query:\s*(.*?)\s*(?:\nAnswer:\s*)?$", prompt, flags=re.S)
        text, prompt_tokens, completion_tokens = offline.answer(context.group(1) if context else prompt, question.group(1) if question else "")
        return CompletionResponse(text=text, raw={"model": "ragbench-offline", "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}})

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("the offline model does not stream")


class OpenAIEmbedding(BaseEmbedding):
    """`text-embedding-3-small` through RAGBench's client."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in row] for row in create_embedding_model(DEFAULT_EMBEDDING_MODEL).embed_texts(texts).vectors]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class OpenAI(CustomLLM):
    """`gpt-6-luna` through RAGBench's client, as a chat model at temperature 0 (what `llama_index.llms.openai.OpenAI(temperature=0)` would be)."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=DEFAULT_GENERATOR_MODEL, is_chat_model=True)

    def _generate(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        result = create_llm(DEFAULT_GENERATOR_MODEL).generate(messages, temperature=0)
        return result.text, {"model": result.model, "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}}

    @llm_chat_callback()
    def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        text, raw = self._generate([{"role": message.role.value, "content": message.content or ""} for message in messages])
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=raw)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text, raw = self._generate([{"role": "user", "content": prompt}])
        return CompletionResponse(text=text, raw=raw)

    @llm_chat_callback()
    def stream_chat(self, messages: Any, **kwargs: Any) -> ChatResponseGen:
        raise NotImplementedError("the reference pipeline does not stream")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("the reference pipeline does not stream")


_index: VectorStoreIndex | None = None
_llm: Any = None


def ingest(documents: list[Any]) -> None:
    global _index, _llm
    if offline.live():
        embed_model: Any = OpenAIEmbedding()
        _llm = OpenAI()
    else:
        embed_model, _llm = OfflineEmbedding(), OfflineLLM()
    nodes_source = [
        Document(text=doc.text, metadata={"file_path": doc.path}, excluded_embed_metadata_keys=["file_path"], excluded_llm_metadata_keys=["file_path"])
        for doc in documents
    ]
    _index = VectorStoreIndex.from_documents(nodes_source, embed_model=embed_model)


def answer(question: str, top_k: int) -> dict[str, Any]:
    assert _index is not None, "ingest() runs first"
    return from_query_engine(_index.as_query_engine(llm=_llm, similarity_top_k=top_k), model=DEFAULT_GENERATOR_MODEL)(question, top_k)
