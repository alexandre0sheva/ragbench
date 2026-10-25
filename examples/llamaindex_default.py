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

With no key, or without `RAGBENCH_REFERENCE_LIVE=1`, it runs on RAGBench's mock models (see `ragbench.adapters.offline`).
"""

from __future__ import annotations

import re
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

from ragbench.adapters import offline
from ragbench.adapters.llamaindex import from_query_engine
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL


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


_index: VectorStoreIndex | None = None
_llm: Any = None


def ingest(documents: list[Any]) -> None:
    global _index, _llm
    if offline.live():
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        embed_model: Any = OpenAIEmbedding(model=DEFAULT_EMBEDDING_MODEL)
        _llm = OpenAI(model=DEFAULT_GENERATOR_MODEL, temperature=0)
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
