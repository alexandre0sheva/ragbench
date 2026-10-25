"""LangChain out of the box: the pipeline its RAG tutorial builds, benchmarked as `python_callable`.

    systems:
      - type: python_callable
        name: langchain_default
        retrieval:
          target: examples/langchain_default.py:answer
          ingest_target: examples/langchain_default.py:ingest
          model: gpt-6-luna

Needs `pip install 'ragbench[langchain]'`. The tutorial's defaults: `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap), an in-memory
vector store, one retrieval, a "use the context" prompt, a chat model at temperature 0. The model and the embeddings are the built-in systems'
(`gpt-6-luna`, `text-embedding-3-small`); the retrieval depth is RAGBench's `top_k`, so the systems are compared at the same depth.

With no key, or without `RAGBENCH_REFERENCE_LIVE=1`, it runs on RAGBench's mock models (see `ragbench.adapters.offline`).
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbench.adapters import offline
from ragbench.adapters.langchain import from_retriever
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL

PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


class OfflineEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return offline.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return offline.embed([text])[0]


class OfflineChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "ragbench-offline"

    def _generate(self, messages: list[Any], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        text, prompt_tokens, completion_tokens = offline.complete("\n".join(str(message.content) for message in messages))
        usage = {"input_tokens": prompt_tokens, "output_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=usage))])  # type: ignore[arg-type]


_store: InMemoryVectorStore | None = None
_generate: Any = None


def _format(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"question": inputs["question"], "context": "\n\n".join(doc.page_content for doc in inputs["context"])}


def ingest(documents: list[Any]) -> None:
    global _store, _generate
    if offline.live():
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        embeddings: Embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
        llm: Any = ChatOpenAI(model=DEFAULT_GENERATOR_MODEL, temperature=0)
    else:
        embeddings, llm = OfflineEmbeddings(), OfflineChatModel()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([Document(page_content=doc.text, metadata={"source": doc.path}) for doc in documents])
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    _store, _generate = store, (_format | PROMPT | llm | StrOutputParser())


def answer(question: str, top_k: int) -> dict[str, Any]:
    assert _store is not None, "ingest() runs first"
    retriever = _store.as_retriever(search_kwargs={"k": top_k})
    return from_retriever(retriever, _generate, model=DEFAULT_GENERATOR_MODEL)(question, top_k)
