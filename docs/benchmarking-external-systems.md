# Benchmarking external systems

RAGBench can score a RAG it did not build: a service you already run behind HTTP (`external_http`), or a Python function of your own (`python_callable`). It asks your system the same questions as the built-in ones, judges the answers with the same judge, scores retrieval against the same labels, and puts everything on one leaderboard, so the question "is my production RAG better than a tuned BM25?" gets a number with a confidence interval.

Contents: [Your deployed RAG in 3 steps](#your-deployed-rag-in-3-steps) · [A function of your own](#a-function-of-your-own) · [Framework pipelines](#framework-pipelines) · [Mapping the reply](#mapping-the-reply) · [Document ids and retrieval metrics](#document-ids-and-retrieval-metrics) · [Cost](#cost) · [Secrets](#secrets) · [Mock runs, estimates and caching](#mock-runs-estimates-and-caching) · [What is measured](#what-is-measured)

Every option, with its type and default, is listed under [`external_http`](systems.md#external_http) and [`python_callable`](systems.md#python_callable) in the generated systems reference.

## Your deployed RAG in 3 steps

**1. Point a system at your endpoint.** Add it to a config next to the built-in systems you want to compare with ([`configs/external_example.yaml`](../configs/external_example.yaml) is a complete one):

```yaml
systems:
  - type: bm25
    name: bm25_default
  - type: external_http
    name: my_rag
    retrieval:
      url: https://rag.internal.example/ask
      headers: {Authorization: "Bearer ${MY_RAG_TOKEN}"}
```

By default each question is sent as `POST {"question": "...", "top_k": 10}` and the reply is read as `{"answer": "...", "contexts": [{"doc_id": "...", "text": "..."}], "usage": {...}, "cost_usd": ...}`. If your API looks different, change `request` (a Jinja template) and `response` (JSONPath), see [Mapping the reply](#mapping-the-reply).

**2. Use the documents your system answers from.** Retrieval is scored against your dataset's labels, so `dataset.documents_path` must hold the same documents your system has indexed, and the ids it returns must name them ([how ids are matched](#document-ids-and-retrieval-metrics)). If your service can index documents it is sent, set `post_documents_to: https://rag.internal.example/documents` and RAGBench posts the corpus there once before the questions.

**3. Run it.** Check the configuration first with the stand-in service, then run against yours:

```bash
python examples/fake_rag_server.py --docs data/demo/docs      # in another terminal: a stand-in for your service
ragbench run --config configs/external_example.yaml --mock
```

The leaderboard, report and recommendation treat your system like any other. Each of its answers is one `external` step in the trace (RAGBench cannot see inside it), with the latency and cost of the call; a [function of your own](#a-function-of-your-own) can record finer steps.

## A function of your own

`python_callable` calls a function in-process, which suits a pipeline you are still building or one written with a framework:

```yaml
  - type: python_callable
    name: my_pipeline
    retrieval:
      target: my_pkg.rag:answer               # or a file: path/to/rag.py:answer (relative to the working directory)
      ingest_target: my_pkg.rag:ingest        # optional: called once with the documents
      model: gpt-6-luna                       # optional: prices reported token usage, keeps the judge independent
```

`answer(question, top_k)` returns a dict with an `answer` and, optionally, `contexts` (a list of `{"text", "doc_id", "score"}`, or plain strings), `usage` (`prompt_tokens`, `completion_tokens`) and `cost_usd`; or an `AnswerResult`. `ingest(documents)` receives RAGBench's `Document` objects (`doc_id`, `path`, `title`, `text`). Questions run concurrently (`evaluation.max_workers`), so the function must be thread-safe. [`examples/simple_rag.py`](../examples/simple_rag.py) is a minimal working example; a target and its ingest function in the same file share the file's state.

By default the whole call is one `external` step in the trace. To split it (retrieval, reranking, generation, each with its own latency, tokens and cost, so the report's cost-by-stage table shows where the money goes), record steps while the function runs, on the tracer RAGBench activates for the question:

```python
from ragbench.rag_systems.trace import current_tracer

def answer(question, top_k):
    tracer = current_tracer()
    with tracer.step("retrieve", "my_search") as step:
        passages = search(question, top_k)
        step.set_cost(embedding_cost)                    # a CostBreakdown
    with tracer.step("generate", "my_llm") as step:
        text, usage = generate(question, passages)
        step.set_cost(llm_cost)
        step.set_tokens(usage.prompt, usage.completion)
    return {"answer": text, "contexts": passages}
```

Or return them: an `AnswerResult` keeps its `steps`, and a dict may carry `steps` as plain dicts (`{"kind": "generate", "name": "llm", "latency_ms": 820, "prompt_tokens": 1200, "completion_tokens": 90, "cost": {"llm_cost": 0.0002}}`). Steps recorded on the tracer win over returned ones. When the answer reports no `cost_usd` (and no `AnswerResult` cost), its cost is the sum of its steps' costs; any difference between the steps and the cost appears as an `untracked` step, as for the built-in systems.

## Framework pipelines

For a pipeline written with LangChain, LlamaIndex or Haystack, `ragbench.adapters` turns the pipeline object into the `answer(question, top_k)` function above, and splits what it did into `retrieve` and `generate` steps with the tokens each model call reported. Install the extra for your framework (`pip install 'ragbench[langchain]'`, `'ragbench[llamaindex]'` or `'ragbench[haystack]'`); without it the adapter fails with that one-line command.

```python
# my_rag.py
from ragbench.adapters.langchain import from_chain, from_retriever
answer = from_retriever(retriever, generate, model="gpt-6-luna")   # a retriever, and a runnable taking {"question", "context": [Document]}
answer = from_chain(retrieval_chain, model="gpt-6-luna")           # or one runnable that retrieves and answers

from ragbench.adapters.llamaindex import from_query_engine
answer = from_query_engine(index.as_query_engine(similarity_top_k=10), model="gpt-6-luna")

from ragbench.adapters.haystack import from_pipeline
answer = from_pipeline(pipeline, inputs=lambda question, top_k: {...}, retriever="retriever", generator="llm", model="gpt-6-luna")
```

Point `target` at `my_rag.py:answer` and declare the same `model`. What the adapters read:

| Framework | Steps come from | Passages and their file | Tokens and model |
| --- | --- | --- | --- |
| LangChain | a callback handler passed to the runnable | the documents the chain returns under `context` (else what its retrievers returned); `metadata["source"]`, `file_path`, ... | `llm_output["token_usage"]`, or the message's `usage_metadata` |
| LlamaIndex | instrumentation events (registered once, filed under the question being answered) | the response's `source_nodes`; node `metadata["file_path"]`, `file_name`, ... | the provider's raw `usage` |
| Haystack | the retriever and generator components, wrapped in place to time them | the retriever's `documents`; `meta["file_path"]`, `source_id`, ... | the generator's `meta["usage"]` |

Three things to know. The framework decides how many passages it retrieves, so set `k` on your retriever or engine (Haystack's `inputs` receives RAGBench's `top_k`); to score all systems at one depth, pass the run's `top_k` through yourself, as the reference pipelines do. A document whose metadata names no corpus file leaves its question's retrieval blank ([ids](#document-ids-and-retrieval-metrics)). Query-embedding calls made inside the framework are not seen, so their (small) cost is not counted; the generator's is, priced at the model the framework reports, or at your declared `model` when it reports none or one without a price.

**The three frameworks, out of the box.** [`examples/langchain_default.py`](../examples/langchain_default.py), [`examples/llamaindex_default.py`](../examples/llamaindex_default.py) and [`examples/haystack_default.py`](../examples/haystack_default.py) build the pipeline each framework's own tutorial recommends (its default splitter, an in-memory vector store, one retrieval, a plain question-answering prompt), with the built-in systems' generator and embedding model, and [`configs/frameworks_example.yaml`](../configs/frameworks_example.yaml) runs them next to `bm25`. They call OpenAI only when you set `RAGBENCH_REFERENCE_LIVE=1` (and `OPENAI_API_KEY`), because `--mock` does not stub an external system; otherwise they use RAGBench's mock models, so the configuration and the trace can be checked with no key and no cost. They are also the starting point for your own: copy one and replace the pieces.

## Mapping the reply

`request` is a Jinja template that renders a JSON object from `question` and `top_k`; `POST` sends it as the body and `GET` as query parameters:

```yaml
      request: '{"query": {{ question | tojson }}, "k": {{ top_k }}, "filters": {"lang": "en"}}'
```

`response` says where each part of the reply is, in a small JSONPath subset (`$.a.b`, `$['a-b']`, `$.a[0]`, `$.a[*]`). `contexts` selects one value per passage; `text`, `doc_id` and `score` are then read from each passage:

```yaml
      response:
        answer: $.data.output.text
        contexts: $.data.sources[*]
        text: $.content
        doc_id: $.meta.id
        cost_usd: $.billing.usd
```

A reply without the answer fails that question (it is counted as an error, like any failed question); every other field is optional. A reply that is not JSON, an HTTP 4xx, or a template that does not render JSON fails the question with a message naming the URL. Timeouts, connection errors, HTTP 429 and 5xx are retried (`retries`, with exponential backoff), and `limits` throttles requests to the endpoint.

## Document ids and retrieval metrics

A passage counts for retrieval only if its id names a document of your dataset. An id matches a document by the document's id, its path, its file name or its file stem, so `docs/policy.md`, `policy.md` and `policy` all name `policy.md`; ambiguous names (two files with the same stem) do not match.

When your system returns no passages at all (`response.contexts` unset), or any of its passages carries an id that matches no document, its retrieval metrics and context metrics are left **blank for every question**, never zero: a 0 would read as "retrieval failed" when RAGBench simply cannot tell. The run says so in a notice with example ids, and `run_summary.json` lists them under `external`. The answer is still judged, but faithfulness can only be checked against the passages you return.

## Cost

An external system's cost is what it reports (`cost_usd`), else its reported token usage priced at the declared `model`, else unknown: it then shows $0 per question and every report carries a notice naming it. The cost counts toward `evaluation.max_cost_usd` like any other system's. `ragbench estimate` never calls an external system, so its own cost, and the judge's cost on its answers, are not in the estimate (it says so).

## Secrets

Write secrets as `${VARIABLE}` in `headers` or `url`; they are filled in from the environment (and the project `.env`) when each request is sent. The configuration that RAGBench saves with a run (`run_manifest.json`, `config.yaml`) keeps the `${VARIABLE}` placeholder, error messages name the URL as written, and a variable that is not set fails the question naming the variable, never a value.

## Mock runs, estimates and caching

`--mock` replaces RAGBench's own models (the built-in systems' generators and embeddings, and the judge) with the offline mock; it does not stub your system, which is always called, because it is what is being measured. Point a mock run at a stand-in (such as `examples/fake_rag_server.py`) when you do not want to call the real service. External calls are not cached: every run asks your system again, so its latency and answers are current.

## What is measured

- **Latency** is the wall time of one call as RAGBench sees it, network included, so measure from where your users are. Live runs re-ask a few questions one at a time for the p95 ([latency measurement](methodology.md#latency-measurement)).
- **The judge** sees the answer and the passages your system reports as the context it used; RAGBench's `context_k` does not cut them. Declare `model` so a judge of another family scores it ([judge panels](methodology.md#judge-panels-and-agreement)); without it any judge may score it.
- **Retrieval depth.** `top_k` in the request is the run's retrieval depth (`evaluation.retrieval_depth`, at least the context size), the same depth the built-in systems are scored at.
