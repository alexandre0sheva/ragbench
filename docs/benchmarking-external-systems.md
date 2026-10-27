# Benchmarking external systems

RAGBench can score a RAG it did not build: a service you already run behind HTTP (`external_http`), a Python function of your own (`python_callable`), or an agent harness such as Claude Code, Codex or OpenCode (`cli_agent`). It asks your system the same questions as the built-in ones, judges the answers with the same judge, scores retrieval against the same labels, and puts everything on one leaderboard, so the question "is my production RAG better than a tuned BM25?" gets a number with a confidence interval.

Contents: [Your deployed RAG in 3 steps](#your-deployed-rag-in-3-steps) · [A function of your own](#a-function-of-your-own) · [Framework pipelines](#framework-pipelines) · [Agent harnesses](#agent-harnesses) · [Mapping the reply](#mapping-the-reply) · [Document ids and retrieval metrics](#document-ids-and-retrieval-metrics) · [Cost](#cost) · [Secrets](#secrets) · [Mock runs, estimates and caching](#mock-runs-estimates-and-caching) · [What is measured](#what-is-measured)

Every option, with its type and default, is listed under [`external_http`](systems.md#external_http), [`python_callable`](systems.md#python_callable) and [`cli_agent`](systems.md#cli_agent) in the generated systems reference.

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

## Agent harnesses

`cli_agent` runs a command-line agent harness on every question, as it is, and scores what it finds:

```yaml
systems:
  - type: bm25
  - type: cli_agent
    name: claude_code
    retrieval:
      preset: claude_code              # claude_code | codex | opencode | custom
      model: anthropic:claude-haiku-5-5
      max_steps: 30                    # tool calls; the harness is stopped at the next one
      timeout_s: 300
  - type: cli_agent
    name: codex
    retrieval: {preset: codex, model: gpt-6-luna}
  - type: cli_agent
    name: opencode_glm
    retrieval: {preset: opencode, model: "openrouter:z-ai/glm-5.3"}
```

**What the harness gets.** Each question runs in a fresh temporary directory holding a read-only copy of the corpus under `docs/` (file names and folders as in your dataset; converted files such as PDFs end in `.txt`) and the task in `TASK.md`, which is also the prompt: answer from `docs/` only, name the documents used, say so when the answer is not there. Reference answers, labels and other questions never enter it. The directory is deleted afterwards. Optional harness files: `system_prompt` (text), `instructions` (a file, copied as `AGENTS.md`; Claude Code gets both as an appended system prompt) and `files` (any workspace path, for example a skills directory).

**The presets.** Each builds the command line, reads the harness's event stream, and switches its web access off unless `network: on`:

| Preset | Command | Isolation |
| --- | --- | --- |
| `claude_code` | `claude -p … --output-format stream-json --verbose --restricted --strict-mcp-config --disable-slash-commands --tools Read,Grep,Glob` | file tools confined to the workspace, none of your settings files, plugins, MCP servers or skills, no Bash, no WebFetch/WebSearch; `ANTHROPIC_API_KEY` from your environment or `.env` |
| `codex` | `codex exec --json --sandbox read-only --ephemeral --ignore-user-config -C <workspace>` | Codex's sandbox: read-only, no network; not your `config.toml` (model, reasoning effort, plugins), so Codex runs on its own defaults |
| `opencode` | `opencode run --format json --pure --dir <workspace>`, plus an `opencode.json` in the workspace (with an `openrouter:` model, its OpenRouter base URL is RAGBench's local relay) | no external plugins; edits, shell and webfetch denied |
| `custom` | your `command` (`{prompt}`, `{model}`, `{workspace}`, `{task_file}` are filled in) | none that RAGBench can enforce: it warns |

`model` is a RAGBench model ref: it prices the harness's token usage and keeps the judge independent of it; the harness gets its own spelling (`claude-haiku-5-5`; `openrouter/z-ai/glm-5.3` for OpenCode), or `harness_model` when set. A `custom` harness reads its stdout as `output_format: text` (stdout is the answer) unless you name one of the three event formats; `answer_file` takes the answer from a file the harness writes instead.

A harness may still read a global instructions file of yours on every run (`~/.claude/CLAUDE.md`, `$CODEX_HOME/AGENTS.md`, `~/.config/opencode/AGENTS.md`); RAGBench warns when one has content, so you can empty it for a clean benchmark. Every harness runs with the workspace as its working directory and as `PWD` (a harness that trusts `PWD` would otherwise work in your directory). A file tool pointed at a path outside the workspace is flagged on its step (`outside_workspace`), counted in the answer's `agent.outside_workspace_calls` and logged as a warning: in a sound run it stays 0.

**What is recorded.** Every model turn is an `llm` step (the last one is the `generate` step) and every tool call a `tool` step with its input, its output and the documents it searched or read; a failed call carries its error. The documents the agent read, with the text it read and in the order it read them, are its first passages, followed by documents that only showed up in search results (with the matching lines), so retrieval metrics and the judge see what the agent saw. A search counts the files whose matching text it showed; a bare list of file names (`rg -l`, `rg --files`, Claude Code's default `files_with_matches`, `Glob`, `ls`) is no evidence. The `agent` block of each answer holds the turns, tool calls, whether the step cap stopped it, and `max_prompt_tokens` (the largest single prompt, which decides the price tier of a model such as Haiku 5.5). The cost is what the harness reports (OpenCode's per-step `cost` on a provider other than OpenRouter; Claude Code's `total_cost_usd`, unless Claude Code says it guessed it, `costBasis: unknown`, as it does for a model newer than itself), else its tokens at `model`'s price (Codex; Claude Code on a model it does not know). Cached input is priced as uncached, so a priced cost is an upper bound. Codex reports usage once per run, so its `max_prompt_tokens` is the sum over its model calls, not one prompt. A harness logged in with a subscription (Codex with ChatGPT) is not billed per token: the priced cost is what the same tokens cost on the API. A harness that fails, times out or reports an error fails the question; one stopped by `max_steps` answers with what it last said.

**OpenCode on OpenRouter is charged OpenRouter's bill.** OpenCode prices its own tokens (reasoning tokens included, at the output price) from a price table that holds, for an OpenRouter model, the cheapest upstream's list price; that is also the price in OpenRouter's model catalog. OpenRouter routes each request to one of the model's upstream providers and bills that provider's price, which for an open-weight model is often several times the cheapest: GLM 5.3 lists at $0.07 / $1.40 per 1M tokens, while most of its ~45 upstreams charge $0.50–2.80 input and $2–8.80 output. OpenCode's own `cost` is therefore a floor: on the Task 26 benchmark it came to an estimated quarter to a third of what OpenRouter billed. So with an `openrouter:` model, the `opencode` preset points OpenCode's OpenRouter base URL (`provider.openrouter.options.baseURL` in the workspace `opencode.json`) at a relay RAGBench runs on `127.0.0.1`. The relay forwards every request to OpenRouter with `usage: {include: true}` and records each response's `usage.cost`, the amount OpenRouter charged. That bill is the question's cost, spread over its turns by tokens, and it includes requests that are not turns, such as the session title. The relay also sends the `openrouter:` section's settings in place of OpenCode's own, as for RAGBench's own OpenRouter calls: `provider` routing (so `data_collection: deny` holds for the harness too) and the reasoning effort (the lowest one the model accepts unless you set one). It never sends `exclude`, so OpenCode still gets its reasoning back. A response that carries no cost (a failed request, or one cut off when the harness was stopped) is logged, and the question's cost is then a lower bound. If no billed request goes through the relay, for example with a `command` that is not OpenCode, OpenCode's own cost is kept and a warning says so. Claude Code and Codex call their own vendors' APIs at those vendors' list prices, so their costs need no relay.

**Mock runs.** `--mock` never starts a harness, whatever the preset: a bundled offline harness (`python -m ragbench.rag_systems.cli_agent_mock`) greps the workspace, reads two files and prints the preset's event format, so parsing, step caps and the trajectory metrics are exercised for free. A run that includes a `cli_agent` is live unless you pass `--mock`, even if it calls no RAGBench model. Without `--mock` the real harness runs and spends money: run `ragbench estimate` knowing that it cannot price harnesses, and cap the run with `evaluation.max_cost_usd`.

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
