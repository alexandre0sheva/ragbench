# Tools

A tool is a small, deterministic function an agent can call during a question: a calculator, a date calculator, a grep over the corpus, a lookup in a table, or your own Python function. Tools are traced (every call is a `tool` step with its arguments, output, latency and cost), budgeted, safe to run on untrusted model output, and work in `--mock` mode with no network.

Why they are a benchmark axis: the same agent with `tools: []` and with `tools: [calculator, date_calc, corpus_grep]` are two systems in one comparison, so the report shows what the tools bought (answer score on the questions that need them) and what they cost (`$/Q`, latency, error rate).

## Using tools

Give a system that supports tools a `tools:` list. An entry is a built-in's name, or a mapping with the tool's options:

```yaml
systems:
  - type: agent_search          # a system that accepts tools (see the list at the end of this page)
    name: agent_with_tools
    tools:
      - calculator
      - date_calc
      - {name: corpus_grep, max_results: 10}
```

A system that does not use tools rejects a `tools:` list when the config loads, and an unknown tool name fails there too, with a did-you-mean hint.

Questions can declare the tools they need (`requires_tools`, see [dataset-format.md](dataset-format.md)); names are checked against the registry below. The run reports `required_tool_used_rate` for them.

## Built-in tools

<!-- tools:start -->
| Tool | What it does | Side effects | Tool options |
| --- | --- | --- | --- |
| [`calculator`](#calculator) | Evaluate an arithmetic expression exactly. Use it for any calculation instead of doing the arithmetic yourself. | none | none |
| [`list_documents`](#list_documents) | List the documents in the collection: id, title and length, in order. Use it to see what exists before reading. | none | none |
| [`read_document`](#read_document) | Read a window of a document's text by id. Long documents are read in several calls by moving `start`. | none | none |
| [`corpus_grep`](#corpus_grep) | Search every document's lines for words or a pattern, like grep. Returns `doc_id:line: text` with a little context. Good for identifiers, names and exact figures. | none | `max_results` |
| [`lookup_table`](#lookup_table) | Look up rows of the CSV / JSON tables in the collection by column value and return chosen columns. Exact cell values instead of searching prose. | none | none |
| [`date_calc`](#date_calc) | Date arithmetic: the difference between two dates, a date plus or minus an offset, or the weekday of a date. `today` is a fixed date for the whole run. | none | none |
| [`search`](#search) | Search the document collection and return the best matching passages with their document ids. Cite those ids in the answer. | none | none |

#### `calculator`

Evaluate an arithmetic expression exactly. Use it for any calculation instead of doing the arithmetic yourself.

Arguments the model passes:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `expression` | `str` | *required* | Arithmetic to evaluate, e.g. `(1200 - 950) / 950 * 100`. Supports + - * / // % ** (or ^), parentheses, round(x, digits), min, max, sum, abs, pi and e. |

#### `list_documents`

List the documents in the collection: id, title and length, in order. Use it to see what exists before reading.

Arguments the model passes:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `contains` | `str` | unset | Only documents whose title or id contains this text (case-insensitive). |
| `limit` | `int` | `50` | Most documents to list. |
| `offset` | `int` | `0` | Documents to skip (to page through a long list). |

#### `read_document`

Read a window of a document's text by id. Long documents are read in several calls by moving `start`.

Arguments the model passes:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `doc_id` | `str` | *required* | Document id, as listed by `list_documents` or cited in search results. |
| `start` | `int` | `0` | Character offset to start reading at. |
| `length` | `int` | `2000` | Characters to read (at most 8000). |

#### `corpus_grep`

Search every document's lines for words or a pattern, like grep. Returns `doc_id:line: text` with a little context. Good for identifiers, names and exact figures.

Arguments the model passes:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `pattern` | `str` | *required* | What to look for. By default every word must appear on the same line (case-insensitive); with `regex` it is a regular expression. |
| `regex` | `bool` | `false` | Treat `pattern` as a regular expression (simple patterns only). |
| `case_sensitive` | `bool` | `false` |  |
| `doc_id` | `str` | unset | Search only this document. |
| `context_lines` | `int` | `1` | Lines of context around each match. |
| `max_results` | `int` | `20` | Most matching lines to return. |

Options (`tools: [{name: corpus_grep, ...}]`):

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `max_results` | `int` | `50` | Hard cap on matching lines per call, whatever the model asks for. |

#### `lookup_table`

Look up rows of the CSV / JSON tables in the collection by column value and return chosen columns. Exact cell values instead of searching prose.

Arguments the model passes:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `where` | `dict` | *required* | Column values the row must have, e.g. {"product": "HarborShield"}. Compared ignoring case and surrounding spaces. |
| `select` | `list` | unset | Columns to return (default: all of them). |
| `source` | `str` | unset | Only rows of the table whose file path or title contains this text. |
| `limit` | `int` | `5` | Most rows to return. |

#### `date_calc`

Date arithmetic: the difference between two dates, a date plus or minus an offset, or the weekday of a date. `today` is a fixed date for the whole run.

Arguments the model passes:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `operation` | `diff` \| `add` \| `weekday` \| `today` | *required* | `diff`: days from `date` to `other_date`. `add`: `date` plus the given offsets. `weekday`: the weekday of `date`. `today`: the current date. |
| `date` | `str` | unset | A date: YYYY-MM-DD, `March 5, 2025`, or `today`. The start for `diff`, the base for `add` (default today). |
| `other_date` | `str` | unset | `diff` only: the end date. |
| `days` | `int` | `0` | `add` only: days to add (negative to subtract). |
| `weeks` | `int` | `0` | `add` only: weeks to add. |
| `months` | `int` | `0` | `add` only: calendar months to add. |
| `years` | `int` | `0` | `add` only: years to add. |

#### `search`

Search the document collection and return the best matching passages with their document ids. Cite those ids in the answer.

Arguments the model passes:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `str` | *required* | What to search for; a short natural-language query or the key terms. |
| `top_k` | `int` | `5` | Passages to return. |

#### Run-wide rules (`tools:` section)

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `allow` | `list` | *required* | Tools with side effects (network, filesystem) that may be used; a tool with side effects that is not listed here is refused when the config loads. |
| `allow_network` | `bool` | `false` | Permit tools that declare `side_effects: network` (they must also be listed in `allow`). No network tool is built in. |
| `timeout_s` | `float` | `10.0` | A tool call that takes longer returns an error result instead of hanging the question. |
| `max_output_chars` | `int` | `4000` | Tool output longer than this is cut and flagged `truncated`, so one call cannot flood the model's context. |

`evaluation.tools_now` (default `2026-01-01`) is what `date_calc` treats as today for the whole run.

Systems that accept a `tools:` list: [`agent_search`](systems.md#agent_search).
<!-- tools:end -->

Network tools (`http_get`, `web_search`) are deliberately **not** built in: a benchmark that depends on the live web is not reproducible. See [Network tools](#network-tools).

## Tool-using systems

Two built-in systems call tools (their options are in [systems.md](systems.md)):

- **`agent_search`** is a function-calling agent. It always has a `search` tool (the same hybrid or vector retrieval the plain systems use) plus the tools in its `tools:` list, decides when to search, compute or grep, and writes the cited answer itself. Retrieval is scored on the passages its tool calls returned: each call that returned passages (search hits, grep matches, a document window) is one ranking, and the rankings are merged with Reciprocal Rank Fusion. `agent_mode: react_json` replaces native function calling with one JSON action per reply, for models that cannot call functions.
- **`grep_agent`** is index-free: no chunking, no embeddings, no vector store. Its only tools are `list_documents`, `corpus_grep` and `read_document`. It shows whether your corpus needs retrieval infrastructure at all. Ingestion is free; the cost is paid per question, in agent turns, and it takes no `tools:` list.

**Tool sets are benchmark variants.** `configs/all.yaml` runs `agent_search` twice, once with `tools: []` (search only) and once with `tools: [calculator, date_calc, corpus_grep]`, so every comparison shows what the tools bought (the `required_tool_used_rate` and answer score on questions that need them) against what they cost (`$/Q`, latency, steps).

Limits per question: `max_steps` (model turns), `max_tool_calls`, `max_cost_usd`, `max_tokens`. When one is reached the agent stops using tools and is asked to answer with what it found; that answer is flagged `best_effort` in the `agent` block of `per_question_results.jsonl`, and a cap (not the step limit) also sets `budget_exhausted`. The final answer call is always made, so a question can end slightly over a cost cap by that one call.

## Your own tools

Any Python function with type hints is a tool. Point to it by `package.module:function`:

```yaml
tools:
  - {name: lookup_price, path: "my_pkg.tools:lookup_price"}
```

```python
def lookup_price(product: str, quantity: int = 1) -> str:
    """Price of a product in the internal catalogue."""   # the first paragraph becomes the description
    ...
```

- The parameters' type hints become the JSON schema the model sees (`Annotated[str, Field(description="...")]` adds a description to one parameter). Every parameter needs a hint; `*args` / `**kwargs` are not supported.
- A parameter named `ctx` (or annotated `ToolContext`) receives the per-question context (`ctx.corpus`, `ctx.now`, `ctx.state`) and is hidden from the model.
- Return a `str` (used as is), any JSON-serialisable value (serialised), or a `ToolResult` (to set `data`, `cost_usd` or `error` yourself).
- Options for a custom tool: `description` (overrides the docstring), `side_effects` (`none` default, `network`, `filesystem`), `timeout_s`.
- A custom tool cannot reuse a built-in's name. Packages can also contribute tools to the registry (see [extending.md](extending.md#plugins-from-other-packages)), which makes them valid in `requires_tools` too.

## Safety model

Tool arguments come from a language model, so a tool must be safe against whatever it sends.

- **No `eval`.** `calculator` parses with `ast` and evaluates a whitelist (numbers, `+ - * / // % **`, parentheses, `round`, `min`, `max`, `sum`, `abs`, `pi`, `e`). Names, attributes, calls to anything else, strings, comprehensions and lambdas are rejected. Expressions are limited to 500 characters and 200 nodes, exponents to 1000 and integer results to about 3000 digits, so `9**9**9` is an error, not a hang.
- **Regular expressions are vetted, not just timed.** Python's `re` holds the interpreter while it backtracks, so no timeout can stop `(a+)+$`. `corpus_grep` refuses such patterns (a repeated group containing a repeat or an alternation, back-references, patterns over 200 characters) before matching, and only scans the first 2000 characters of a line. Without `regex: true` the pattern is plain words.
- **Timeouts.** A call that exceeds `tools.timeout_s` (default 10 s) returns an error result; the question carries on. The abandoned thread is a daemon and cannot keep the process alive.
- **Exceptions never propagate.** An exception, a malformed return value, an unknown tool name or bad arguments all become a `ToolResult` with `error` set, which the model sees as an observation and can recover from.
- **Bounded output.** Output over `tools.max_output_chars` (default 4000) is cut and flagged `truncated`, so one call cannot flood the model's context.
- **Permissions.** Built-in tools only read the corpus given to them. A tool with side effects must declare `side_effects` and be listed under `tools.allow`; a network tool also needs `tools.allow_network: true`. Otherwise the config does not load.
- **Determinism.** "Today" is `evaluation.tools_now`, fixed for the run, so date answers do not depend on when the benchmark runs. Each question gets its own `ToolContext`, so concurrent questions never share state.

## Network tools

None ships with RAGBench, because results from the live web cannot be reproduced and a benchmark should not make requests nobody asked for. To use one, write it as a custom tool with `side_effects: network`, list it under `tools.allow`, and set `tools.allow_network: true`:

```yaml
tools:
  allow: [web_search]
  allow_network: true
systems:
  - type: agent_search
    tools: [{name: web_search, path: "my_pkg.tools:web_search", side_effects: network, timeout_s: 20}]
```

Mock runs call your function too, so guard it yourself (for example by returning a canned result when an API key is missing) if you want `--mock` to stay offline.

## Metrics

Every tool call is a `tool` step in `per_question_results.jsonl`, and its cost counts as `tool_cost`. For systems that have tools, `metrics_summary.csv` adds:

| Column | Meaning |
| --- | --- |
| `avg_tool_calls` | Mean tool calls per question. |
| `tool_error_rate` | Share of calls that returned an error (empty if no call was made). |
| `required_tool_used_rate` | Over the questions that declare `requires_tools`: the share where the system called every required tool at least once. |

`tool_usage.csv` has one row per system and tool: `calls`, `errors`, `error_rate`, `truncated` (outputs that hit the size cap), `questions_using`, `avg_latency_ms` and `total_cost_usd`. A tool that was offered but never called still gets a row with 0 calls.

An agent's budget can cap tool use with `max_tool_calls` (see `AgentBudget` in [extending.md](extending.md)).
