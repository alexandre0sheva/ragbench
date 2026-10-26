# Exporting traces

Every answer a system gives is traced step by step (retrieve, rerank, llm, tool, ...). `ragbench export-traces` turns a **finished run's** traces into OpenTelemetry / [OpenInference](https://github.com/Arize-ai/openinference) spans, so you can browse them in [Phoenix](https://phoenix.arize.com), [Langfuse](https://langfuse.com) or any OTLP backend. It reads the run's files: nothing is re-run and nothing is paid for. Every flag: [the command-line reference](cli.md#ragbench-export-traces).

```bash
ragbench export-traces latest                                   # offline: traces.jsonl in the run directory
pip install 'ragbench[otel]'                                    # only for sending
ragbench export-traces latest --format openinference --endpoint http://localhost:6006/v1/traces      # Phoenix
ragbench export-traces latest --format openinference --system hybrid_rerank --limit 20 --project my-rag
```

## Formats

| `--format` | Where it goes | Attributes |
| --- | --- | --- |
| `jsonl` (default) | a file, offline (`--out`, default `traces.jsonl` in the run) | OpenInference |
| `openinference` | an OTLP/HTTP `--endpoint` | OpenInference: `openinference.span.kind`, `input.value`, `output.value`, `llm.token_count.*`, `llm.cost.total`, `retrieval.documents.N.document.id`, `session.id` |
| `otlp` | an OTLP/HTTP `--endpoint` | OpenTelemetry GenAI: `gen_ai.operation.name`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.tool.name`, plus `ragbench.input` / `ragbench.output` |

Phoenix and Langfuse both read OpenInference. Use `otlp` for a collector that expects the GenAI names.

The endpoint is the full traces URL. Phoenix: `http://localhost:6006/v1/traces`. Langfuse: `https://cloud.langfuse.com/api/public/otel/v1/traces` (or your host's) with an `Authorization` header. `--endpoint` falls back to `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, and the standard `OTEL_EXPORTER_OTLP_HEADERS` is honoured when no `--header` is given.

```bash
export LANGFUSE_AUTH="Basic $(printf '%s:%s' "$LANGFUSE_PUBLIC_KEY" "$LANGFUSE_SECRET_KEY" | base64)"
ragbench export-traces latest --format openinference \
  --endpoint https://cloud.langfuse.com/api/public/otel/v1/traces --header 'Authorization=${LANGFUSE_AUTH}'
```

`${ENV_VAR}` in a header value is filled in when the command runs; a variable that is not set stops the command before anything is sent, and header values are never printed.

## What a trace holds

One trace per answered question (and per trial, when a run has several): a root span `ragbench.answer` with the question, the answer, the system, the run, the category, the failure type, the judge's answer score and the cost, and one child span per step, named `kind:name` (`retrieve:bm25_search`), with its input and output preview, tokens, cost and the documents a retrieval surfaced. A question that failed, or a step that raised, is an error span carrying the message.

- **Ids are derived** from the run, system, question and trial, so exporting a run twice updates the same traces in a backend instead of duplicating them.
- **Durations are real; start times are laid out.** A run records how long each step took, not when it began. A system's traces start at the run's start and follow one another, and a trace's steps follow one another.
- **Never exported:** reference answers and the judge's reasoning. A trace backend is somebody else's server; the question and the system's answer and retrieved document ids do leave your machine, so point `--endpoint` only at a backend you trust with your dataset.
- The resource names the project (`openinference.project.name`, the run's name unless `--project` is given) and `service.name=ragbench`.
