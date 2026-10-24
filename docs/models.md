# Models and providers

Which model to use for each role in a benchmark, what it costs, and how OpenRouter models (frontier and open-weight) are called. This page is the only place model shortlists and prices are written down; the ref syntax and every config key are in [configuration.md](configuration.md#providers--model-refs), and how costs are counted is in [methodology.md](methodology.md#cost-accounting).

Contents: [Picking models](#picking-models) · [Shortlist](#shortlist) · [OpenRouter](#openrouter) · [Prices](#prices) · [Keeping this page current](#keeping-this-page-current)

## Picking models

A run uses up to four kinds of model, each a model ref (`provider:model`):

- **Generator** (`models.generator`, per system): the model that answers. It is what you are choosing, so compare the candidates in one run with a `sweep:` over `models.generator` ([`configs/models_compare.yaml`](../configs/models_compare.yaml) does that for the open-weight shortlist). Start from a cheap frontier model (`gpt-6-luna`, `anthropic:claude-haiku-5-5`) and add the open-weight models you could self-host.
- **Judge** (`evaluation.judge_model`): grades every answer, so it is called once per question per system and must be cheap. **Take it from another model family than every generator**: a judge rates answers from its own family higher (self-preference, see [methodology.md](methodology.md#bias-and-noise)). `anthropic:claude-haiku-5-5` is the default independent judge for OpenAI and open-weight generators; `gpt-6-luna` judges Anthropic generators. The judge reads what the generator read, and Haiku 5.5 is 5× dearer above 100k prompt tokens, so keep the context a judge sees under that (for `full_context`, `context_token_budget` ≤ 50k).
- **Embedding** (`models.embedding`): `text-embedding-3-small` unless you have a reason; `local:` models (sentence-transformers) cost nothing and keep documents on your machine. OpenRouter serves no embedding models.
- **Proposer / heavy LLM** (`llm_heavy`, and the optimizer of a later release): a strong model, `claude-sonnet-5-5` or `gpt-6.1-sol`; an open-weight alternative is `deepseek/deepseek-v4-pro-0813` through OpenRouter.

Reasoning models spend hidden thinking tokens that are billed as output. RAGBench switches reasoning off, or to the lowest effort a model allows, for every provider: GPT-6 gets `reasoning_effort: none`, Claude models with effort get `low`, and OpenRouter models get `reasoning: {effort: none}` or their lowest effort ([below](#reasoning)).

## Shortlist

USD per 1M input / output tokens, checked against the providers' own pages and OpenRouter's `GET /api/v1/models` on **2026-10-24**. Prices change; `ragbench models list` shows what this installation will charge.

| Role | Model ref | Price | Notes |
| --- | --- | --- | --- |
| Judge (independent), cheap generator | `anthropic:claude-haiku-5-5` | 0.10 / 0.50 | 0.50 / 2.50 for a request whose prompt is over 100,000 tokens |
| Generator (default), judge of Anthropic generators | `gpt-6-luna` | 0.10 / 0.50 | 0.20 / 0.75 above 272K prompt tokens |
| Strong: proposer, `llm_heavy` | `anthropic:claude-sonnet-5-5`, `gpt-6.1-sol` | 2 / 10 | `gpt-6.1-sol`: 4 / 15 above 272K |
| Strongest | `anthropic:claude-opus-5-5` / `anthropic:claude-fable-5-1` / `gpt-6-astra` | 4 / 20, 10 / 50, 10 / 50 | Rarely worth it for RAG answers |
| Frontier via OpenRouter | `openrouter:google/gemini-3.8-flash` | 0.825 / 4.125 | Must reason: lowest effort `low` |
| Frontier via OpenRouter | `openrouter:x-ai/grok-4.7` | 2 / 6 | Must reason: lowest effort `low` |
| Open-weight | `openrouter:deepseek/deepseek-v4.1-flash` | 0.02 / 1.00 | `deepseek/deepseek-v4-pro-0813` 0.66 / 1.98; `deepseek-v4.1-flash-1017` (2026-10-17) is a separate model at 0.15 / 0.60 |
| Open-weight | `openrouter:z-ai/glm-5.3` | 0.079 / 1.40 | Must reason: lowest effort `low`. `glm-5.3-flash` 0.15 / 0.50 |
| Open-weight | `openrouter:moonshotai/kimi-k3` | 0.239 / 10.80 | Expensive output: keep answers short. `kimi-k3.1-pro` (2026-10-23) lists no open weights |
| Open-weight | `openrouter:xiaomi/mimo-v2.6-pro` | 0.435 / 0.87 | `mimo-v2.6-flash` 0.14 / 0.28 |
| Open-weight | `openrouter:qwen/qwen3.8-flash` | 0.15 / 0.47 | `qwen3.8-2.4t-a95b` 2 / 6, must reason |
| Open-weight | `openrouter:minimax/minimax-m3` | 0.30 / 1.20 | `minimax-m3.1-flash` 0.15 / 0.60, must reason |
| Open-weight | `openrouter:mistralai/mistral-small-2603` | 0.15 / 0.60 | |
| Embedding | `text-embedding-3-small` | 0.02 | `text-embedding-3-large` 0.13; `local:BAAI/bge-small-en-v1.5` free |

Every open-weight row has a Hugging Face id on OpenRouter (`ragbench models list --open-weights`), takes tools and JSON output, and is a reasoning model.

**Smoke test, 2026-10-24** (each main model above, 5 demo questions through `hybrid_rerank` and `agent_search`, Haiku 5.5 judging; about $0.19 in all): every model answered without errors, called the search tool natively and returned valid JSON in JSON mode, and with reasoning at `none` none of them billed reasoning tokens. Two behaviours to know about:

- `moonshotai/kimi-k3` and `minimax/minimax-m3` sometimes answer an agent's first turn without searching (2 of 5 questions each), so they score low as `agent_search` generators. They are fine in pipelines that retrieve for them.
- `x-ai/grok-4.7` cannot switch reasoning off; at `low` it still spent most of a short reply's output tokens thinking, which is why it is the dearest per question here.

## OpenRouter

`openrouter:<vendor>/<model>` calls any model on [OpenRouter](https://openrouter.ai) with `OPENROUTER_API_KEY` (in `.env` or your shell). The model id is OpenRouter's, exactly: `openrouter:deepseek/deepseek-v4.1-flash`, `openrouter:anthropic/claude-haiku-5.5`. Without the key the model is the mock, like any hosted provider. (A generic `openai_compatible:` endpoint pointed at OpenRouter still works, but gets none of the following.)

### Billed cost

Every OpenRouter response says what the request cost (`usage.cost`), and that is the cost RAGBench charges, including on a cache hit (the stored billed cost is replayed). The price table is only used to estimate (`ragbench estimate`) and as the fallback for a response without a cost. See [methodology.md](methodology.md#cost-accounting).

### Reasoning

Each request carries `reasoning: {effort: none}`, or, for a model that cannot switch reasoning off, the lowest effort it lists (`low` for `z-ai/glm-5.3`), with `exclude: true` so the thinking text is not returned. Override it for all models or one model under `openrouter:`; an effort the model does not accept fails before the first request. The settings in force are written to `run_manifest.json` (`model_settings`).

### Provider pinning

OpenRouter routes a request to one of several upstream providers, which can differ in quantization, speed and price. For a result someone else can reproduce, pin them:

```yaml
openrouter:
  reasoning: {effort: none}              # the default; or {max_tokens: 512} for a thinking budget
  provider:
    data_collection: deny                # skip upstreams that may store or train on prompts
  models:
    z-ai/glm-5.3:
      reasoning: {effort: low}
      provider: {order: [z-ai], allow_fallbacks: false}   # an `order` pins: fallbacks stay off unless you allow them
```

A per-model entry replaces the section-wide `reasoning` or `provider`, it does not merge with it. Settings change answers, so they are part of the cache key.

### Tools and JSON

Whether a model takes function calling and JSON output is read from its `supported_parameters` in the catalog. A model without tools runs the tool-using agents (`agent_search`, `grep_agent`) in `react_json` mode, with the tools described in the prompt; one without JSON output is asked for JSON in the prompt instead.

### The catalog: `ragbench models sync`

OpenRouter prices, tool support and reasoning switches come from a catalog file, read offline. A snapshot of the shortlist above ships with RAGBench; for any other model, or fresher prices, run:

```bash
ragbench models sync
```

It downloads `GET https://openrouter.ai/api/v1/models` (public, no key, free) and saves a trimmed, dated copy to `~/.cache/ragbench/openrouter_models.json` (or `$RAGBENCH_OPENROUTER_CATALOG`). The newer of that file and the snapshot is used. A prompt-length price becomes a tier, a time-of-day discount is ignored (the base price is kept), and a router with a variable price gets none. `ragbench models list` shows every price RAGBench knows, with filters (`--provider openrouter --open-weights --search qwen`).

## Prices

The built-in table for OpenAI and Anthropic is `src/ragbench/models/cost.py` (`PRICING_AS_OF` is its date, and the source pages are listed above it); OpenRouter prices come from the catalog. Some prices depend on prompt length: each request is priced on its own, at the tier its prompt falls in, and a request over the threshold pays the higher rates for all of its tokens. `pricing:` in a config overrides any of them, tiers included ([configuration.md](configuration.md#pricing)).

## Keeping this page current

Re-run the audit before every paid run and every release ([release checklist](release-checklist.md)): fetch OpenAI's and Anthropic's pricing and model pages and OpenRouter's model list on the day, move each default and shipped config to the newest model in its tier, delete superseded rows from the price table, run `ragbench models sync`, and update the shortlist above.
