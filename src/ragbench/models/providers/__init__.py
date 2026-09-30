"""Built-in model providers. Importing this package registers them in `LLM_PROVIDERS` / `EMBEDDERS`.

Third-party providers can register the same way; see docs/extending.md. Each SDK is imported lazily, so importing this
package needs none of the optional extras.
"""

from __future__ import annotations

from ragbench.models.providers import anthropic, local_embeddings, openai, openai_compatible
from ragbench.registry import EMBEDDERS, LLM_PROVIDERS

# Third-party providers: `factory(model, *, providers) -> LLM` (or `EmbeddingModel`), see docs/extending.md.
LLM_PROVIDERS.load_entry_points("ragbench.llm_providers")
EMBEDDERS.load_entry_points("ragbench.embedders")

__all__ = ["anthropic", "local_embeddings", "openai", "openai_compatible"]
