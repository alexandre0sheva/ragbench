"""Which model family a model ref belongs to, so a judge can be kept from scoring answers written by its own family (self-preference)."""

from __future__ import annotations

import re

from ragbench.models.refs import parse_model_ref

# Matched against the model's own name (the part after any `vendor/` or `endpoint/`), so a model served by an endpoint or a
# router is still recognised; the first pattern that matches wins.
_NAME_FAMILIES: tuple[tuple[str, str], ...] = (
    (r"claude", "anthropic"),
    (r"^(gpt|chatgpt|o\d)", "openai"),
    (r"gemini|gemma", "google"),
    (r"llama", "meta"),
    (r"mistral|mixtral|ministral|magistral|codestral|devstral|pixtral", "mistral"),
    (r"qwen|qwq", "qwen"),
    (r"deepseek", "deepseek"),
    (r"kimi", "moonshot"),
    (r"glm", "zhipu"),
    (r"grok", "xai"),
    (r"minimax", "minimax"),
    (r"mimo", "xiaomi"),
)
# OpenRouter vendor prefixes whose name is not the family's.
_VENDOR_FAMILIES = {"meta-llama": "meta", "mistralai": "mistral", "moonshotai": "moonshot", "z-ai": "zhipu", "x-ai": "xai"}


UNKNOWN_FAMILY = "unknown"


def model_family(ref: str | None) -> str:
    """`gpt-6-luna` -> `openai`, `openrouter:anthropic/claude-haiku-5.5` -> `anthropic`; an unrecognised model is a family of its own
    (its OpenRouter vendor, else its name). None (an external system that does not say what it generates with) is `unknown`, a family no judge belongs to."""
    if ref is None:
        return UNKNOWN_FAMILY
    provider, model = parse_model_ref(ref)
    vendor, _, name = model.rpartition("/")
    lowered = name.lower()
    for pattern, family in _NAME_FAMILIES:
        if re.search(pattern, lowered):
            return family
    if provider == "openrouter" and vendor:
        return _VENDOR_FAMILIES.get(vendor, vendor)
    if provider in {"openai", "anthropic"}:
        return provider
    return lowered
