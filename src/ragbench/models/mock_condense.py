"""Mock-safe answers to the prompts that condense text: `compression`'s quote extraction and `raptor`'s cluster summaries.

The quotes are the passage's sentences that share a content word with the question (verbatim, as the real prompt demands); a summary is the
first sentence of each passage, so a mock run builds a real compressed context and a real tree with no network.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragbench.models.prompts import COMPRESS_PASSAGES_MARKER, SUMMARIZE_CLUSTER_MARKER
from ragbench.utils.text import lead_sentences, sentence_pieces, tokenize

SUMMARY_CHARS = 600


def _content_words(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 3}


def _passages(prompt: str) -> list[tuple[int, str]]:
    return [(int(number), body) for number, body in re.findall(r"^\[(\d+)\] (.*)$", prompt.split("Passages:", 1)[-1], flags=re.M)]


def _quotes(prompt: str) -> str:
    match = re.search(r"Question:\s*(.*?)\n\nPassages:", prompt, flags=re.S)
    wanted = _content_words(match[1] if match else "")
    replies = []
    for number, body in _passages(prompt):
        quotes = [sentence for sentence in sentence_pieces(body) if wanted & _content_words(sentence)]
        replies.append({"id": number, "quotes": quotes})
    return json.dumps({"passages": replies})


def _summary(prompt: str) -> str:
    return lead_sentences([body for _, body in _passages(prompt)], SUMMARY_CHARS)


def register(mock_llm: Any) -> None:
    """Add the responders to `MockLLM` (called when `ragbench.models.llms` is imported); registered after JSON mode so they win over it."""
    for name, marker, respond in (("compress_passages", COMPRESS_PASSAGES_MARKER, _quotes), ("summarize_cluster", SUMMARIZE_CLUSTER_MARKER, _summary)):
        mock_llm.register_responder(lambda prompt, json_mode, marker=marker: marker in prompt, respond, name=name)
