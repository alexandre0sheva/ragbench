"""Token spans for chunking: real `o200k_base` tokens when tiktoken can load its vocabulary, otherwise a deterministic approximation.

tiktoken downloads its vocabulary on first use, which is impossible offline (CI, `--mock` on a plane). Rather than make
chunking depend on the network, the fallback splits text the way BPE roughly would (short runs of word characters,
single punctuation marks, one token per CJK character) and logs a warning once; chunk metadata records which tokenizer ran.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ENCODING = "o200k_base"


class Tokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token, in order."""
        ...


class TiktokenTokenizer:
    def __init__(self, encoding: Any):
        self.name = ENCODING
        self._encoding = encoding

    def spans(self, text: str) -> list[tuple[int, int]]:
        tokens = self._encoding.encode(text, disallowed_special=())
        if not tokens:
            return []
        _, starts = self._encoding.decode_with_offsets(tokens)
        ends = [*starts[1:], len(text)]
        return list(zip(starts, ends, strict=True))


class ApproxTokenizer:
    """~1.2 tokens per English word: word-character runs of up to 5, single punctuation/CJK characters."""

    name = "approx"
    _PIECE = re.compile(r"[A-Za-z0-9_]{1,5}|[^\sA-Za-z0-9_]")

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in self._PIECE.finditer(text)]


def load_tokenizer(loader: Callable[[], Any] | None = None) -> Tokenizer:
    """tiktoken's `o200k_base` if `loader()` (default: tiktoken itself) succeeds, else `ApproxTokenizer` with a warning."""
    try:
        if loader is None:
            import tiktoken

            return TiktokenTokenizer(tiktoken.get_encoding(ENCODING))
        return TiktokenTokenizer(loader())
    except Exception as exc:  # noqa: BLE001  (missing package, no network, corrupted cache: all mean "use the approximation")
        logger.warning("Real tokens unavailable (%s); chunk sizes use an approximate tokenizer, so chunk boundaries can differ from a run with network access.", exc)
        return ApproxTokenizer()


def _load_tokenizer() -> Tokenizer:
    return load_tokenizer()


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    return _load_tokenizer()


def count_tokens(text: str) -> int:
    return len(get_tokenizer().spans(text))


class TokenCounter:
    """Token counts over character ranges of one text, from a single tokenization (a token counts where it starts)."""

    def __init__(self, text: str, tokenizer: Tokenizer | None = None):
        self.spans = (tokenizer or get_tokenizer()).spans(text)
        self._starts = [start for start, _ in self.spans]

    def __call__(self, start: int, end: int) -> int:
        return bisect_left(self._starts, end) - bisect_left(self._starts, start)

    def start_of_last(self, start: int, end: int, n: int) -> int | None:
        """Character offset where the last `n` tokens of `[start, end)` begin, or None when the range holds fewer."""
        high = bisect_left(self._starts, end)
        low = high - n
        return self.spans[low][0] if n > 0 and low >= bisect_left(self._starts, start) else None
